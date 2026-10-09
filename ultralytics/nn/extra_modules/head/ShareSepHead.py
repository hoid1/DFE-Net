"""0527 版检测头 ShareSepHead 的适配层（新式接口版）。

本体位于 ultralytics/nn/modules/Head/ShareSepHead.py，逐字保留，未作任何修改。

为什么要重写这个适配层
----------------------
旧版适配层只做了一件事：吸收 parse_model 追加的 reg_max / end2end。
但 0527 版检测头是 ultralytics v8.3 时代的写法，training 时 forward 返回的是
**一个列表**，每级一个 [B, 4*reg_max+nc, H, W] 的拼接张量；
而本迁移库已经是新式接口，v8DetectionLoss 期望的是
**一个字典** dict(boxes=..., scores=..., feats=...)，于是训练一开始就报：

    TypeError: list indices must be integers or slices, not str
    （loss.py 里 preds["boxes"] 拿到的是 list）

还有第二个更隐蔽的问题：0527 检测头是 nn.Module 而非 Detect 子类，
DetectionModel.__init__ 里 `isinstance(m, Detect)` 判定为假，
于是 self.stride 被留在默认的 [32]，bias_init() 也不会被调用。
即使 loss 格式修好了，训练也是错的。

本适配层的做法
--------------
继承本迁移库的 Detect，拿到全部新式管线（forward / forward_head / _inference /
bias_init / stride 计算），再把 0527 本体的分支模块按原有顺序组装成
逐层的 box_head / cls_head 塞进 self.cv2 / self.cv3。

  本体每层的计算：cv2_[i](cv2_2[i](cv2_1[i](x))) 与 cv3_[i](cv3_2[i](cv3_1[i](x)))，其中 cv2_2/cv3_2 三尺度共享

计算图与 0527 版逐位一致，参数量相同，权重命名因为包了一层 Sequential 会变，
但这只影响加载旧权重，从头训练不受影响。

yaml 中请引用 Detect_ShareSepHead，不要直接引用本体类 ShareSepHead。
"""

import torch.nn as nn

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.Head.ShareSepHead import ShareSepHead as _Body

__all__ = ("Detect_ShareSepHead",)


class Detect_ShareSepHead(Detect):
    """ShareSepHead 的新式接口适配层。"""

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, reg_max, end2end, ch)
        body = _Body(nc, ch)          # 0527 本体，只用它造出来的分支模块
        # 本体在每次 forward 里把 cv2_2[i] 的权重强行赋成 cv2_2[0] 的，
        # 即三个尺度共享这一层。这里直接让三个尺度引用同一个模块，语义相同且更干净。
        share2, share3 = body.cv2_2[0], body.cv3_2[0]
        self.cv2 = nn.ModuleList(
            nn.Sequential(body.cv2_1[i], share2, body.cv2_[i]) for i in range(self.nl)
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(body.cv3_1[i], share3, body.cv3_[i]) for i in range(self.nl)
        )

        if end2end:
            import copy
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)
