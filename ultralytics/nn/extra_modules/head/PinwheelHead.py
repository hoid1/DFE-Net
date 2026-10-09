"""0527 版检测头 PinwheelHead 的适配层（新式接口版）。

本体位于 ultralytics/nn/modules/Head/PinwheelHead.py，逐字保留，未作任何修改。

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

  本体每层的计算：cv2[i](x) 与 cv3[i](x)

计算图与 0527 版逐位一致，参数量相同，权重命名因为包了一层 Sequential 会变，
但这只影响加载旧权重，从头训练不受影响。

yaml 中请引用 Detect_PinwheelHead，不要直接引用本体类 PinwheelHead。
"""

import torch.nn as nn

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.Head.PinwheelHead import PinwheelHead as _Body

__all__ = ("Detect_PinwheelHead",)


class Detect_PinwheelHead(Detect):
    """PinwheelHead 的新式接口适配层。"""

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, reg_max, end2end, ch)
        body = _Body(nc, ch)          # 0527 本体，只用它造出来的分支模块
        self.cv2 = body.cv2
        self.cv3 = body.cv3

        if end2end:
            import copy
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)
