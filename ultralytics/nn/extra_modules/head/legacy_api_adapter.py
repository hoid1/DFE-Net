'''
0526 版检测头 -> 2107(8.4.11) 版检测头「输出协议 / 基类」运行时适配层。

背景
----
2107 版把检测头的输出协议整个换掉了：

    0526 版   训练: list[Tensor(bs, no, h, w)]        推理: (y, list[...])
    2107 版   训练: dict(boxes=, scores=, feats=)     推理: (y, dict(...))

`ultralytics/utils/loss.py` 全部按 dict 取值（v8DetectionLoss.loss 里
`preds["boxes"].shape[0]`），0526 版检测头返回 list 就会报

    TypeError: list indices must be integers or slices, not str

同时 2107 版 `tasks.py` 用 `isinstance(m, Detect)` 判断是否要
  ① 前向一次量 stride  ② 调 m.bias_init()  ③ _apply 时搬运 anchors/strides。
0526 版检测头直接继承 nn.Module，这三件事会被静默跳过，
m.stride 一直是 torch.zeros(nl)，make_anchors 出来的全是 0，
即使不报错训练出来也是废的。

设计原则
--------
**_legacy_*.py 里迁移进来的检测头本体一个字节都不改**，
适配全部放在 head/<名>.py 这层薄壳里：

    class Detect_LSCD(LegacyHeadAPIMixin, _Detect_LSCD, LegacyDetectBase):
        ...

MRO 为  Detect_LSCD -> LegacyHeadAPIMixin -> _Detect_LSCD -> LegacyDetectBase
        -> Detect -> nn.Module

* LegacyHeadAPIMixin 排最前，只覆盖 forward，做一次输出格式转换；
* _Detect_LSCD 排第二，decode_bboxes / bias_init / kpts_decode 等全部仍走 0526 版实现；
* LegacyDetectBase 只提供 isinstance(m, Detect) 这一个身份，
  __init__ 被退化成 nn.Module.__init__，
  这样 0526 版检测头内部那句 super().__init__() 落到本类时不会误跑 Detect 的构造。

计算图、权重命名、参数量与 0526 版仍逐位一致：
boxes / scores 只是把 0526 版本来就要 cat + split 的那一步提前做了，
和 8.3.x 版 v8DetectionLoss 内部做的事完全相同。
'''

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.head import Detect

__all__ = ('LegacyDetectBase', 'LegacyHeadAPIMixin')


class LegacyDetectBase(Detect):
    """仅用于让 0526 版检测头通过 2107 版的 isinstance(m, Detect) 检查。"""

    def __init__(self, *args, **kwargs):
        # 故意不调用 Detect.__init__：0526 版检测头会自己建 cv2/cv3/dfl 等全部子模块
        nn.Module.__init__(self)


class LegacyHeadAPIMixin:
    """把 0526 版检测头输出的 list / tuple 转成 2107 版的 dict 协议。"""

    def _legacy_extra_keys(self) -> tuple:
        """0526 版 forward 在 x 之后额外返回的东西，按任务映射到 2107 版的 key。"""
        if hasattr(self, 'nm'):  # Segment_*   0526: return x, mc, p
            return ('mask_coefficient', 'proto')
        if hasattr(self, 'kpt_shape'):  # Pose_*     0526: return x, kpt
            return ('kpts',)
        if hasattr(self, 'ne'):  # OBB_*      0526: return x, angle
            return ('angle',)
        return ()  # Detect_*   0526: return x

    def _legacy_to_dict(self, out):
        """list[(bs, no, h, w)] 或 (list, extra...) -> dict(boxes, scores, feats, ...)"""
        if isinstance(out, dict):  # 已经是新协议（例如底层继承了 2107 版 Detect）
            return out
        if torch.is_tensor(out[0]):  # 纯检测头：out 本身就是各层特征
            feats, extras = list(out), ()
        else:  # 分割/姿态/旋转框：out = (list, extra...)
            feats, extras = list(out[0]), tuple(out[1:])

        bs = feats[0].shape[0]
        x_cat = torch.cat([xi.view(bs, self.no, -1) for xi in feats], 2)
        boxes, scores = x_cat.split((self.reg_max * 4, self.nc), 1)

        # NOTE: feats 只被 make_anchors / imgsz 用到，取的是 shape[2:]，
        #       直接把 0526 版的 x 传进去即可，不需要另存一份骨干特征。
        preds = {'boxes': boxes, 'scores': scores, 'feats': feats}
        for k, v in zip(self._legacy_extra_keys(), extras):
            preds[k] = v
        return preds

    def forward(self, x):
        out = super().forward(x)  # 0526 版原始 forward，未作任何修改
        if self.training:
            return self._legacy_to_dict(out)
        if getattr(self, 'export', False):
            return out  # 导出：0526 版已经返回纯 Tensor / (Tensor, proto)
        return out[0], self._legacy_to_dict(out[1])
