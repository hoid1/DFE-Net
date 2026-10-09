'''
本文件由 0526 版迁移而来 —— 检测头本体，逐字保留，未作任何修改
来源：nn/extra_modules/dyhead_prune.py:163-228 , nn/extra_modules/head.py:399-455
0526 版 DyHead-Prune 检测头。
注意：依赖 mmcv / mmengine。

请勿直接在 yaml 中引用本文件的类，yaml 请使用同目录 DyHeadPrune.py 中的同名适配类。
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn.functional as F
from mmcv.cnn import build_activation_layer
from mmengine.model import constant_init
import math
import torch.nn as nn
from mmengine.model import normal_init
import torch
from ultralytics.nn.extra_modules.head.DyHead import DyDCNv2, DyReLU
from ultralytics.nn.modules.block import DFL
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.utils.tal import dist2bbox, make_anchors


# ---- 原样迁移自 nn/extra_modules/dyhead_prune.py:163-228 ----
class DyHeadBlock_Prune(nn.Module):
    """DyHead Block with three types of attention.
    HSigmoid arguments in default act_cfg follow official code, not paper.
    https://github.com/microsoft/DynamicHead/blob/master/dyhead/dyrelu.py
    """

    def __init__(self,
                 in_channels,
                 norm_type='GN',
                 zero_init_offset=True,
                 act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)):
        super().__init__()
        self.zero_init_offset = zero_init_offset
        # (offset_x, offset_y, mask) * kernel_size_y * kernel_size_x
        self.offset_and_mask_dim = 3 * 3 * 3
        self.offset_dim = 2 * 3 * 3

        if norm_type == 'GN':
            norm_dict = dict(type='GN', num_groups=16, requires_grad=True)
        elif norm_type == 'BN':
            norm_dict = dict(type='BN', requires_grad=True)
        
        self.spatial_conv_high = DyDCNv2(in_channels, in_channels, norm_cfg=norm_dict)
        self.spatial_conv_mid = DyDCNv2(in_channels, in_channels)
        self.spatial_conv_low = DyDCNv2(in_channels, in_channels, stride=2)
        self.spatial_conv_offset = nn.Conv2d(
            in_channels, self.offset_and_mask_dim, 3, padding=1)
        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, 1, 1),
            nn.ReLU(inplace=True), build_activation_layer(act_cfg))
        self.task_attn_module = DyReLU(in_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)
        if self.zero_init_offset:
            constant_init(self.spatial_conv_offset, 0)

    def forward(self, x, level):
        """Forward function."""
        # calculate offset and mask of DCNv2 from middle-level feature
        offset_and_mask = self.spatial_conv_offset(x[level])
        offset = offset_and_mask[:, :self.offset_dim, :, :]
        mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()

        mid_feat = self.spatial_conv_mid(x[level], offset, mask)
        sum_feat = mid_feat * self.scale_attn_module(mid_feat)
        summed_levels = 1
        if level > 0:
            low_feat = self.spatial_conv_low(x[level - 1], offset, mask)
            sum_feat += low_feat * self.scale_attn_module(low_feat)
            summed_levels += 1
        if level < len(x) - 1:
            # this upsample order is weird, but faster than natural order
            # https://github.com/microsoft/DynamicHead/issues/25
            high_feat = F.interpolate(
                self.spatial_conv_high(x[level + 1], offset, mask),
                size=x[level].shape[-2:],
                mode='bilinear',
                align_corners=True)
            sum_feat += high_feat * self.scale_attn_module(high_feat)
            summed_levels += 1

        return self.task_attn_module(sum_feat / summed_levels)

# ---- 原样迁移自 nn/extra_modules/head.py:399-455 ----
class Detect_DyHead_Prune(nn.Module):
    """YOLOv8 Detect head with DyHead for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, hidc=256, block_num=2, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.conv = nn.ModuleList(nn.Sequential(Conv(x, hidc, 1)) for x in ch)
        self.dyhead = DyHeadBlock_Prune(hidc)
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(hidc, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for _ in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(nn.Sequential(DWConv(hidc, hidc, 3), Conv(hidc, c3, 1)), nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)), nn.Conv2d(c3, self.nc, 1)) for _ in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        new_x = []
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])
        for i in range(self.nl):
            new_x.append(self.dyhead(x, i))
        x = new_x
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
