'''
本文件由 0526 版迁移而来 —— 检测头本体，逐字保留，未作任何修改
来源：nn/extra_modules/head.py:1362-1367
0526 版 NMS-Free 检测头（旧库 md #156，仿 YOLOv10 双重标签分配）。

请勿直接在 yaml 中引用本文件的类，yaml 请使用同目录 NMSFree.py 中的同名适配类。
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import copy
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import v10Detect


# ---- 原样迁移自 nn/extra_modules/head.py:1362-1367 ----
class Detect_NMSFree(v10Detect):
    def __init__(self, nc=80, ch=...):
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.one2one_cv3 = copy.deepcopy(self.cv3)
