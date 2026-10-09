'''
0526 版 NMS-Free 检测头（旧库 md #156，仿 YOLOv10 双重标签分配）。

适配层说明
新版 parse_model 对所有检测头统一执行 args.extend([reg_max, end2end, ch])，
而 0526 版检测头签名为 (nc, ..., ch)，参数个数不匹配。
本文件仅在原类外面套一层构造函数，吸收 reg_max / end2end 两个参数后原样转调，
检测头本体位于 head/_legacy_nmsfree.py，一个字未改。
计算图、权重命名、参数量与 0526 版逐位一致。
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

from ultralytics.nn.extra_modules.head._legacy_nmsfree import (
    Detect_NMSFree as _Detect_NMSFree,
)


class Detect_NMSFree(_Detect_NMSFree):
    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, ch)


