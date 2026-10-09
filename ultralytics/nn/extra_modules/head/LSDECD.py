'''
二次创新(1)：自研 LSCD + DEA-Net 的 DEConv（旧库 md #161）

适配层说明
新版 parse_model 对所有检测头统一执行 args.extend([reg_max, end2end, ch])，
而 0526 版检测头签名为 (nc, ..., ch)，参数个数不匹配。
本文件仅在原类外面套一层构造函数，吸收 reg_max / end2end 两个参数后原样转调，
检测头本体位于 head/_legacy_lsdecd.py，一个字未改。
计算图、权重命名、参数量与 0526 版逐位一致。
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

from ultralytics.nn.extra_modules.head._legacy_lsdecd import (
    Detect_LSDECD as _Detect_LSDECD,
    Segment_LSDECD as _Segment_LSDECD,
    Pose_LSDECD as _Pose_LSDECD,
    OBB_LSDECD as _OBB_LSDECD,
)


from ultralytics.nn.extra_modules.head.legacy_api_adapter import LegacyDetectBase, LegacyHeadAPIMixin


class Detect_LSDECD(LegacyHeadAPIMixin, _Detect_LSDECD, LegacyDetectBase):
    def __init__(self, nc=80, hidc=256, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, hidc, ch)


class Segment_LSDECD(LegacyHeadAPIMixin, _Segment_LSDECD, LegacyDetectBase):
    def __init__(self, nc=80, nm=32, npr=256, hidc=256, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, nm, npr, hidc, ch)


class Pose_LSDECD(LegacyHeadAPIMixin, _Pose_LSDECD, LegacyDetectBase):
    def __init__(self, nc=80, kpt_shape=(17, 3), hidc=256, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, kpt_shape, hidc, ch)


class OBB_LSDECD(LegacyHeadAPIMixin, _OBB_LSDECD, LegacyDetectBase):
    def __init__(self, nc=80, ne=1, hidc=256, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, ne, hidc, ch)

if __name__ == '__main__':
    RED, GREEN, ORANGE, RESET = "\033[91m", "\033[92m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    chs = (64, 128, 256)
    feats = [torch.randn(1, c, 80 // (2 ** i), 80 // (2 ** i)).to(device) for i, c in enumerate(chs)]
    try:
        head = Detect_LSDECD(nc=80, hidc=256, ch=chs).to(device)
        head.stride = torch.tensor([8., 16., 32.])
        head.train()
        out = head([f.clone() for f in feats])
        print(GREEN + f'Detect_LSDECD  参数量={sum(p.numel() for p in head.parameters()):,}' + RESET)
    except Exception as e:
        print(RED + f'Detect_LSDECD  自测跳过: {e}' + RESET)
    try:
        head = Segment_LSDECD(nc=80, ch=chs).to(device)
        head.stride = torch.tensor([8., 16., 32.])
        head.train()
        out = head([f.clone() for f in feats])
        print(GREEN + f'Segment_LSDECD  参数量={sum(p.numel() for p in head.parameters()):,}' + RESET)
    except Exception as e:
        print(RED + f'Segment_LSDECD  自测跳过: {e}' + RESET)
    try:
        head = Pose_LSDECD(nc=80, ch=chs).to(device)
        head.stride = torch.tensor([8., 16., 32.])
        head.train()
        out = head([f.clone() for f in feats])
        print(GREEN + f'Pose_LSDECD  参数量={sum(p.numel() for p in head.parameters()):,}' + RESET)
    except Exception as e:
        print(RED + f'Pose_LSDECD  自测跳过: {e}' + RESET)
    try:
        head = OBB_LSDECD(nc=80, ch=chs).to(device)
        head.stride = torch.tensor([8., 16., 32.])
        head.train()
        out = head([f.clone() for f in feats])
        print(GREEN + f'OBB_LSDECD  参数量={sum(p.numel() for p in head.parameters()):,}' + RESET)
    except Exception as e:
        print(RED + f'OBB_LSDECD  自测跳过: {e}' + RESET)

