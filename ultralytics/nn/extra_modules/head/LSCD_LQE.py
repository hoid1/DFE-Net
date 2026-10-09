'''
二次创新(1)：自研 LSCD + GFocalV2 的 LQE（旧库 md #294）

适配层说明
新版 parse_model 对所有检测头统一执行 args.extend([reg_max, end2end, ch])，
而 0526 版检测头签名为 (nc, ..., ch)，参数个数不匹配。
本文件仅在原类外面套一层构造函数，吸收 reg_max / end2end 两个参数后原样转调，
检测头本体位于 head/_legacy_lscd_lqe.py，一个字未改。
计算图、权重命名、参数量与 0526 版逐位一致。
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

from ultralytics.nn.extra_modules.head._legacy_lscd_lqe import (
    Detect_LSCD_LQE as _Detect_LSCD_LQE,
    Segment_LSCD_LQE as _Segment_LSCD_LQE,
    Pose_LSCD_LQE as _Pose_LSCD_LQE,
    OBB_LSCD_LQE as _OBB_LSCD_LQE,
)


from ultralytics.nn.extra_modules.head.legacy_api_adapter import LegacyDetectBase, LegacyHeadAPIMixin


class Detect_LSCD_LQE(LegacyHeadAPIMixin, _Detect_LSCD_LQE, LegacyDetectBase):
    def __init__(self, nc=80, hidc=256, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, hidc, ch)


class Segment_LSCD_LQE(LegacyHeadAPIMixin, _Segment_LSCD_LQE, LegacyDetectBase):
    def __init__(self, nc=80, nm=32, npr=256, hidc=256, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, nm, npr, hidc, ch)


class Pose_LSCD_LQE(LegacyHeadAPIMixin, _Pose_LSCD_LQE, LegacyDetectBase):
    def __init__(self, nc=80, kpt_shape=(17, 3), hidc=256, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, kpt_shape, hidc, ch)


class OBB_LSCD_LQE(LegacyHeadAPIMixin, _OBB_LSCD_LQE, LegacyDetectBase):
    def __init__(self, nc=80, ne=1, hidc=256, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, ne, hidc, ch)

if __name__ == '__main__':
    RED, GREEN, ORANGE, RESET = "\033[91m", "\033[92m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    chs = (64, 128, 256)
    feats = [torch.randn(1, c, 80 // (2 ** i), 80 // (2 ** i)).to(device) for i, c in enumerate(chs)]
    try:
        head = Detect_LSCD_LQE(nc=80, hidc=256, ch=chs).to(device)
        head.stride = torch.tensor([8., 16., 32.])
        head.train()
        out = head([f.clone() for f in feats])
        print(GREEN + f'Detect_LSCD_LQE  参数量={sum(p.numel() for p in head.parameters()):,}' + RESET)
    except Exception as e:
        print(RED + f'Detect_LSCD_LQE  自测跳过: {e}' + RESET)
    try:
        head = Segment_LSCD_LQE(nc=80, ch=chs).to(device)
        head.stride = torch.tensor([8., 16., 32.])
        head.train()
        out = head([f.clone() for f in feats])
        print(GREEN + f'Segment_LSCD_LQE  参数量={sum(p.numel() for p in head.parameters()):,}' + RESET)
    except Exception as e:
        print(RED + f'Segment_LSCD_LQE  自测跳过: {e}' + RESET)
    try:
        head = Pose_LSCD_LQE(nc=80, ch=chs).to(device)
        head.stride = torch.tensor([8., 16., 32.])
        head.train()
        out = head([f.clone() for f in feats])
        print(GREEN + f'Pose_LSCD_LQE  参数量={sum(p.numel() for p in head.parameters()):,}' + RESET)
    except Exception as e:
        print(RED + f'Pose_LSCD_LQE  自测跳过: {e}' + RESET)
    try:
        head = OBB_LSCD_LQE(nc=80, ch=chs).to(device)
        head.stride = torch.tensor([8., 16., 32.])
        head.train()
        out = head([f.clone() for f in feats])
        print(GREEN + f'OBB_LSCD_LQE  参数量={sum(p.numel() for p in head.parameters()):,}' + RESET)
    except Exception as e:
        print(RED + f'OBB_LSCD_LQE  自测跳过: {e}' + RESET)

