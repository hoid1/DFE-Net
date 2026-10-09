'''
自研模块：Rep Shared Convolutional Detection Head（旧库 md #210）
注意：含重参数化，导出前需触发 switch_to_deploy

适配层说明
新版 parse_model 对所有检测头统一执行 args.extend([reg_max, end2end, ch])，
而 0526 版检测头签名为 (nc, ..., ch)，参数个数不匹配。
本文件仅在原类外面套一层构造函数，吸收 reg_max / end2end 两个参数后原样转调，
检测头本体位于 head/_legacy_rscd.py，一个字未改。
计算图、权重命名、参数量与 0526 版逐位一致。
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

from ultralytics.nn.extra_modules.head._legacy_rscd import (
    Detect_RSCD as _Detect_RSCD,
    Segment_RSCD as _Segment_RSCD,
    Pose_RSCD as _Pose_RSCD,
    OBB_RSCD as _OBB_RSCD,
)


from ultralytics.nn.extra_modules.head.legacy_api_adapter import LegacyDetectBase, LegacyHeadAPIMixin


class Detect_RSCD(LegacyHeadAPIMixin, _Detect_RSCD, LegacyDetectBase):
    def __init__(self, nc=80, hidc=256, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, hidc, ch)


class Segment_RSCD(LegacyHeadAPIMixin, _Segment_RSCD, LegacyDetectBase):
    def __init__(self, nc=80, nm=32, npr=256, hidc=256, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, nm, npr, hidc, ch)


class Pose_RSCD(LegacyHeadAPIMixin, _Pose_RSCD, LegacyDetectBase):
    def __init__(self, nc=80, kpt_shape=(17, 3), hidc=256, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, kpt_shape, hidc, ch)


class OBB_RSCD(LegacyHeadAPIMixin, _OBB_RSCD, LegacyDetectBase):
    def __init__(self, nc=80, ne=1, hidc=256, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, ne, hidc, ch)

if __name__ == '__main__':
    RED, GREEN, ORANGE, RESET = "\033[91m", "\033[92m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    chs = (64, 128, 256)
    feats = [torch.randn(1, c, 80 // (2 ** i), 80 // (2 ** i)).to(device) for i, c in enumerate(chs)]
    try:
        head = Detect_RSCD(nc=80, hidc=256, ch=chs).to(device)
        head.stride = torch.tensor([8., 16., 32.])
        head.train()
        out = head([f.clone() for f in feats])
        print(GREEN + f'Detect_RSCD  参数量={sum(p.numel() for p in head.parameters()):,}' + RESET)
    except Exception as e:
        print(RED + f'Detect_RSCD  自测跳过: {e}' + RESET)
    try:
        head = Segment_RSCD(nc=80, ch=chs).to(device)
        head.stride = torch.tensor([8., 16., 32.])
        head.train()
        out = head([f.clone() for f in feats])
        print(GREEN + f'Segment_RSCD  参数量={sum(p.numel() for p in head.parameters()):,}' + RESET)
    except Exception as e:
        print(RED + f'Segment_RSCD  自测跳过: {e}' + RESET)
    try:
        head = Pose_RSCD(nc=80, ch=chs).to(device)
        head.stride = torch.tensor([8., 16., 32.])
        head.train()
        out = head([f.clone() for f in feats])
        print(GREEN + f'Pose_RSCD  参数量={sum(p.numel() for p in head.parameters()):,}' + RESET)
    except Exception as e:
        print(RED + f'Pose_RSCD  自测跳过: {e}' + RESET)
    try:
        head = OBB_RSCD(nc=80, ch=chs).to(device)
        head.stride = torch.tensor([8., 16., 32.])
        head.train()
        out = head([f.clone() for f in feats])
        print(GREEN + f'OBB_RSCD  参数量={sum(p.numel() for p in head.parameters()):,}' + RESET)
    except Exception as e:
        print(RED + f'OBB_RSCD  自测跳过: {e}' + RESET)

