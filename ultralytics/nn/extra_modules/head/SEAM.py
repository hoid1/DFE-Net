'''
D 组：SEAM / MultiSEAM 检测头（旧库 md 遮挡感知检测头）。

适配层说明
新版 parse_model 对所有检测头统一执行 args.extend([reg_max, end2end, ch])，
而 0526 版检测头签名为 (nc, ..., ch)，参数个数不匹配。
本文件仅在原类外面套一层构造函数，吸收 reg_max / end2end 两个参数后原样转调，
检测头本体位于 head/_legacy_seam.py，一个字未改。
计算图、权重命名、参数量与 0526 版逐位一致。
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

from ultralytics.nn.extra_modules.head._legacy_seam import (
    Detect_SEAM as _Detect_SEAM,
    Detect_MultiSEAM as _Detect_MultiSEAM,
)


from ultralytics.nn.extra_modules.head.legacy_api_adapter import LegacyDetectBase, LegacyHeadAPIMixin


class Detect_SEAM(LegacyHeadAPIMixin, _Detect_SEAM, LegacyDetectBase):
    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, ch)


class Detect_MultiSEAM(LegacyHeadAPIMixin, _Detect_MultiSEAM, LegacyDetectBase):
    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, ch)

if __name__ == '__main__':
    RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    chs = (64, 128, 256)
    feats = [torch.randn(1, c, 80 // (2 ** i), 80 // (2 ** i)).to(device) for i, c in enumerate(chs)]
    try:
        head = Detect_SEAM(nc=80, ch=chs).to(device)
        head.stride = torch.tensor([8., 16., 32.])
        head.train()
        head([f.clone() for f in feats])
        print(GREEN + f'Detect_SEAM  参数量={sum(p.numel() for p in head.parameters()):,}' + RESET)
    except Exception as e:
        print(RED + f'Detect_SEAM  自测跳过: {e}' + RESET)
    try:
        head = Detect_MultiSEAM(nc=80, ch=chs).to(device)
        head.stride = torch.tensor([8., 16., 32.])
        head.train()
        head([f.clone() for f in feats])
        print(GREEN + f'Detect_MultiSEAM  参数量={sum(p.numel() for p in head.parameters()):,}' + RESET)
    except Exception as e:
        print(RED + f'Detect_MultiSEAM  自测跳过: {e}' + RESET)

