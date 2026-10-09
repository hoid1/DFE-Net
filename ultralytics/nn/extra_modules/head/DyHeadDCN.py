'''
D 组：DyHead + DCNv3 / DCNv4（旧库 md #89 等）。
注意：需先编译 compile_module/ops_dcnv3 与 compile_module/DCNv4_op，且依赖 mmcv/mmengine。

适配层说明
新版 parse_model 对所有检测头统一执行 args.extend([reg_max, end2end, ch])，
而 0526 版检测头签名为 (nc, ..., ch)，参数个数不匹配。
本文件仅在原类外面套一层构造函数，吸收 reg_max / end2end 两个参数后原样转调，
检测头本体位于 head/_legacy_dyheaddcn.py，一个字未改。
计算图、权重命名、参数量与 0526 版逐位一致。
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

from ultralytics.nn.extra_modules.head._legacy_dyheaddcn import (
    Detect_DyHeadWithDCNV3 as _Detect_DyHeadWithDCNV3,
    Detect_DyHeadWithDCNV4 as _Detect_DyHeadWithDCNV4,
)


class Detect_DyHeadWithDCNV3(_Detect_DyHeadWithDCNV3):
    def __init__(self, nc=80, hidc=256, block_num=2, _reg_max=16, _end2end=False, ch=()):
        super().__init__(nc, hidc, block_num, ch)


class Detect_DyHeadWithDCNV4(_Detect_DyHeadWithDCNV4):
    def __init__(self, nc=80, hidc=256, block_num=2, _reg_max=16, _end2end=False, ch=()):
        super().__init__(nc, hidc, block_num, ch)

if __name__ == '__main__':
    RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    chs = (64, 128, 256)
    feats = [torch.randn(1, c, 80 // (2 ** i), 80 // (2 ** i)).to(device) for i, c in enumerate(chs)]
    try:
        head = Detect_DyHeadWithDCNV3(nc=80, ch=chs).to(device)
        head.stride = torch.tensor([8., 16., 32.])
        head.train()
        head([f.clone() for f in feats])
        print(GREEN + f'Detect_DyHeadWithDCNV3  参数量={sum(p.numel() for p in head.parameters()):,}' + RESET)
    except Exception as e:
        print(RED + f'Detect_DyHeadWithDCNV3  自测跳过: {e}' + RESET)
    try:
        head = Detect_DyHeadWithDCNV4(nc=80, ch=chs).to(device)
        head.stride = torch.tensor([8., 16., 32.])
        head.train()
        head([f.clone() for f in feats])
        print(GREEN + f'Detect_DyHeadWithDCNV4  参数量={sum(p.numel() for p in head.parameters()):,}' + RESET)
    except Exception as e:
        print(RED + f'Detect_DyHeadWithDCNV4  自测跳过: {e}' + RESET)

