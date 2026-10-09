'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:3529-3546 , nn/extra_modules/block.py:3548-3552 , nn/extra_modules/block.py:3554-3557
二次创新(2)：DWRSeg 的 DWR + UniRepLKNet 的 DilatedReparamBlock（旧库 md #74）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
import torch
from ultralytics.nn.extra_modules.block.RepHMS import DilatedReparamBlock
from ultralytics.nn.modules.block import C3k, C3k2
from ultralytics.nn.modules.conv import Conv


# ---- 原样迁移自 nn/extra_modules/block.py:3529-3546 ----
class DWR_DRB(nn.Module):
    def __init__(self, dim, act=True) -> None:
        super().__init__()

        self.conv_3x3 = Conv(dim, dim // 2, 3, act=act)
        
        self.conv_3x3_d1 = Conv(dim // 2, dim, 3, d=1, act=act)
        self.conv_3x3_d3 = DilatedReparamBlock(dim // 2, 5)
        self.conv_3x3_d5 = DilatedReparamBlock(dim // 2, 7)
        
        self.conv_1x1 = Conv(dim * 2, dim, k=1, act=act)
        
    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)
        x1, x2, x3 = self.conv_3x3_d1(conv_3x3), self.conv_3x3_d3(conv_3x3), self.conv_3x3_d5(conv_3x3)
        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv_1x1(x_out) + x
        return x_out

# ---- 原样迁移自 nn/extra_modules/block.py:3548-3552 ----
class C3k_DWR_DRB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DWR_DRB(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:3554-3557 ----
class C3k2_DWR_DRB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DWR_DRB(self.c, self.c, 2, shortcut, g) if c3k else DWR_DRB(self.c) for _ in range(n))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- DWR_DRB ----
    try:
        module = DWR_DRB(in_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'DWR_DRB  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'DWR_DRB  自测跳过: {e}' + RESET)
    # ---- C3k_DWR_DRB ----
    try:
        module = C3k_DWR_DRB(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k_DWR_DRB  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k_DWR_DRB  自测跳过: {e}' + RESET)
    # ---- C3k2_DWR_DRB ----
    try:
        module = C3k2_DWR_DRB(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_DWR_DRB  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_DWR_DRB  自测跳过: {e}' + RESET)

