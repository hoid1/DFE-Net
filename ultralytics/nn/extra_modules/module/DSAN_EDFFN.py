'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:13822-13826 , nn/extra_modules/block.py:13828-13831 , nn/extra_modules/dsan.py:147-183
二次创新(2)：DSA 可变形空间注意力 + CVPR2025 EDFFN（旧库 md #340）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

from timm.layers import DropPath
import math
import torch.nn as nn
import torch
from timm.layers import trunc_normal_
from ultralytics.nn.extra_modules.conv_module.DSA import DSA
from ultralytics.nn.extra_modules.mlp.EDFFN import EDFFN
from ultralytics.nn.modules.block import C3k, C3k2


# ---- 原样迁移自 nn/extra_modules/dsan.py:147-183 ----
class DSAN_EDFFN(nn.Module):
    def __init__(self, dim, kernel_size=7, dw_kernel_size=5, stride=1, dilation=1, group=1, 
                 mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = DSA(dim, kernel_size, dw_kernel_size, stride, dilation, group)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = EDFFN(dim, mlp_ratio)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:13822-13826 ----
class C3k_DSAN_EDFFN(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DSAN_EDFFN(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:13828-13831 ----
class C3k2_DSAN_EDFFN(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DSAN_EDFFN(self.c, self.c, 2, shortcut, g) if c3k else DSAN_EDFFN(self.c) for _ in range(n))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- DSAN_EDFFN ----
    try:
        module = DSAN_EDFFN(in_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'DSAN_EDFFN  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'DSAN_EDFFN  自测跳过: {e}' + RESET)
    # ---- C3k_DSAN_EDFFN ----
    try:
        module = C3k_DSAN_EDFFN(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k_DSAN_EDFFN  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k_DSAN_EDFFN  自测跳过: {e}' + RESET)
    # ---- C3k2_DSAN_EDFFN ----
    try:
        module = C3k2_DSAN_EDFFN(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_DSAN_EDFFN  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_DSAN_EDFFN  自测跳过: {e}' + RESET)

