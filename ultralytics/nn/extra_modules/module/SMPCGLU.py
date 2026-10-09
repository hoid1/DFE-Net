'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:6260-6279 , nn/extra_modules/block.py:6281-6285 , nn/extra_modules/block.py:6287-6290
二次创新(2)：SMPConv + TransNeXt 的 ConvolutionalGLU（旧库 md #162）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch

from timm.layers import DropPath
import torch.nn as nn
from ultralytics.nn.extra_modules.conv_module.SMPConv import SMPConv
from ultralytics.nn.extra_modules.module.fasterblock import ConvolutionalGLU
from ultralytics.nn.modules.block import C3k, C3k2
from ultralytics.nn.modules.conv import Conv


# ---- 原样迁移自 nn/extra_modules/block.py:6260-6279 ----
class SMPCGLU(nn.Module):
    def __init__(self,
                 inc,
                 kernel_size,
                 drop_path=0.1,
                 n_points=4
                 ):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = ConvolutionalGLU(inc)
        self.smpconv = nn.Sequential(
            SMPConv(inc, kernel_size, n_points, 1, padding=kernel_size // 2, groups=1),
            Conv.default_act
        )

    def forward(self, x):
        shortcut = x
        x = self.smpconv(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:6281-6285 ----
class C3k_SMPCGLU(C3k):
    def __init__(self, c1, c2, n=1, kernel_size=13, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(SMPCGLU(c_, kernel_size) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:6287-6290 ----
class C3k2_SMPCGLU(C3k2):
    def __init__(self, c1, c2, n=1, kernel_size=13, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_SMPCGLU(self.c, self.c, 2, kernel_size, shortcut, g) if c3k else SMPCGLU(self.c, kernel_size) for _ in range(n))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- C3k_SMPCGLU ----
    try:
        module = C3k_SMPCGLU(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k_SMPCGLU  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k_SMPCGLU  自测跳过: {e}' + RESET)
    # ---- C3k2_SMPCGLU ----
    try:
        module = C3k2_SMPCGLU(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_SMPCGLU  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_SMPCGLU  自测跳过: {e}' + RESET)

