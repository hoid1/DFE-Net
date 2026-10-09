'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:6062-6075 , nn/extra_modules/block.py:6088-6092 , nn/extra_modules/block.py:6094-6097
二次创新(2)：StarNet 的 StarBlock + PKINet 的 CAA（旧库 md #153）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch

import torch.nn as nn
from ultralytics.nn.extra_modules.module.PKIBlock import CAA
from ultralytics.nn.extra_modules.module.starblock import Star_Block
from ultralytics.nn.modules.block import C3k, C3k2


# ---- 原样迁移自 nn/extra_modules/block.py:6062-6075 ----
class Star_Block_CAA(Star_Block):
    def __init__(self, dim, mlp_ratio=3, drop_path=0):
        super().__init__(dim, mlp_ratio, drop_path)
        
        self.attention = CAA(mlp_ratio * dim)
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(self.attention(x)))
        x = input + self.drop_path(x)
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:6088-6092 ----
class C3k_Star_CAA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Star_Block_CAA(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:6094-6097 ----
class C3k2_Star_CAA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_Star_CAA(self.c, self.c, 2, shortcut, g) if c3k else Star_Block_CAA(self.c) for _ in range(n))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- Star_Block_CAA ----
    try:
        module = Star_Block_CAA(in_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'Star_Block_CAA  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'Star_Block_CAA  自测跳过: {e}' + RESET)
    # ---- C3k_Star_CAA ----
    try:
        module = C3k_Star_CAA(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k_Star_CAA  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k_Star_CAA  自测跳过: {e}' + RESET)
    # ---- C3k2_Star_CAA ----
    try:
        module = C3k2_Star_CAA(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_Star_CAA  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_Star_CAA  自测跳过: {e}' + RESET)

