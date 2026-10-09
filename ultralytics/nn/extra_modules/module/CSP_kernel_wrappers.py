'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:12905-12909 , nn/extra_modules/block.py:12911-12914 , nn/extra_modules/block.py:5895-5899 , nn/extra_modules/block.py:5901-5904 , nn/extra_modules/block.py:6077-6081 , nn/extra_modules/block.py:6083-6086
二次创新(2)：C3k2 + Faster_Block_CGLU / EfficientViMBlock_CGLU / Star_Block 的 0526 原始包装类
（旧库 md #147/#292，以及 #154 用到的 C3k2_Star）。新版可用 C3k2_Block 组合语法复现，此处保留原类以保证 yaml 逐字复现。
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch

import torch.nn as nn
from ultralytics.nn.extra_modules.module.efficientVIM import EfficientViMBlock_CGLU
from ultralytics.nn.extra_modules.module.fasterblock import Faster_Block_CGLU
from ultralytics.nn.extra_modules.module.starblock import Star_Block
from ultralytics.nn.modules.block import C3k, C3k2


# ---- 原样迁移自 nn/extra_modules/block.py:12905-12909 ----
class C3k_EfficientVIM_CGLU(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(EfficientViMBlock_CGLU(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:12911-12914 ----
class C3k2_EfficientVIM_CGLU(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_EfficientVIM_CGLU(self.c, self.c, n, shortcut, g) if c3k else EfficientViMBlock_CGLU(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:5895-5899 ----
class C3k_Faster_CGLU(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Faster_Block_CGLU(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:5901-5904 ----
class C3k2_Faster_CGLU(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_Faster_CGLU(self.c, self.c, 2, shortcut, g) if c3k else Faster_Block_CGLU(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:6077-6081 ----
class C3k_Star(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Star_Block(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:6083-6086 ----
class C3k2_Star(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_Star(self.c, self.c, 2, shortcut, g) if c3k else Star_Block(self.c) for _ in range(n))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- C3k2_Faster_CGLU ----
    try:
        module = C3k2_Faster_CGLU(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_Faster_CGLU  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_Faster_CGLU  自测跳过: {e}' + RESET)
    # ---- C3k_Star ----
    try:
        module = C3k_Star(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k_Star  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k_Star  自测跳过: {e}' + RESET)
    # ---- C3k2_Star ----
    try:
        module = C3k2_Star(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_Star  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_Star  自测跳过: {e}' + RESET)

