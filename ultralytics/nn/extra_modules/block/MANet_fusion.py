'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:10650-10653 , nn/extra_modules/block.py:10655-10658 , nn/extra_modules/block.py:10660-10663 , nn/extra_modules/block.py:15332-15337 , nn/extra_modules/block.py:15350-15353
二次创新(2)：Hyper-YOLO 的 MANet 分别融合 FasterBlock / FasterBlock+CGLU / StarBlock / GCConv
（旧库 md #235/#236/#237/#250/#383）。新版有通用 MANet(module=...)，此处保留 0526 原始包装类以保证 yaml 逐字复现。
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch

import torch.nn as nn
from ultralytics.nn.extra_modules.block.MANet import MANet
from ultralytics.nn.extra_modules.conv_module.gcconv import GCConv
from ultralytics.nn.extra_modules.module.fasterblock import Faster_Block, Faster_Block_CGLU
from ultralytics.nn.extra_modules.module.starblock import Star_Block
from ultralytics.nn.modules.block import Bottleneck


# ---- 原样迁移自 nn/extra_modules/block.py:15332-15337 ----
class Bottleneck_GCConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = GCConv(c1, c1, 3)
        self.cv2 = GCConv(c2, c2, 3)

# ---- 原样迁移自 nn/extra_modules/block.py:10650-10653 ----
class MANet_FasterBlock(MANet):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, p, kernel_size, g, e)
        self.m = nn.ModuleList(Faster_Block(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:10655-10658 ----
class MANet_FasterCGLU(MANet):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, p, kernel_size, g, e)
        self.m = nn.ModuleList(Faster_Block_CGLU(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:15350-15353 ----
class MANet_GCConv(MANet):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, p, kernel_size, g, e)
        self.m = nn.ModuleList(Bottleneck_GCConv(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:10660-10663 ----
class MANet_Star(MANet):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, p, kernel_size, g, e)
        self.m = nn.ModuleList(Star_Block(self.c) for _ in range(n))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- MANet_FasterCGLU ----
    try:
        module = MANet_FasterCGLU(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'MANet_FasterCGLU  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'MANet_FasterCGLU  自测跳过: {e}' + RESET)
    # ---- MANet_GCConv ----
    try:
        module = MANet_GCConv(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'MANet_GCConv  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'MANet_GCConv  自测跳过: {e}' + RESET)
    # ---- MANet_Star ----
    try:
        module = MANet_Star(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'MANet_Star  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'MANet_Star  自测跳过: {e}' + RESET)

