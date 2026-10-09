'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:16374-16378 , nn/extra_modules/block.py:16380-16383 , nn/extra_modules/block.py:2776-2785 , nn/extra_modules/block.py:2787-2791 , nn/extra_modules/block.py:2793-2796
A1 组：0526 版 CSP 包装壳，内核在 2107 版中已存在，此处仅迁入壳本体。
签名 (c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True) 与 base_modules + repeat_modules 契约一致，
旧 yaml 可一字不改直接使用。（等价写法：C3k2_Block, [ch, {'module': <内核>}, ...]）
本组包含：C3k2_DAttention, C3k2_GLCDM
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch

import torch.nn as nn
from ultralytics.nn.extra_modules.transformer.DAttention import DAttention
from ultralytics.nn.extra_modules.transformer.GLCDM import GLCDM
from ultralytics.nn.modules.block import Bottleneck, C3k, C3k2


# ---- 原样迁移自 nn/extra_modules/block.py:2776-2785 ----
class Bottleneck_DAttention(Bottleneck):
    """Standard bottleneck with DAttention."""

    def __init__(self, c1, c2, fmapsize, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.attention = DAttention(c2, fmapsize)
    
    def forward(self, x):
        return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))

# ---- 原样迁移自 nn/extra_modules/block.py:2787-2791 ----
class C3k_DAttention(C3k):
    def __init__(self, c1, c2, n=1, fmapsize=None, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DAttention(c_, c_, fmapsize, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:2793-2796 ----
class C3k2_DAttention(C3k2):
    def __init__(self, c1, c2, n=1, fmapsize=None, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DAttention(self.c, self.c, 2, fmapsize, shortcut, g) if c3k else Bottleneck_DAttention(self.c, self.c, fmapsize, shortcut, g) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:16374-16378 ----
class C3k_GLCDM(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GLCDM(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:16380-16383 ----
class C3k2_GLCDM(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_GLCDM(self.c, self.c, 2, shortcut, g) if c3k else GLCDM(self.c) for _ in range(n))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- C3k2_DAttention ----
    try:
        module = C3k2_DAttention(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_DAttention  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_DAttention  自测跳过: {e}' + RESET)
    # ---- C3k_GLCDM ----
    try:
        module = C3k_GLCDM(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k_GLCDM  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k_GLCDM  自测跳过: {e}' + RESET)
    # ---- C3k2_GLCDM ----
    try:
        module = C3k2_GLCDM(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_GLCDM  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_GLCDM  自测跳过: {e}' + RESET)

