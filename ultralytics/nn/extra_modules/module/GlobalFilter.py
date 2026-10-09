'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:12351-12362 , nn/extra_modules/block.py:12364-12377 , nn/extra_modules/block.py:12379-12383 , nn/extra_modules/block.py:12385-12388
二次创新(2)：GFNet 的 GlobalFilter 与 CSP 结构融合（旧库 md #265）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

from timm.layers import DropPath
import torch.nn as nn
import torch
from ultralytics.nn.extra_modules.mamba.TransMixer import LayerNorm
from ultralytics.nn.extra_modules.module.fasterblock import ConvolutionalGLU
from ultralytics.nn.modules.block import C3k, C3k2


# ---- 原样迁移自 nn/extra_modules/block.py:12351-12362 ----
class GlobalFilter(nn.Module):
    def __init__(self, dim, size):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, size, size // 2 + 1, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        _, c, a, b = x.size()
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(2, 3), norm='ortho')
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:12364-12377 ----
class GlobalFilterBlock(nn.Module):

    def __init__(self, dim, size, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.filter = GlobalFilter(dim, size=size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvolutionalGLU(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:12379-12383 ----
class C3k_GlobalFilter(C3k):
    def __init__(self, c1, c2, n=1, size=None, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GlobalFilterBlock(c_, size) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:12385-12388 ----
class C3k2_GlobalFilter(C3k2):
    def __init__(self, c1, c2, n=1, size=None, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_GlobalFilter(self.c, self.c, 2, size, shortcut, g) if c3k else GlobalFilterBlock(self.c, size) for _ in range(n))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- C3k_GlobalFilter ----
    try:
        module = C3k_GlobalFilter(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k_GlobalFilter  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k_GlobalFilter  自测跳过: {e}' + RESET)
    # ---- C3k2_GlobalFilter ----
    try:
        module = C3k2_GlobalFilter(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_GlobalFilter  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_GlobalFilter  自测跳过: {e}' + RESET)

