'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:12787-12798 , nn/extra_modules/block.py:12800-12810 , nn/extra_modules/block.py:12812-12825 , nn/extra_modules/block.py:12827-12837 , nn/extra_modules/block.py:13087-13100 , nn/extra_modules/block.py:13102-13112 , nn/extra_modules/block.py:13434-13449 , nn/extra_modules/block.py:13451-13461
二次创新(2)：YOLOv12 的 A2C2f 融合 CVPR2025 DyT + CGLU / DFFN / FMFFN / Mona
（旧库 md #288/#289/#302/#320，yolo12 系列）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch

import torch.nn as nn
from ultralytics.nn.extra_modules.block.MetaFormer import Mona
from ultralytics.nn.extra_modules.mlp.DFFN import DFFN
from ultralytics.nn.extra_modules.mlp.FMFFN import FMFFN
from ultralytics.nn.extra_modules.module.fasterblock import ConvolutionalGLU
from ultralytics.nn.extra_modules.norm.dyt import DynamicTanh
from ultralytics.nn.modules.block import A2C2f, ABlock, C3k


# ---- 原样迁移自 nn/extra_modules/block.py:12787-12798 ----
class ABlock_CGLU_DYT(ABlock):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__(dim, num_heads, mlp_ratio, area)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.dyt1 = DynamicTanh(normalized_shape=dim, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=dim, channels_last=False)
        self.mlp = ConvolutionalGLU(dim, mlp_hidden_dim)
    
    def forward(self, x):
        x = x + self.attn(self.dyt1(x))
        return x + self.mlp(self.dyt2(x))

# ---- 原样迁移自 nn/extra_modules/block.py:12800-12810 ----
class A2C2f_CGLU_DYT(A2C2f):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock_CGLU_DYT(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

# ---- 原样迁移自 nn/extra_modules/block.py:12812-12825 ----
class ABlock_DFFN_DYT(ABlock):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__(dim, num_heads, mlp_ratio, area)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.dyt1 = DynamicTanh(normalized_shape=dim, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=dim, channels_last=False)
        self.mlp = DFFN(dim, mlp_hidden_dim)
    
    def forward(self, x):
        """Forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        _, C, H, W = x.size()
        x = x + self.attn(self.dyt1(x))
        return x + self.mlp(self.dyt2(x).flatten(2).permute(0, 2, 1), H, W).permute(0, 2, 1).view([-1, C, H, W]).contiguous()

# ---- 原样迁移自 nn/extra_modules/block.py:12827-12837 ----
class A2C2f_DFFN_DYT(A2C2f):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock_DFFN_DYT(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

# ---- 原样迁移自 nn/extra_modules/block.py:13434-13449 ----
class ABlock_DFFN_DYT_Mona(ABlock):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__(dim, num_heads, mlp_ratio, area)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.dyt1 = DynamicTanh(normalized_shape=dim, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=dim, channels_last=False)
        self.mona1 = Mona(dim)
        self.mona2 = Mona(dim)
        self.mlp = DFFN(dim, mlp_hidden_dim)
    
    def forward(self, x):
        """Forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        _, C, H, W = x.size()
        x = self.mona1(x + self.attn(self.dyt1(x)))
        return x + self.mona2(self.mlp(self.dyt2(x).flatten(2).permute(0, 2, 1), H, W).permute(0, 2, 1).view([-1, C, H, W]).contiguous())

# ---- 原样迁移自 nn/extra_modules/block.py:13451-13461 ----
class A2C2f_DFFN_DYT_Mona(A2C2f):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock_DFFN_DYT_Mona(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

# ---- 原样迁移自 nn/extra_modules/block.py:13087-13100 ----
class ABlock_FMFFN_DYT(ABlock):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__(dim, num_heads, mlp_ratio, area)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.dyt1 = DynamicTanh(normalized_shape=dim, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=dim, channels_last=False)
        self.mlp = FMFFN(dim, mlp_hidden_dim, dim)
    
    def forward(self, x):
        """Forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        _, C, H, W = x.size()
        x = x + self.attn(self.dyt1(x))
        return x + self.mlp(self.dyt2(x))

# ---- 原样迁移自 nn/extra_modules/block.py:13102-13112 ----
class A2C2f_FMFFN_DYT(A2C2f):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock_FMFFN_DYT(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- A2C2f_DFFN_DYT ----
    try:
        module = A2C2f_DFFN_DYT(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'A2C2f_DFFN_DYT  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'A2C2f_DFFN_DYT  自测跳过: {e}' + RESET)
    # ---- A2C2f_DFFN_DYT_Mona ----
    try:
        module = A2C2f_DFFN_DYT_Mona(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'A2C2f_DFFN_DYT_Mona  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'A2C2f_DFFN_DYT_Mona  自测跳过: {e}' + RESET)
    # ---- A2C2f_FMFFN_DYT ----
    try:
        module = A2C2f_FMFFN_DYT(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'A2C2f_FMFFN_DYT  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'A2C2f_FMFFN_DYT  自测跳过: {e}' + RESET)

