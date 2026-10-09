'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/ast.py:288-315 , nn/extra_modules/block.py:12578-12583 , nn/extra_modules/block.py:12585-12595 , nn/extra_modules/block.py:12601-12612 , nn/extra_modules/block.py:12614-12624 , nn/extra_modules/block.py:12630-12641 , nn/extra_modules/block.py:12643-12653 , nn/extra_modules/block.py:12694-12699 , nn/extra_modules/block.py:12701-12711 , nn/extra_modules/block.py:12762-12772 , nn/extra_modules/block.py:12774-12785 , nn/extra_modules/block.py:13068-13073 , nn/extra_modules/block.py:13075-13085 , nn/extra_modules/block.py:13382-13391 , nn/extra_modules/block.py:13393-13403 , nn/extra_modules/block.py:13409-13419 , nn/extra_modules/block.py:13421-13432 , nn/extra_modules/block.py:13540-13548 , nn/extra_modules/block.py:13550-13560 , nn/extra_modules/block.py:13577-13585 , nn/extra_modules/block.py:13587-13597 , nn/extra_modules/transMamba.py:296-333
0526 版 A2C2f 系列变体（yolo12）。签名与新版 A2C2f 同形，走 base_modules + repeat_modules。
包含：A2C2f_CGLU, A2C2f_DFFN, A2C2f_DYT, A2C2f_EDFFN, A2C2f_FMFFN, A2C2f_FRFN, A2C2f_KAN, A2C2f_Mona, A2C2f_SEFFN, A2C2f_SEFN
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn.functional as F
import torch.nn as nn
import torch
from ultralytics.nn.extra_modules.block.MetaFormer import Mona
from ultralytics.nn.extra_modules.mlp.DFFN import DFFN
from ultralytics.nn.extra_modules.mlp.EDFFN import EDFFN
from ultralytics.nn.extra_modules.mlp.FMFFN import FMFFN
from ultralytics.nn.extra_modules.mlp.KAN import KAN
from ultralytics.nn.extra_modules.mlp.SEFN import SEFN
from ultralytics.nn.extra_modules.module.fasterblock import ConvolutionalGLU
from ultralytics.nn.extra_modules.norm.dyt import DynamicTanh
from ultralytics.nn.modules.block import A2C2f, ABlock, C3k


# ---- 原样迁移自 nn/extra_modules/block.py:12578-12583 ----
class ABlock_CGLU(ABlock):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__(dim, num_heads, mlp_ratio, area)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvolutionalGLU(dim, mlp_hidden_dim)

# ---- 原样迁移自 nn/extra_modules/block.py:12585-12595 ----
class A2C2f_CGLU(A2C2f):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock_CGLU(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

# ---- 原样迁移自 nn/extra_modules/block.py:12630-12641 ----
class ABlock_DFFN(ABlock):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__(dim, num_heads, mlp_ratio, area)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DFFN(dim, mlp_hidden_dim)
    
    def forward(self, x):
        """Forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        _, C, H, W = x.size()
        x = x + self.attn(x)
        return x + self.mlp(x.flatten(2).permute(0, 2, 1), H, W).permute(0, 2, 1).view([-1, C, H, W]).contiguous()

# ---- 原样迁移自 nn/extra_modules/block.py:12643-12653 ----
class A2C2f_DFFN(A2C2f):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock_DFFN(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

# ---- 原样迁移自 nn/extra_modules/block.py:12762-12772 ----
class ABlock_DYT(ABlock):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__(dim, num_heads, mlp_ratio, area)

        self.dyt1 = DynamicTanh(normalized_shape=dim, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=dim, channels_last=False)
    
    def forward(self, x):
        """Forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(self.dyt1(x))
        return x + self.mlp(self.dyt2(x))

# ---- 原样迁移自 nn/extra_modules/block.py:12774-12785 ----
class A2C2f_DYT(A2C2f):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock_DYT(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

# ---- 原样迁移自 nn/extra_modules/block.py:13577-13585 ----
class ABlock_EDFFN(ABlock):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__(dim, num_heads, mlp_ratio, area)

        self.mlp = EDFFN(dim, mlp_ratio, False)
    
    def forward(self, x):
        x = x + self.attn(x)
        return x + self.mlp(x)

# ---- 原样迁移自 nn/extra_modules/block.py:13587-13597 ----
class A2C2f_EDFFN(A2C2f):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock_EDFFN(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

# ---- 原样迁移自 nn/extra_modules/block.py:13068-13073 ----
class ABlock_FMFFN(ABlock):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__(dim, num_heads, mlp_ratio, area)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FMFFN(dim, mlp_hidden_dim, dim)

# ---- 原样迁移自 nn/extra_modules/block.py:13075-13085 ----
class A2C2f_FMFFN(A2C2f):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock_FMFFN(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

# ---- 原样迁移自 nn/extra_modules/ast.py:288-315 ----
class FRFN2D(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Conv2d(dim, hidden_dim*2, 1),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv 
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        x1, x2,= torch.split(x, [self.dim_conv,self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        x = self.linear1(x)
        #gate mechanism
        x_1, x_2 = x.chunk(2,dim=1)
        x_1 = self.dwconv(x_1)
        x = x_1 * x_2
        
        x = self.linear2(x)
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:12694-12699 ----
class ABlock_FRFN(ABlock):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__(dim, num_heads, mlp_ratio, area)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FRFN2D(dim, mlp_hidden_dim)

# ---- 原样迁移自 nn/extra_modules/block.py:12701-12711 ----
class A2C2f_FRFN(A2C2f):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock_FRFN(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

# ---- 原样迁移自 nn/extra_modules/block.py:12601-12612 ----
class ABlock_KAN(ABlock):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__(dim, num_heads, mlp_ratio, area)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = KAN(dim, mlp_hidden_dim)
    
    def forward(self, x):
        """Forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        _, C, H, W = x.size()
        x = x + self.attn(x)
        return x + self.mlp(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()

# ---- 原样迁移自 nn/extra_modules/block.py:12614-12624 ----
class A2C2f_KAN(A2C2f):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock_KAN(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

# ---- 原样迁移自 nn/extra_modules/block.py:13409-13419 ----
class ABlock_Mona(ABlock):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__(dim, num_heads, mlp_ratio, area)

        self.mona1 = Mona(dim)
        self.mona2 = Mona(dim)
    
    def forward(self, x):
        """Forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = self.mona1(x + self.attn(x))
        return self.mona2(x + self.mlp(x))

# ---- 原样迁移自 nn/extra_modules/block.py:13421-13432 ----
class A2C2f_Mona(A2C2f):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock_Mona(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

# ---- 原样迁移自 nn/extra_modules/transMamba.py:296-333 ----
class SpectralEnhancedFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(SpectralEnhancedFFN, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=2, groups=hidden_features*2, bias=bias, dilation=2)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.fft_channel_weight = nn.Parameter(torch.randn((1, hidden_features * 2, 1, 1)))
        self.fft_channel_bias = nn.Parameter(torch.randn((1, hidden_features * 2, 1, 1)))

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw//factor+1)*factor-hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad
    def unpad(self, x, t_pad):
        hw = x.shape[-1]
        return x[...,t_pad[0]:hw-t_pad[1]]

    def forward(self, x):
        x_dtype = x.dtype
        x = self.project_in(x)
        x = self.dwconv(x)
        x, pad_w = self.pad(x,2)
        x = torch.fft.rfft2(x.float())
        x = self.fft_channel_weight * x + self.fft_channel_bias
#        x = torch.nn.functional.normalize(x, 1)
        x = torch.fft.irfft2(x)
        x = self.unpad(x, pad_w)
        x1, x2 = x.chunk(2, dim=1)
        
        x = F.silu(x1) * x2
        x = self.project_out(x.to(x_dtype))
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:13540-13548 ----
class ABlock_SEFFN(ABlock):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__(dim, num_heads, mlp_ratio, area)

        self.mlp = SpectralEnhancedFFN(dim, mlp_ratio, False)
    
    def forward(self, x):
        x = x + self.attn(x)
        return x + self.mlp(x)

# ---- 原样迁移自 nn/extra_modules/block.py:13550-13560 ----
class A2C2f_SEFFN(A2C2f):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock_SEFFN(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

# ---- 原样迁移自 nn/extra_modules/block.py:13382-13391 ----
class ABlock_SEFN(ABlock):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__(dim, num_heads, mlp_ratio, area)

        self.mlp = SEFN(dim, mlp_ratio, False)
    
    def forward(self, x):
        x_spatial = x
        x = x + self.attn(x)
        return x + self.mlp(x, x_spatial)

# ---- 原样迁移自 nn/extra_modules/block.py:13393-13403 ----
class A2C2f_SEFN(A2C2f):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock_SEFN(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- A2C2f_Mona ----
    try:
        module = A2C2f_Mona(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'A2C2f_Mona  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'A2C2f_Mona  自测跳过: {e}' + RESET)
    # ---- A2C2f_SEFFN ----
    try:
        module = A2C2f_SEFFN(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'A2C2f_SEFFN  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'A2C2f_SEFFN  自测跳过: {e}' + RESET)
    # ---- A2C2f_SEFN ----
    try:
        module = A2C2f_SEFN(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'A2C2f_SEFN  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'A2C2f_SEFN  自测跳过: {e}' + RESET)

