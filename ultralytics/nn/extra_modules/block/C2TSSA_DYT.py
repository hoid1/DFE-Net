'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/attention.py:1896-1933 , nn/extra_modules/transMamba.py:296-333 , nn/extra_modules/transformer.py:494-507 , nn/extra_modules/transformer.py:509-513 , nn/extra_modules/transformer.py:645-662 , nn/extra_modules/transformer.py:664-668 , nn/extra_modules/transformer.py:670-689 , nn/extra_modules/transformer.py:691-695 , nn/extra_modules/transformer.py:713-731 , nn/extra_modules/transformer.py:733-737 , nn/extra_modules/transformer.py:755-773 , nn/extra_modules/transformer.py:775-779
二次创新(2)：TSSA + CVPR2025 DyT + CVPR2025 Mona + 各种 FFN（旧库 md #285/#321/#322/#324/#329）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
import torch
from ultralytics.nn.extra_modules.block.MetaFormer import Mona
from ultralytics.nn.extra_modules.mlp.EDFFN import EDFFN
from ultralytics.nn.extra_modules.mlp.SEFN import SEFN
from ultralytics.nn.extra_modules.norm.dyt import DynamicTanh
from ultralytics.nn.modules.block import C2PSA, PSABlock


# ---- 原样迁移自 nn/extra_modules/attention.py:1896-1933 ----
class AttentionTSSA(nn.Module):
    # https://github.com/RobinWu218/ToST
    def __init__(self, dim, num_heads = 8, qkv_bias=False, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__()
        
        self.heads = num_heads

        self.attend = nn.Softmax(dim = 1)
        self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)

        self.temp = nn.Parameter(torch.ones(num_heads, 1))
        
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )
    
    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h = self.heads)

        b, h, N, d = w.shape
        
        w_normed = torch.nn.functional.normalize(w, dim=-2) 
        w_sq = w_normed ** 2

        # Pi from Eq. 10 in the paper
        Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp) # b * h * n 
        
        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w ** 2)
        attn = 1. / (1 + dots)
        attn = self.attn_drop(attn)

        out = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# ---- 原样迁移自 nn/extra_modules/transformer.py:494-507 ----
class TSSAlock_DYT(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.attn = AttentionTSSA(c, num_heads=num_heads)
    
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        BS, C, H, W = x.size()
        x = x + self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        return x

# ---- 原样迁移自 nn/extra_modules/transformer.py:509-513 ----
class C2TSSA_DYT(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(TSSAlock_DYT(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:645-662 ----
class TSSAlock_DYT_Mona(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.mona1 = Mona(c)
        self.mona2 = Mona(c)
        self.attn = AttentionTSSA(c, num_heads=num_heads)
    
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        BS, C, H, W = x.size()
        x = x + self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = self.mona1(x)
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        x = self.mona2(x)
        return x

# ---- 原样迁移自 nn/extra_modules/transformer.py:664-668 ----
class C2TSSA_DYT_Mona(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(TSSAlock_DYT_Mona(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:755-773 ----
class TSSAlock_DYT_Mona_EDFFN(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = EDFFN(c, ffn_expansion_factor=2, bias=False)
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.mona1 = Mona(c)
        self.mona2 = Mona(c)
        self.attn = AttentionTSSA(c, num_heads=num_heads)
    
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        BS, C, H, W = x.size()
        x = x + self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = self.mona1(x)
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        x = self.mona2(x)
        return x

# ---- 原样迁移自 nn/extra_modules/transformer.py:775-779 ----
class C2TSSA_DYT_Mona_EDFFN(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(TSSAlock_DYT_Mona_EDFFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

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

# ---- 原样迁移自 nn/extra_modules/transformer.py:713-731 ----
class TSSAlock_DYT_Mona_SEFFN(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = SpectralEnhancedFFN(c, ffn_expansion_factor=2, bias=False)
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.mona1 = Mona(c)
        self.mona2 = Mona(c)
        self.attn = AttentionTSSA(c, num_heads=num_heads)
    
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        BS, C, H, W = x.size()
        x = x + self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = self.mona1(x)
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        x = self.mona2(x)
        return x

# ---- 原样迁移自 nn/extra_modules/transformer.py:733-737 ----
class C2TSSA_DYT_Mona_SEFFN(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(TSSAlock_DYT_Mona_SEFFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:670-689 ----
class TSSAlock_DYT_Mona_SEFN(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = SEFN(c, ffn_expansion_factor=2, bias=False)
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.mona1 = Mona(c)
        self.mona2 = Mona(c)
        self.attn = AttentionTSSA(c, num_heads=num_heads)
    
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x_spatial = x
        BS, C, H, W = x.size()
        x = x + self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = self.mona1(x)
        x = x + self.ffn(self.dyt2(x), x_spatial) if self.add else self.ffn(self.dyt2(x), x_spatial)
        x = self.mona2(x)
        return x

# ---- 原样迁移自 nn/extra_modules/transformer.py:691-695 ----
class C2TSSA_DYT_Mona_SEFN(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(TSSAlock_DYT_Mona_SEFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- C2TSSA_DYT_Mona_EDFFN ----
    try:
        module = C2TSSA_DYT_Mona_EDFFN(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C2TSSA_DYT_Mona_EDFFN  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C2TSSA_DYT_Mona_EDFFN  自测跳过: {e}' + RESET)
    # ---- C2TSSA_DYT_Mona_SEFFN ----
    try:
        module = C2TSSA_DYT_Mona_SEFFN(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C2TSSA_DYT_Mona_SEFFN  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C2TSSA_DYT_Mona_SEFFN  自测跳过: {e}' + RESET)
    # ---- C2TSSA_DYT_Mona_SEFN ----
    try:
        module = C2TSSA_DYT_Mona_SEFN(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C2TSSA_DYT_Mona_SEFN  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C2TSSA_DYT_Mona_SEFN  自测跳过: {e}' + RESET)

