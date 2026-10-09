'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/attention.py:1683-1745 , nn/extra_modules/attention.py:1896-1933 , nn/extra_modules/transformer.py:102-106 , nn/extra_modules/transformer.py:108-112 , nn/extra_modules/transformer.py:243-254 , nn/extra_modules/transformer.py:256-266 , nn/extra_modules/transformer.py:366-377 , nn/extra_modules/transformer.py:379-389 , nn/extra_modules/transformer.py:395-406 , nn/extra_modules/transformer.py:408-419 , nn/extra_modules/transformer.py:425-436 , nn/extra_modules/transformer.py:438-448 , nn/extra_modules/transformer.py:78-82 , nn/extra_modules/transformer.py:84-88 , nn/extra_modules/transformer.py:880-891 , nn/extra_modules/transformer.py:893-903 , nn/extra_modules/transformer.py:90-94 , nn/extra_modules/transformer.py:96-100
B 组：0526 版 C2PSA 系注意力封装。签名 (c1, c2, n=1, e=0.5)，继承 C2PSA，
与 base_modules + repeat_modules 契约一致，旧 yaml 可一字不改。
包含：C2ASSA, C2BRA, C2CGA, C2DA, C2DPB, C2MSLA, C2Pola, C2TSSA
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
from einops import rearrange
import torch
from ultralytics.nn.extra_modules.transformer.AdaptiveSparseSA import AdaptiveSparseSA
from ultralytics.nn.extra_modules.transformer.CascadedGroupAttention import CascadedGroupAttention
from ultralytics.nn.extra_modules.transformer.DAttention import DAttention
from ultralytics.nn.extra_modules.transformer.DPBAttention import DPB_Attention
from ultralytics.nn.extra_modules.transformer.MSLA import MSLA
from ultralytics.nn.extra_modules.transformer.PolaLinearAttention import PolaLinearAttention
from ultralytics.nn.extra_modules.transformer.biformer import BiLevelRoutingAttention_nchw
from ultralytics.nn.modules.block import C2PSA, PSABlock


# ---- 原样迁移自 nn/extra_modules/transformer.py:425-436 ----
class ASSAlock(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        
        self.attn = AdaptiveSparseSA(c, num_heads=num_heads, sparseAtt=True)
    
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        BS, C, H, W = x.size()
        x = x + self.attn(x).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(x).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

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

# ---- 原样迁移自 nn/extra_modules/transformer.py:78-82 ----
class BRABlock(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        
        self.attn = BiLevelRoutingAttention_nchw(dim=c)

# ---- 原样迁移自 nn/extra_modules/transformer.py:438-448 ----
class C2ASSA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        
        self.m = nn.Sequential(*(ASSAlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
    
    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

# ---- 原样迁移自 nn/extra_modules/transformer.py:84-88 ----
class C2BRA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        
        self.m = nn.Sequential(*(BRABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/attention.py:1683-1745 ----
class LocalWindowAttention(torch.nn.Module):
    r""" Local Window Attention. CVPR2023-EfficientViT

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, dim, key_dim=16, num_heads=4,
                 attn_ratio=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution
        
        self.attn = CascadedGroupAttention(dim, key_dim, num_heads,
                                attn_ratio=attn_ratio, 
                                resolution=window_resolution,
                                kernels=kernels)

    def forward(self, x):
        B, C, H, W = x.shape
               
        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)
            pad_b = (self.window_resolution - H %
                     self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W %
                     self.window_resolution) % self.window_resolution
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            # window partition, BHWC -> B(nHh)(nWw)C -> BnHnWhwC -> (BnHnW)hwC -> (BnHnW)Chw
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C
            ).permute(0, 3, 1, 2)
            x = self.attn(x)
            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution,
                       C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

            x = x.permute(0, 3, 1, 2)

        return x

# ---- 原样迁移自 nn/extra_modules/transformer.py:90-94 ----
class CGABlock(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        
        self.attn = LocalWindowAttention(dim=c)

# ---- 原样迁移自 nn/extra_modules/transformer.py:96-100 ----
class C2CGA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        
        self.m = nn.Sequential(*(CGABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:102-106 ----
class DABlock(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        
        self.attn = DAttention(c, q_size=[20, 20])

# ---- 原样迁移自 nn/extra_modules/transformer.py:108-112 ----
class C2DA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        
        self.m = nn.Sequential(*(DABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:243-254 ----
class DPBlock(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        
        self.attn = DPB_Attention(c, group_size=[20, 20], num_heads=num_heads)
    
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        BS, C, H, W = x.size()
        x = x + self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

# ---- 原样迁移自 nn/extra_modules/transformer.py:256-266 ----
class C2DPB(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        
        self.m = nn.Sequential(*(DPBlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
    
    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

# ---- 原样迁移自 nn/extra_modules/transformer.py:880-891 ----
class MSLAlock(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        
        self.attn = MSLA(c, num_heads=num_heads)
    
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        BS, C, H, W = x.size()
        x = x + self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

# ---- 原样迁移自 nn/extra_modules/transformer.py:893-903 ----
class C2MSLA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        
        self.m = nn.Sequential(*(MSLAlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
    
    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

# ---- 原样迁移自 nn/extra_modules/transformer.py:366-377 ----
class Polalock(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        
        self.attn = PolaLinearAttention(c, hw=[20, 20], num_heads=num_heads)
    
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        BS, C, H, W = x.size()
        x = x + self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

# ---- 原样迁移自 nn/extra_modules/transformer.py:379-389 ----
class C2Pola(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        
        self.m = nn.Sequential(*(Polalock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
    
    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

# ---- 原样迁移自 nn/extra_modules/transformer.py:395-406 ----
class TSSAlock(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        
        self.attn = AttentionTSSA(c, num_heads=num_heads)
    
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        BS, C, H, W = x.size()
        x = x + self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

# ---- 原样迁移自 nn/extra_modules/transformer.py:408-419 ----
class C2TSSA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        
        self.m = nn.Sequential(*(TSSAlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
    
    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        BS, C, H, W = b.size()
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- C2MSLA ----
    try:
        module = C2MSLA(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C2MSLA  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C2MSLA  自测跳过: {e}' + RESET)
    # ---- C2Pola ----
    try:
        module = C2Pola(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C2Pola  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C2Pola  自测跳过: {e}' + RESET)
    # ---- C2TSSA ----
    try:
        module = C2TSSA(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C2TSSA  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C2TSSA  自测跳过: {e}' + RESET)

