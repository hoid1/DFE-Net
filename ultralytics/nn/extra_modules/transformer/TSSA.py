'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:12520-12557 , nn/extra_modules/block.py:12559-12565 , nn/extra_modules/block.py:12567-12572 , nn/extra_modules/metaformer.py:157-171 , nn/extra_modules/metaformer.py:276-319
二次创新(2)：Token Statistics Self-Attention 与 MetaFormer 融合（旧库 md #267）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

from timm.layers import DropPath
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
from einops import rearrange
import torch
from ultralytics.nn.extra_modules.head.LSPCD import Scale
from ultralytics.nn.extra_modules.mamba.TransMixer import Mlp
from ultralytics.nn.modules.block import C3k, C3k2


# ---- 原样迁移自 nn/extra_modules/block.py:12520-12557 ----
class AttentionTSSA_Meta(nn.Module):
    # https://github.com/RobinWu218/ToST
    def __init__(self, dim, num_heads = 8, qkv_bias=False, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__()
        
        self.heads = num_heads

        self.attend = nn.Softmax(dim = 1)
        self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)

        self.temp = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )
    
    def forward(self, x):
        w = rearrange(self.qkv(x), 'b x y (h d) -> b h x y d', h = self.heads)

        # b, h, N, d = w.shape
        
        w_normed = torch.nn.functional.normalize(w, dim=-2) 
        w_sq = w_normed ** 2

        # Pi from Eq. 10 in the paper
        Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp) # b * h * n 
        
        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w ** 2)
        attn = 1. / (1 + dots)
        attn = self.attn_drop(attn)

        out = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)

        out = rearrange(out, 'b h x y d -> b x y (h d)')
        return self.to_out(out)

# ---- 原样迁移自 nn/extra_modules/metaformer.py:157-171 ----
class LayerNormWithoutBias(nn.Module):
    """
    Equal to partial(LayerNormGeneral, bias=False) but faster, 
    because it directly utilizes otpimized F.layer_norm
    """
    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.bias = None
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)

# ---- 原样迁移自 nn/extra_modules/metaformer.py:276-319 ----
class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """
    def __init__(self, dim,
                 token_mixer=nn.Identity, mlp=Mlp,
                 norm_layer=partial(LayerNormWithoutBias, eps=1e-6),
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None
                 ):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x.permute(0, 3, 1, 2)

# ---- 原样迁移自 nn/extra_modules/block.py:12559-12565 ----
class C3k_TSSA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MetaFormerBlock(
                dim=c_, token_mixer=AttentionTSSA_Meta,
            ) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:12567-12572 ----
class C3k2_TSSA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_TSSA(self.c, self.c, n, shortcut, g) if c3k else MetaFormerBlock(
                dim=self.c, token_mixer=AttentionTSSA_Meta,
            ) for _ in range(n))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- MetaFormerBlock ----
    try:
        module = MetaFormerBlock(in_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'MetaFormerBlock  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'MetaFormerBlock  自测跳过: {e}' + RESET)
    # ---- C3k_TSSA ----
    try:
        module = C3k_TSSA(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k_TSSA  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k_TSSA  自测跳过: {e}' + RESET)
    # ---- C3k2_TSSA ----
    try:
        module = C3k2_TSSA(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_TSSA  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_TSSA  自测跳过: {e}' + RESET)

