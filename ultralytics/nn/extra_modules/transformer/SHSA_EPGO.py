'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:8261-8270 , nn/extra_modules/block.py:8303-8346 , nn/extra_modules/block.py:8358-8366 , nn/extra_modules/block.py:8379-8383 , nn/extra_modules/block.py:8385-8388 , nn/extra_modules/block.py:8400-8408 , nn/extra_modules/block.py:8421-8425 , nn/extra_modules/block.py:8427-8430
二次创新(2)：CVPR2024 SHSA + EPGO，以及再叠加 TransNeXt CGLU（旧库 md #373/#374）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
import torch
from ultralytics.nn.extra_modules.mamba.TinyViM import Conv2d_BN
from ultralytics.nn.extra_modules.module.fasterblock import ConvolutionalGLU
from ultralytics.nn.extra_modules.transformer.SHSA import SHSA_GroupNorm
from ultralytics.nn.modules.block import C3k, C3k2

class Residual(nn.Module):
    """通用残差包装（迁移自旧版 nn/extra_modules/block.py，勿与 ultralytics.nn.modules.block.Residual 混淆）。"""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x



# ---- 原样迁移自 nn/extra_modules/block.py:8261-8270 ----
class SHSABlock_FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.SiLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:8303-8346 ----
class SHSA_EPGO(torch.nn.Module):
    """Single-Head Self-Attention"""
    def __init__(self, dim, qk_dim, pdim):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim

        self.pre_norm = SHSA_GroupNorm(pdim)

        self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)
        self.proj = torch.nn.Sequential(torch.nn.SiLU(), Conv2d_BN(
            dim, dim, bn_weight_init = 0))
        
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1),  # 输出动态 K
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim = 1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim = 1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        
        attn = (q.transpose(-2, -1) @ k) * self.scale

        dynamic_k = int(N * self.gate(x).view(B, -1).mean())
        mask = torch.zeros(B, N, N, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=dynamic_k, dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))

        attn = attn.softmax(dim = -1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        x = self.proj(torch.cat([x1, x2], dim = 1))

        return x

# ---- 原样迁移自 nn/extra_modules/block.py:8358-8366 ----
class SHSABlock_EPGO(torch.nn.Module):
    def __init__(self, dim, qk_dim=16, pdim=32):
        super().__init__()
        self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups = dim, bn_weight_init = 0))
        self.mixer = Residual(SHSA_EPGO(dim, qk_dim, pdim))
        self.ffn = Residual(SHSABlock_FFN(dim, int(dim * 2)))
    
    def forward(self, x):
        return self.ffn(self.mixer(self.conv(x)))

# ---- 原样迁移自 nn/extra_modules/block.py:8379-8383 ----
class C3k_SHSA_EPGO(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(SHSABlock_EPGO(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:8385-8388 ----
class C3k2_SHSA_EPGO(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_SHSA_EPGO(self.c, self.c, 2, shortcut, g) if c3k else SHSABlock_EPGO(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:8400-8408 ----
class SHSABlock_EPGO_CGLU(torch.nn.Module):
    def __init__(self, dim, qk_dim=16, pdim=32):
        super().__init__()
        self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups = dim, bn_weight_init = 0))
        self.mixer = Residual(SHSA_EPGO(dim, qk_dim, pdim))
        self.ffn = ConvolutionalGLU(dim, int(dim * 2))
    
    def forward(self, x):
        return self.ffn(self.mixer(self.conv(x)))

# ---- 原样迁移自 nn/extra_modules/block.py:8421-8425 ----
class C3k_SHSA_EPGO_CGLU(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(SHSABlock_EPGO_CGLU(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:8427-8430 ----
class C3k2_SHSA_EPGO_CGLU(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_SHSA_EPGO_CGLU(self.c, self.c, 2, shortcut, g) if c3k else SHSABlock_EPGO_CGLU(self.c) for _ in range(n))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- SHSABlock_EPGO_CGLU ----
    try:
        module = SHSABlock_EPGO_CGLU(in_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'SHSABlock_EPGO_CGLU  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'SHSABlock_EPGO_CGLU  自测跳过: {e}' + RESET)
    # ---- C3k_SHSA_EPGO_CGLU ----
    try:
        module = C3k_SHSA_EPGO_CGLU(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k_SHSA_EPGO_CGLU  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k_SHSA_EPGO_CGLU  自测跳过: {e}' + RESET)
    # ---- C3k2_SHSA_EPGO_CGLU ----
    try:
        module = C3k2_SHSA_EPGO_CGLU(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_SHSA_EPGO_CGLU  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_SHSA_EPGO_CGLU  自测跳过: {e}' + RESET)

