'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/srconvnet.py:220-241 , nn/extra_modules/transMamba.py:296-333 , nn/extra_modules/transformer.py:1003-1007 , nn/extra_modules/transformer.py:1110-1114 , nn/extra_modules/transformer.py:1116-1120 , nn/extra_modules/transformer.py:1225-1229 , nn/extra_modules/transformer.py:1231-1235 , nn/extra_modules/transformer.py:1268-1272 , nn/extra_modules/transformer.py:1274-1278 , nn/extra_modules/transformer.py:1369-1373 , nn/extra_modules/transformer.py:1375-1379 , nn/extra_modules/transformer.py:1385-1389 , nn/extra_modules/transformer.py:1391-1395 , nn/extra_modules/transformer.py:1401-1405 , nn/extra_modules/transformer.py:1407-1411 , nn/extra_modules/transformer.py:1417-1421 , nn/extra_modules/transformer.py:1423-1427 , nn/extra_modules/transformer.py:1433-1437 , nn/extra_modules/transformer.py:1439-1443 , nn/extra_modules/transformer.py:1445-1449 , nn/extra_modules/transformer.py:1451-1455 , nn/extra_modules/transformer.py:1462-1466 , nn/extra_modules/transformer.py:1468-1472 , nn/extra_modules/transformer.py:1478-1482 , nn/extra_modules/transformer.py:1484-1488 , nn/extra_modules/transformer.py:1494-1498 , nn/extra_modules/transformer.py:1500-1504 , nn/extra_modules/transformer.py:476-486 , nn/extra_modules/transformer.py:488-492 , nn/extra_modules/transformer.py:540-544 , nn/extra_modules/transformer.py:546-550 , nn/extra_modules/transformer.py:587-591 , nn/extra_modules/transformer.py:593-597 , nn/extra_modules/transformer.py:603-613 , nn/extra_modules/transformer.py:615-619 , nn/extra_modules/transformer.py:625-637 , nn/extra_modules/transformer.py:639-643 , nn/extra_modules/transformer.py:701-705 , nn/extra_modules/transformer.py:707-711 , nn/extra_modules/transformer.py:743-747 , nn/extra_modules/transformer.py:749-753 , nn/extra_modules/transformer.py:909-979 , nn/extra_modules/transformer.py:981-985 , nn/extra_modules/transformer.py:987-991 , nn/extra_modules/transformer.py:997-1001
0526 版 C2PSA 系列注意力/FFN 变体。签名 (c1, c2, n=1, e=0.5)，走 base_modules + repeat_modules。
包含：C2PSA_AFFN, C2PSA_BinaryAttn, C2PSA_CGLU, C2PSA_CGTA, C2PSA_CirculantAtt, C2PSA_DML, C2PSA_DWMMSA, C2PSA_DYT, C2PSA_EDFFN, C2PSA_EGSA, C2PSA_EPGO, C2PSA_FMFFN, C2PSA_LCGA, C2PSA_LRSA, C2PSA_MALA, C2PSA_Mona, C2PSA_SEFFN, C2PSA_SEFN, C2PSA_SWSA, C2PSA_WCA, C2PSA_WDAM
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
from ultralytics.nn.extra_modules.mamba.TransMixer import LayerNorm
from ultralytics.nn.extra_modules.mlp.AFFN import AFFN
from ultralytics.nn.extra_modules.mlp.DML import DyConv
from ultralytics.nn.extra_modules.mlp.EDFFN import EDFFN
from ultralytics.nn.extra_modules.mlp.FMFFN import FMFFN
from ultralytics.nn.extra_modules.mlp.SEFN import SEFN
from ultralytics.nn.extra_modules.module.fasterblock import ConvolutionalGLU
from ultralytics.nn.extra_modules.norm.dyt import DynamicTanh
from ultralytics.nn.extra_modules.transformer.BinaryAttention import BinaryAttention
from ultralytics.nn.extra_modules.transformer.CGTA import CGTA
from ultralytics.nn.extra_modules.transformer.CirculantAttention import CirculantAttention
from ultralytics.nn.extra_modules.transformer.DWM_MSA import DWM_MSA
from ultralytics.nn.extra_modules.transformer.EGSA import EfficientGlobalSA
from ultralytics.nn.extra_modules.transformer.LCGA import LCGA
from ultralytics.nn.extra_modules.transformer.LRSA import LRSA
from ultralytics.nn.extra_modules.transformer.MALA import MALA
from ultralytics.nn.extra_modules.transformer.SWSA import PatchSA
from ultralytics.nn.extra_modules.transformer.WDAM import WDAM
from ultralytics.nn.extra_modules.transformer.wca import WCA
from ultralytics.nn.modules.block import C2PSA, PSABlock
from ultralytics.nn.modules.conv import Conv


# ---- 原样迁移自 nn/extra_modules/transformer.py:909-979 ----
class Attention_EPGO(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads # 512 // 8 = 64
        self.key_dim = int(self.head_dim * attn_ratio) # 64 * 0.5 = 32
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads # 32 * 8 = 256
        h = dim + nh_kd * 2 # 512 + 256 * 2 = 1024
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1),  # 输出动态 K
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x) # B, dim + nh_kd * 2, H, W
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split( # 1024 / 8 = 128
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        # q: B, 8, 32, HW
        # k: B, 8, 32, HW
        # v: B, 8, 64, HW

        attn = (q.transpose(-2, -1) @ k) * self.scale

        dynamic_k = int(N * self.gate(x).view(B, -1).mean())
        mask = torch.zeros(B, self.num_heads, N, N, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=dynamic_k, dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))

        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

# ---- 原样迁移自 nn/extra_modules/transformer.py:1478-1482 ----
class PSABlock_AFFN(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = AFFN(c, c * 2, c)

# ---- 原样迁移自 nn/extra_modules/transformer.py:1484-1488 ----
class C2PSA_AFFN(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_AFFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:1401-1405 ----
class PSABlock_BinaryAttn(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.attn = BinaryAttention(c)

# ---- 原样迁移自 nn/extra_modules/transformer.py:1407-1411 ----
class C2PSA_BinaryAttn(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_BinaryAttn(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:587-591 ----
class PSABlock_CGLU(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = ConvolutionalGLU(c, c * 2, c)

# ---- 原样迁移自 nn/extra_modules/transformer.py:593-597 ----
class C2PSA_CGLU(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_CGLU(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:1445-1449 ----
class PSABlock_CGTA(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.attn = CGTA(c)

# ---- 原样迁移自 nn/extra_modules/transformer.py:1451-1455 ----
class C2PSA_CGTA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_CGTA(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:1462-1466 ----
class PSABlock_CirculantAtt(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.attn = CirculantAttention(c)

# ---- 原样迁移自 nn/extra_modules/transformer.py:1468-1472 ----
class C2PSA_CirculantAtt(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_CirculantAtt(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/srconvnet.py:220-241 ----
class MixFFN(nn.Module):
    def __init__(self, dim, num_kernels=16):
        super().__init__()
        self.proj_in = nn.Conv2d(dim, dim * 2, 1)
        self.conv1 = DyConv(dim, kernel_size=5, groups=dim, num_kernels=num_kernels)
        self.conv2 = DyConv(dim, kernel_size=7, groups=dim, num_kernels=num_kernels)
        self.proj_out = nn.Conv2d(dim * 2, dim, 1)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.act(self.proj_in(x))
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.act(self.conv1(x1)).unsqueeze(dim=2)
        x2 = self.act(self.conv2(x2)).unsqueeze(dim=2)
        x = torch.cat([x1, x2], dim=2)
        x = rearrange(x, 'b c g h w -> b (c g) h w')
        x = self.proj_out(x) 
        x = x + shortcut
        return x

# ---- 原样迁移自 nn/extra_modules/transformer.py:997-1001 ----
class PSABlock_DML(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = MixFFN(c, 16)

# ---- 原样迁移自 nn/extra_modules/transformer.py:1003-1007 ----
class C2PSA_DML(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_DML(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:1385-1389 ----
class PSABlock_DWMMSA(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.attn = DWM_MSA(c)

# ---- 原样迁移自 nn/extra_modules/transformer.py:1391-1395 ----
class C2PSA_DWMMSA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_DWMMSA(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:476-486 ----
class PSABlock_DYT(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)
    
    def forward(self, x):
        x = x + self.attn(self.dyt1(x)) if self.add else self.attn(self.dyt1(x))
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        return x

# ---- 原样迁移自 nn/extra_modules/transformer.py:488-492 ----
class C2PSA_DYT(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_DYT(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:743-747 ----
class PSABlock_EDFFN(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = EDFFN(c, 2, False)

# ---- 原样迁移自 nn/extra_modules/transformer.py:749-753 ----
class C2PSA_EDFFN(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_EDFFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:1268-1272 ----
class PSABlock_EGSA(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.attn = EfficientGlobalSA(c)

# ---- 原样迁移自 nn/extra_modules/transformer.py:1274-1278 ----
class C2PSA_EGSA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_EGSA(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:981-985 ----
class PSABlock_EPGO(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.attn = Attention_EPGO(c, attn_ratio=attn_ratio, num_heads=num_heads)

# ---- 原样迁移自 nn/extra_modules/transformer.py:987-991 ----
class C2PSA_EPGO(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_EPGO(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:540-544 ----
class PSABlock_FMFFN(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = FMFFN(c, c * 2, c)

# ---- 原样迁移自 nn/extra_modules/transformer.py:546-550 ----
class C2PSA_FMFFN(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_FMFFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:1433-1437 ----
class PSABlock_LCGA(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.attn = LCGA(c)

# ---- 原样迁移自 nn/extra_modules/transformer.py:1439-1443 ----
class C2PSA_LCGA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_LCGA(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:1110-1114 ----
class PSABlock_LRSA(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.attn = LRSA(c, num_heads=num_heads)

# ---- 原样迁移自 nn/extra_modules/transformer.py:1116-1120 ----
class C2PSA_LRSA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_LRSA(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:1225-1229 ----
class PSABlock_MALA(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.attn = MALA(c, num_heads=num_heads)

# ---- 原样迁移自 nn/extra_modules/transformer.py:1231-1235 ----
class C2PSA_MALA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_MALA(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:625-637 ----
class PSABlock_Mona(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.mona1 = Mona(c)
        self.mona2 = Mona(c)
    
    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = self.mona1(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        x = self.mona2(x)
        return x

# ---- 原样迁移自 nn/extra_modules/transformer.py:639-643 ----
class C2PSA_Mona(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_Mona(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

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

# ---- 原样迁移自 nn/extra_modules/transformer.py:701-705 ----
class PSABlock_SEFFN(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = SpectralEnhancedFFN(c, 2, False)

# ---- 原样迁移自 nn/extra_modules/transformer.py:707-711 ----
class C2PSA_SEFFN(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_SEFFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:603-613 ----
class PSABlock_SEFN(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.ffn = SEFN(c, ffn_expansion_factor=2, bias=False)
    
    def forward(self, x):
        x_spatial = x
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x, x_spatial) if self.add else self.ffn(x, x_spatial)
        return x

# ---- 原样迁移自 nn/extra_modules/transformer.py:615-619 ----
class C2PSA_SEFN(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_SEFN(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:1369-1373 ----
class PSABlock_SWSA(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.attn = PatchSA(c)

# ---- 原样迁移自 nn/extra_modules/transformer.py:1375-1379 ----
class C2PSA_SWSA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_SWSA(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:1417-1421 ----
class PSABlock_WCA(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.attn = WCA(c)

# ---- 原样迁移自 nn/extra_modules/transformer.py:1423-1427 ----
class C2PSA_WCA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_WCA(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/transformer.py:1494-1498 ----
class PSABlock_WDAM(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.attn = WDAM(c)

# ---- 原样迁移自 nn/extra_modules/transformer.py:1500-1504 ----
class C2PSA_WDAM(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_WDAM(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- C2PSA_SWSA ----
    try:
        module = C2PSA_SWSA(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C2PSA_SWSA  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C2PSA_SWSA  自测跳过: {e}' + RESET)
    # ---- C2PSA_WCA ----
    try:
        module = C2PSA_WCA(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C2PSA_WCA  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C2PSA_WCA  自测跳过: {e}' + RESET)
    # ---- C2PSA_WDAM ----
    try:
        module = C2PSA_WDAM(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C2PSA_WDAM  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C2PSA_WDAM  自测跳过: {e}' + RESET)

