'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:11925-11948 , nn/extra_modules/block.py:11950-11954 , nn/extra_modules/block.py:11956-11959 , nn/extra_modules/block.py:12051-12068 , nn/extra_modules/block.py:12070-12074 , nn/extra_modules/block.py:12076-12079 , nn/extra_modules/block.py:7516-7532 , nn/extra_modules/block.py:7534-7546 , nn/extra_modules/block.py:7548-7558 , nn/extra_modules/block.py:7560-7575 , nn/extra_modules/block.py:7577-7603 , nn/extra_modules/block.py:7606-7626 , nn/extra_modules/block.py:7628-7631 , nn/extra_modules/block.py:7644-7648 , nn/extra_modules/block.py:7650-7653 , nn/extra_modules/block.py:7789-7825 , nn/extra_modules/block.py:7827-7871 , nn/extra_modules/block.py:7873-7889 , nn/extra_modules/block.py:7891-7895 , nn/extra_modules/block.py:7898-7901 , nn/extra_modules/block.py:8390-8398 , nn/extra_modules/block.py:8410-8414 , nn/extra_modules/block.py:8416-8419 , nn/extra_modules/block.py:8436-8549 , nn/extra_modules/block.py:8551-8571 , nn/extra_modules/block.py:8624-8642 , nn/extra_modules/block.py:8655-8659 , nn/extra_modules/block.py:8661-8664 , nn/extra_modules/block.py:8782-8788 , nn/extra_modules/block.py:8790-8795 , nn/extra_modules/block.py:8797-8803 , nn/extra_modules/block.py:8805-8810 , nn/extra_modules/block.py:8812-8818 , nn/extra_modules/block.py:8820-8825 , nn/extra_modules/block.py:8827-8833 , nn/extra_modules/block.py:8835-8840 , nn/extra_modules/block.py:8842-8848 , nn/extra_modules/block.py:8850-8855 , nn/extra_modules/metaformer.py:157-171 , nn/extra_modules/metaformer.py:174-200 , nn/extra_modules/metaformer.py:203-217 , nn/extra_modules/metaformer.py:321-364 , nn/extra_modules/metaformer.py:49-86 , nn/extra_modules/metaformer.py:89-100
二次创新(2)：TransNeXt(CVPR2024) 的 ConvolutionalGLU 分别与 CAS-ViT AdditiveBlock / MSMHSA /
SHSA / SMAFormer / MetaFormer 四变体 / ELGCA / StripBlock 融合（旧库 md #186/#188/#193/#195/#203-#207/#255/#257）
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
from ultralytics.nn.extra_modules.block.MetaFormer import StarReLU
from ultralytics.nn.extra_modules.head.LSPCD import Scale
from ultralytics.nn.extra_modules.mamba.TinyViM import Conv2d_BN
from ultralytics.nn.extra_modules.module.elgca import LayerNorm
from ultralytics.nn.extra_modules.module.StripBlock import Strip_Attention
from ultralytics.nn.extra_modules.module.elgca import ELGCA
from ultralytics.nn.extra_modules.module.fasterblock import ConvolutionalGLU
from ultralytics.nn.extra_modules.module.mambaout import LayerNormGeneral
from ultralytics.nn.extra_modules.transformer.SHSA import SHSA
from ultralytics.nn.modules.block import C3k, C3k2

class Residual(nn.Module):
    """通用残差包装（迁移自旧版 nn/extra_modules/block.py，勿与 ultralytics.nn.modules.block.Residual 混淆）。"""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.transformer import LayerNorm2d


# ---- 原样迁移自 nn/extra_modules/block.py:7548-7558 ----
class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

# ---- 原样迁移自 nn/extra_modules/block.py:7534-7546 ----
class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

# ---- 原样迁移自 nn/extra_modules/block.py:7577-7603 ----
class AdditiveTokenMixer(nn.Module):
    """
    改变了proj函数的输入，不对q+k卷积，而是对融合之后的结果proj
    """
    def __init__(self, dim=512, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out

# ---- 原样迁移自 nn/extra_modules/block.py:7560-7575 ----
class LocalIntegration(nn.Module):
    """
    """
    def __init__(self, dim, ratio=1, act_layer=nn.ReLU, norm_layer=nn.GELU):
        super().__init__()
        mid_dim = round(ratio * dim)
        self.network = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1, 1, 0),
            norm_layer(mid_dim),
            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1, groups=mid_dim),
            act_layer(),
            nn.Conv2d(mid_dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.network(x)

# ---- 原样迁移自 nn/extra_modules/block.py:7516-7532 ----
class Mlp_CASVIT(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:7606-7626 ----
class AdditiveBlock(nn.Module):
    """
    """
    def __init__(self, dim, mlp_ratio=4., attn_bias=False, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.local_perception = LocalIntegration(dim, ratio=1, act_layer=act_layer, norm_layer=norm_layer)
        self.norm1 = norm_layer(dim)
        self.attn = AdditiveTokenMixer(dim, attn_bias=attn_bias, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_CASVIT(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.local_perception(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:7628-7631 ----
class AdditiveBlock_CGLU(AdditiveBlock):
    def __init__(self, dim, mlp_ratio=4, attn_bias=False, drop=0, drop_path=0, act_layer=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__(dim, mlp_ratio, attn_bias, drop, drop_path, act_layer, norm_layer)
        self.mlp = ConvolutionalGLU(dim)

# ---- 原样迁移自 nn/extra_modules/block.py:7644-7648 ----
class C3k_AdditiveBlock_CGLU(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(AdditiveBlock_CGLU(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:7650-7653 ----
class C3k2_AdditiveBlock_CGLU(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_AdditiveBlock_CGLU(self.c, self.c, 2, shortcut, g) if c3k else AdditiveBlock_CGLU(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/metaformer.py:49-86 ----
class MF_Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1
        
        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        
    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

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

# ---- 原样迁移自 nn/extra_modules/metaformer.py:321-364 ----
class MetaFormerCGLUBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """
    def __init__(self, dim,
                 token_mixer=nn.Identity, mlp=ConvolutionalGLU,
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
        self.mlp = mlp(dim, drop=drop)
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
        x = self.res_scale2(x.permute(0, 3, 1, 2)) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x).permute(0, 3, 1, 2))
                )
            )
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:8842-8848 ----
class C3k_CaFormerCGLU(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MetaFormerCGLUBlock(
                dim=c_, token_mixer=MF_Attention
            ) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:8850-8855 ----
class C3k2_CaFormerCGLU(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_CaFormerCGLU(self.c, self.c, n, shortcut, g) if c3k else MetaFormerCGLUBlock(
                dim=self.c, token_mixer=MF_Attention
            ) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/metaformer.py:174-200 ----
class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, dim, expansion_ratio=2,
        act1_layer=StarReLU, act2_layer=nn.Identity, 
        bias=False, kernel_size=7, padding=3,
        **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias) # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:8827-8833 ----
class C3k_ConvFormerCGLU(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MetaFormerCGLUBlock(
                dim=c_, token_mixer=SepConv
            ) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:8835-8840 ----
class C3k2_ConvFormerCGLU(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_ConvFormerCGLU(self.c, self.c, n, shortcut, g) if c3k else MetaFormerCGLUBlock(
                dim=self.c, token_mixer=SepConv
            ) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:11925-11948 ----
class ELGCA_CGLU(nn.Module):
    """
    dim: number of channels of input features
    """
    def __init__(self, dim, drop_path=0.1, mlp_ratio=4, heads=4):
        super().__init__()

        self.layer_norm1 = LayerNorm(dim, 'BiasFree')
        self.layer_norm2 = LayerNorm(dim, 'BiasFree')
        self.mlp = ConvolutionalGLU(dim)
        self.attn = ELGCA(dim, heads=heads)
        
    def forward(self, x):
        B, C, H, W = x.shape
        inp_copy = x
              
        x = self.layer_norm1(inp_copy)
        x = self.attn(x)
        out = x + inp_copy

        x = self.layer_norm2(out)
        x = self.mlp(x)
        out = out + x
        return out

# ---- 原样迁移自 nn/extra_modules/block.py:11950-11954 ----
class C3k_ELGCACGLU(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(ELGCA_CGLU(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:11956-11959 ----
class C3k2_ELGCACGLU(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_ELGCACGLU(self.c, self.c, 2, shortcut, g) if c3k else ELGCA_CGLU(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:8782-8788 ----
class C3k_IdentityFormerCGLU(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MetaFormerCGLUBlock(
                dim=c_, token_mixer=nn.Identity, norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False)
            ) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:8790-8795 ----
class C3k2_IdentityFormerCGLU(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_IdentityFormerCGLU(self.c, self.c, n, shortcut, g) if c3k else MetaFormerCGLUBlock(
                dim=self.c, token_mixer=nn.Identity, norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False)
            ) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:7789-7825 ----
class MutilScal(nn.Module):
    def __init__(self, dim=512, fc_ratio=4, dilation=[3, 5, 7], pool_ratio=16):
        super(MutilScal, self).__init__()
        self.conv0_1 = Conv(dim, dim//fc_ratio)
        self.conv0_2 = Conv(dim//fc_ratio, dim//fc_ratio, 3, d=dilation[-3], g=dim//fc_ratio)
        self.conv0_3 = Conv(dim//fc_ratio, dim, 1)

        self.conv1_2 = Conv(dim//fc_ratio, dim//fc_ratio, 3, d=dilation[-2], g=dim // fc_ratio)
        self.conv1_3 = Conv(dim//fc_ratio, dim, 1)

        self.conv2_2 = Conv(dim//fc_ratio, dim//fc_ratio, 3, d=dilation[-1], g=dim//fc_ratio)
        self.conv2_3 = Conv(dim//fc_ratio, dim, 1)

        self.conv3 = Conv(dim, dim, 1)

        self.Avg = nn.AdaptiveAvgPool2d(pool_ratio)

    def forward(self, x):
        u = x.clone()

        attn0_1 = self.conv0_1(x)
        attn0_2 = self.conv0_2(attn0_1)
        attn0_3 = self.conv0_3(attn0_2)

        attn1_2 = self.conv1_2(attn0_1)
        attn1_3 = self.conv1_3(attn1_2)

        attn2_2 = self.conv2_2(attn0_1)
        attn2_3 = self.conv2_3(attn2_2)

        attn = attn0_3 + attn1_3 + attn2_3
        attn = self.conv3(attn)
        attn = attn * u

        pool = self.Avg(attn)

        return pool

# ---- 原样迁移自 nn/extra_modules/block.py:7827-7871 ----
class Mutilscal_MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, atten_drop = 0., proj_drop = 0., dilation = [3, 5, 7], fc_ratio=4, pool_ratio=16):
        super(Mutilscal_MHSA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.MSC = MutilScal(dim=dim, fc_ratio=fc_ratio, dilation=dilation, pool_ratio=pool_ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim//fc_ratio, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(in_channels=dim//fc_ratio, out_channels=dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.kv = Conv(dim, 2 * dim, 1)

    def forward(self, x):
        u = x.clone()
        B, C, H, W = x.shape
        kv = self.MSC(x)
        kv = self.kv(kv)

        B1, C1, H1, W1 = kv.shape

        q = rearrange(x, 'b (h d) (hh) (ww) -> (b) h (hh ww) d', h=self.num_heads,
                      d=C // self.num_heads, hh=H, ww=W)
        k, v = rearrange(kv, 'b (kv h d) (hh) (ww) -> kv (b) h (hh ww) d', h=self.num_heads,
                         d=C // self.num_heads, hh=H1, ww=W1, kv=2)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.atten_drop(attn)
        attn = attn @ v

        attn = rearrange(attn, '(b) h (hh ww) d -> b (h d) (hh) (ww)', h=self.num_heads,
                         d=C // self.num_heads, hh=H, ww=W)
        c_attn = self.avgpool(x)
        c_attn = self.fc(c_attn)
        c_attn = c_attn * u
        return attn + c_attn

# ---- 原样迁移自 nn/extra_modules/block.py:7873-7889 ----
class MSMHSA_CGLU(nn.Module):
    def __init__(self,
                 inc,
                 drop_path=0.1,
                 ):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = ConvolutionalGLU(inc)
        self.msmhsa = nn.Sequential(
            Mutilscal_MHSA(inc),
            nn.BatchNorm2d(inc)
        )

    def forward(self, x):
        x = x + self.drop_path(self.msmhsa(x))
        x = x + self.drop_path(self.mlp(x))
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:7891-7895 ----
class C3k_MSMHSA_CGLU(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MSMHSA_CGLU(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:7898-7901 ----
class C3k2_MSMHSA_CGLU(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MSMHSA_CGLU(self.c, self.c, 2, shortcut, g) if c3k else MSMHSA_CGLU(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/metaformer.py:203-217 ----
class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer: https://arxiv.org/abs/2111.11418
    Modfiled for [B, H, W, C] input
    """
    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        y = x.permute(0, 3, 1, 2)
        y = self.pool(y)
        y = y.permute(0, 2, 3, 1)
        return y - x

# ---- 原样迁移自 nn/extra_modules/block.py:8812-8818 ----
class C3k_PoolingFormerCGLU(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MetaFormerCGLUBlock(
                dim=c_, token_mixer=Pooling, norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False)
            ) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:8820-8825 ----
class C3k2_PoolingFormerCGLU(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_PoolingFormerCGLU(self.c, self.c, n, shortcut, g) if c3k else MetaFormerCGLUBlock(
                dim=self.c, token_mixer=Pooling, norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False)
            ) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/metaformer.py:89-100 ----
class RandomMixing(nn.Module):
    def __init__(self, num_tokens=196, **kwargs):
        super().__init__()
        self.random_matrix = nn.parameter.Parameter(
            data=torch.softmax(torch.rand(num_tokens, num_tokens), dim=-1), 
            requires_grad=False)
    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H*W, C)
        x = torch.einsum('mn, bnc -> bmc', self.random_matrix, x)
        x = x.reshape(B, H, W, C)
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:8797-8803 ----
class C3k_RandomMixingCGLU(C3k):
    def __init__(self, c1, c2, n=1, num_tokens=196, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MetaFormerCGLUBlock(
                dim=c_, token_mixer=partial(RandomMixing, num_tokens=num_tokens), norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False)
            ) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:8805-8810 ----
class C3k2_RandomMixingCGLU(C3k2):
    def __init__(self, c1, c2, n=1, num_tokens=196, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_RandomMixingCGLU(self.c, self.c, n, num_tokens, shortcut, g) if c3k else MetaFormerCGLUBlock(
                dim=self.c, token_mixer=partial(RandomMixing, num_tokens=num_tokens), norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False)
            ) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:8390-8398 ----
class SHSABlock_CGLU(torch.nn.Module):
    def __init__(self, dim, qk_dim=16, pdim=32):
        super().__init__()
        self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups = dim, bn_weight_init = 0))
        self.mixer = Residual(SHSA(dim, qk_dim, pdim))
        self.ffn = ConvolutionalGLU(dim, int(dim * 2))
    
    def forward(self, x):
        return self.ffn(self.mixer(self.conv(x)))

# ---- 原样迁移自 nn/extra_modules/block.py:8410-8414 ----
class C3k_SHSA_CGLU(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(SHSABlock_CGLU(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:8416-8419 ----
class C3k2_SHSA_CGLU(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_SHSA_CGLU(self.c, self.c, 2, shortcut, g) if c3k else SHSABlock_CGLU(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:8436-8549 ----
class Modulator(nn.Module):
    def __init__(self, in_ch, out_ch, with_pos=True):
        super(Modulator, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.rate = [1, 6, 12, 18]
        self.with_pos = with_pos
        self.patch_size = 2
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.CA_fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // 16, in_ch, bias=False),
            nn.Sigmoid(),
        )

        # Pixel Attention
        self.PA_conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.PA_bn = nn.BatchNorm2d(in_ch)
        self.sigmoid = nn.Sigmoid()

        # Spatial Attention
        self.SA_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=rate, dilation=rate),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch)
            ) for rate in self.rate
        ])
        self.SA_out_conv = nn.Conv2d(len(self.rate) * out_ch, out_ch, 1)

        self.output_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self._init_weights()

        self.pj_conv = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=self.patch_size + 1,
                         stride=self.patch_size, padding=self.patch_size // 2)
        self.pos_conv = nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1, groups=self.out_ch, bias=True)
        self.layernorm = nn.LayerNorm(self.out_ch, eps=1e-6)

    def forward(self, x):
        res = x
        pa = self.PA(x)
        ca = self.CA(x)

        # Softmax(PA @ CA)
        pa_ca = torch.softmax(pa @ ca, dim=-1)

        # Spatial Attention
        sa = self.SA(x)

        # (Softmax(PA @ CA)) @ SA
        out = pa_ca @ sa
        out = self.norm(self.output_conv(out))
        out = out + self.bias
        synergistic_attn = out + res
        return synergistic_attn

    # def forward(self, x):
    #     pa_out = self.pa(x)
    #     ca_out = self.ca(x)
    #     sa_out = self.sa(x)
    #     # Concatenate along channel dimension
    #     combined_out = torch.cat([pa_out, ca_out, sa_out], dim=1)
    #
    #     return self.norm(self.output_conv(combined_out))



    def PE(self, x):
        proj = self.pj_conv(x)

        if self.with_pos:
            pos = proj * self.sigmoid(self.pos_conv(proj))

        pos = pos.flatten(2).transpose(1, 2)  # BCHW -> BNC
        embedded_pos = self.layernorm(pos)

        return embedded_pos

    def PA(self, x):
        attn = self.PA_conv(x)
        attn = self.PA_bn(attn)
        attn = self.sigmoid(attn)
        return x * attn

    def CA(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.CA_fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    def SA(self, x):
        sa_outs = [block(x) for block in self.SA_blocks]
        sa_out = torch.cat(sa_outs, dim=1)
        sa_out = self.SA_out_conv(sa_out)
        return sa_out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# ---- 原样迁移自 nn/extra_modules/block.py:8551-8571 ----
class SMA(nn.Module):
    def __init__(self, feature_size, num_heads, dropout):
        super(SMA, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_size, num_heads=num_heads, dropout=dropout)
        self.combined_modulator = Modulator(feature_size, feature_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, value, key, query):
        MSA = self.attention(query, key, value)[0]

        # 将输出转换为适合AttentionBlock的输入格式
        batch_size, seq_len, feature_size = MSA.shape
        MSA = MSA.permute(0, 2, 1).view(batch_size, feature_size, int(seq_len**0.5), int(seq_len**0.5))
        # 通过CombinedModulator进行multi-attn fusion
        synergistic_attn = self.combined_modulator.forward(MSA)


        # 将输出转换回 (batch_size, seq_len, feature_size) 格式
        x = synergistic_attn.view(batch_size, feature_size, -1).permute(0, 2, 1)

        return x

# ---- 原样迁移自 nn/extra_modules/block.py:8624-8642 ----
class SMAFormerBlock_CGLU(nn.Module):
    def __init__(self, ch_out, heads=8, dropout=0.1, forward_expansion=2):
        super(SMAFormerBlock_CGLU, self).__init__()
        self.norm1 = nn.LayerNorm(ch_out)
        # self.norm2 = nn.LayerNorm(ch_out)
        self.norm2 = LayerNorm2d(ch_out)
        self.synergistic_multi_attention = SMA(ch_out, heads, dropout)
        self.e_mlp = ConvolutionalGLU(ch_out, forward_expansion, drop=dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.flatten(2).permute(0, 2, 1)
        value, key, query, res = x, x, x, x
        attention = self.synergistic_multi_attention(query, key, value)
        query = self.dropout(self.norm1(attention + res))
        feed_forward = self.e_mlp(query.permute(0, 2, 1).reshape((b, c, h, w)))
        out = self.dropout(self.norm2(feed_forward))
        return out

# ---- 原样迁移自 nn/extra_modules/block.py:8655-8659 ----
class C3k_SMAFB_CGLU(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(SMAFormerBlock_CGLU(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:8661-8664 ----
class C3k2_SMAFB_CGLU(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_SMAFB_CGLU(self.c, self.c, 2, shortcut, g) if c3k else SMAFormerBlock_CGLU(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:12051-12068 ----
class StripCGLU(nn.Module):
    def __init__(self, dim, mlp_ratio=4., k1=1, k2=19, drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Strip_Attention(dim,k1,k2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = ConvolutionalGLU(dim)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:12070-12074 ----
class C3k_StripCGLU(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(StripCGLU(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:12076-12079 ----
class C3k2_StripCGLU(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_StripCGLU(self.c, self.c, 2, shortcut, g) if c3k else StripCGLU(self.c) for _ in range(n))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- StripCGLU ----
    try:
        module = StripCGLU(in_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'StripCGLU  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'StripCGLU  自测跳过: {e}' + RESET)
    # ---- C3k_StripCGLU ----
    try:
        module = C3k_StripCGLU(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k_StripCGLU  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k_StripCGLU  自测跳过: {e}' + RESET)
    # ---- C3k2_StripCGLU ----
    try:
        module = C3k2_StripCGLU(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_StripCGLU  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_StripCGLU  自测跳过: {e}' + RESET)

