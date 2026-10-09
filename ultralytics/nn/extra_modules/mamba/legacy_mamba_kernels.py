'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/backbone/MambaOut.py:207-240 , nn/extra_modules/GroupMamba/csms6s.py:111-129 , nn/extra_modules/GroupMamba/csms6s.py:133-148 , nn/extra_modules/GroupMamba/csms6s.py:150-168 , nn/extra_modules/GroupMamba/csms6s.py:171-186 , nn/extra_modules/GroupMamba/csms6s.py:188-206 , nn/extra_modules/GroupMamba/csms6s.py:56-71 , nn/extra_modules/GroupMamba/csms6s.py:73-91 , nn/extra_modules/GroupMamba/csms6s.py:94-109 , nn/extra_modules/GroupMamba/groupmamba.py:203-236 , nn/extra_modules/GroupMamba/groupmamba.py:49-78 , nn/extra_modules/GroupMamba/groupmamba.py:80-157 , nn/extra_modules/MambaVision.py:223-236 , nn/extra_modules/block.py:12717-12721 , nn/extra_modules/block.py:12723-12726 , nn/extra_modules/block.py:12732-12736 , nn/extra_modules/block.py:12738-12741 , nn/extra_modules/block.py:12747-12751 , nn/extra_modules/block.py:12753-12756 , nn/extra_modules/block.py:13467-13471 , nn/extra_modules/block.py:13473-13476 , nn/extra_modules/block.py:13529-13533 , nn/extra_modules/block.py:13535-13538 , nn/extra_modules/block.py:14302-14306 , nn/extra_modules/block.py:14308-14311 , nn/extra_modules/block.py:14313-14317 , nn/extra_modules/block.py:14319-14322 , nn/extra_modules/block.py:14328-14332 , nn/extra_modules/block.py:14334-14337 , nn/extra_modules/block.py:4627-4631 , nn/extra_modules/block.py:4633-4637 , nn/extra_modules/block.py:4639-4642 , nn/extra_modules/block.py:4644-4648 , nn/extra_modules/block.py:4650-4653 , nn/extra_modules/mamba/mamba_ssm/ops/triton/layernorm_gated.py:415-437 , nn/extra_modules/mamba_vss.py:204-222 , nn/extra_modules/mamba_yolo.py:418-436 , nn/extra_modules/mobileMamba/mobilemamba.py:334-344 , nn/extra_modules/mobileMamba/mobilemamba.py:347-367 , nn/extra_modules/savss.py:118-353 , nn/extra_modules/savss.py:355-416 , nn/extra_modules/savss.py:38-75 , nn/extra_modules/savss.py:77-115 , nn/extra_modules/transMamba.py:528-541
A2 组：0526 版 CSP 包装壳 + 其内核（内核在 2107 版中不存在，一并迁入）。
壳签名 (c1, c2, n=1, c3k=False, ...) 与 base_modules + repeat_modules 契约一致，旧 yaml 可一字不改。
本组包含：C3k2_GroupMamba, C3k2_GroupMambaBlock, C3k2_LSBlock, C3k2_LVMB, C3k2_MambaOut, C3k2_MambaVision, C3k2_MobileMamba, C3k2_SAVSS, C3k2_TransMamba, C3k2_VSS
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

from typing import Callable
from timm.layers import DropPath
import torch.nn.functional as F
# 原样迁移自 nn/extra_modules/savss.py:31-35
def get_norm_layer(norm_type, channels, num_groups):
    if norm_type == 'GN':
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        return nn.InstanceNorm3d(channels)
import math
import torch.nn as nn
from functools import partial
from einops import rearrange
from einops import repeat
from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
import torch
from timm.layers import trunc_normal_
from ultralytics.nn.extra_modules.mamba.GLVSS import Block, MambaBlock
from ultralytics.nn.extra_modules.mamba.MobileMamba.mobilemamba import MobileMambaModule
from ultralytics.nn.extra_modules.mamba.SAVSS import BottConv
from ultralytics.nn.extra_modules.mamba.TinyViM import Conv2d_BN, FFN, SS2D
from ultralytics.nn.extra_modules.mamba.TransMixer import selective_scan_fn
from ultralytics.nn.extra_modules.module.mambaout import LayerNormGeneral
from ultralytics.nn.modules.block import Bottleneck, C3k, C3k2

class Residual(nn.Module):
    """通用残差包装（迁移自旧版 nn/extra_modules/block.py，勿与 ultralytics.nn.modules.block.Residual 混淆）。"""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

from ultralytics.nn.modules.transformer import LayerNorm2d, TransformerBlock


# ---- 原样迁移自 nn/extra_modules/GroupMamba/csms6s.py:73-91 ----
class CrossMerge_1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        y = ys[:, 0]
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 1, C, L))
        xs[:, 0] = x
        xs = xs.view(B, 1, C, H, W)
        return xs

# ---- 原样迁移自 nn/extra_modules/GroupMamba/csms6s.py:111-129 ----
class CrossMerge_2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        y = ys[:, 0].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 1, C, L))
        xs[:, 0] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs = xs.view(B, 1, C, H, W)
        return xs

# ---- 原样迁移自 nn/extra_modules/GroupMamba/csms6s.py:150-168 ----
class CrossMerge_3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        y = ys[:, 0].flip(dims=[-1]).view(B, D, -1)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 1, C, L))
        xs[:, 0] = torch.flip(x, dims=[-1])
        xs = xs.view(B, 1, C, H, W)
        return xs

# ---- 原样迁移自 nn/extra_modules/GroupMamba/csms6s.py:188-206 ----
class CrossMerge_4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        y = ys[:, 0].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1).flip(dims=[-1])
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 1, C, L))
        xs[:, 0] = torch.flip(x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1])
        xs = xs.view(B, 1, C, H, W)
        return xs

# ---- 原样迁移自 nn/extra_modules/GroupMamba/csms6s.py:56-71 ----
class CrossScan_1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 1, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        y = ys[:, 0]
        return y.view(B, -1, H, W)

# ---- 原样迁移自 nn/extra_modules/GroupMamba/csms6s.py:94-109 ----
class CrossScan_2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 1, C, H * W))
        xs[:, 0] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        y = ys[:, 0].view(B, -1, H, W).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)

# ---- 原样迁移自 nn/extra_modules/GroupMamba/csms6s.py:133-148 ----
class CrossScan_3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 1, C, H * W))
        xs[:, 0] = torch.flip(x.flatten(2, 3), dims=[-1])
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        y = ys[:, 0].flip(dims=[-1]).view(B, 1, -1, L)
        return y.view(B, -1, H, W)

# ---- 原样迁移自 nn/extra_modules/GroupMamba/csms6s.py:171-186 ----
class CrossScan_4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 1, C, H * W))
        xs[:, 0] = torch.flip(x.transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1])
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        y = ys[:, 0].view(B, -1, H, W).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L).flip(dims=[-1]).view(B, 1, -1, L)
        return y.view(B, -1, H, W)

# ---- 原样迁移自 nn/extra_modules/GroupMamba/groupmamba.py:80-157 ----
class GroupMambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=1, d_conv=3, expand=1, reduction=16):
        super().__init__()

        num_channels_reduced = input_dim // reduction
        self.fc1 = nn.Linear(input_dim, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, output_dim, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)

        self.mamba_g1 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )
        self.mamba_g2 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )
        self.mamba_g3 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )
        self.mamba_g4 = SS2D(
            d_model=input_dim // 4,
            d_state=d_state,
            ssm_ratio=expand,
            d_conv=d_conv
        )

        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x_dtype = x.dtype
        B, C, H, W = x.shape
        N = H * W
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm(x)

        # Channel Affinity
        z = x.permute(0, 2, 1).mean(dim=2)

        fc_out_1 = self.relu(self.fc1(z))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        x = rearrange(x, 'b (h w) c -> b h w c', b=B, h=H, w=W, c=C)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=-1)

        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        # Four scans applied to 4 different directions, each is applied for N/4 channels
        x_mamba1 = self.mamba_g1(x1, CrossScan=CrossScan_1, CrossMerge=CrossMerge_1).to(x_dtype)
        x_mamba2 = self.mamba_g2(x2, CrossScan=CrossScan_2, CrossMerge=CrossMerge_2).to(x_dtype)
        x_mamba3 = self.mamba_g3(x3, CrossScan=CrossScan_3, CrossMerge=CrossMerge_3).to(x_dtype)
        x_mamba4 = self.mamba_g4(x4, CrossScan=CrossScan_4, CrossMerge=CrossMerge_4).to(x_dtype)

        # Combine all feature maps
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=-1) * self.skip_scale * x.to(x_dtype)

        x_mamba = rearrange(x_mamba, 'b h w c -> b (h w) c', b=B, h=H, w=W, c=C)

        # Channel Modulation
        x_mamba = x_mamba * fc_out_2.unsqueeze(1)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)

        return x_mamba.permute(0, 2, 1).view([B, C, H, W])

# ---- 原样迁移自 nn/extra_modules/GroupMamba/groupmamba.py:49-78 ----
class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, in_features, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

# ---- 原样迁移自 nn/extra_modules/GroupMamba/groupmamba.py:203-236 ----
class Block_mamba(nn.Module):
    def __init__(self, 
        dim, 
        mlp_ratio=2,
        drop_path=0., 
        norm_layer=LayerNorm2d
    ):
        super().__init__()
        self.norm2 = norm_layer(dim)

        self.attn = GroupMambaLayer(dim, dim)
        self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# ---- 原样迁移自 nn/extra_modules/mamba_vss.py:204-222 ----
class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0.2,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        input = input.permute((0, 2, 3, 1))
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x.permute((0, 3, 1, 2))

# ---- 原样迁移自 nn/extra_modules/block.py:4627-4631 ----
class Bottleneck_VSS(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = VSSBlock(c2)

# ---- 原样迁移自 nn/extra_modules/block.py:14302-14306 ----
class C3k_GroupMamba(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GroupMambaLayer(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:14308-14311 ----
class C3k2_GroupMamba(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_GroupMamba(self.c, self.c, 2, shortcut, g) if c3k else GroupMambaLayer(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:14313-14317 ----
class C3k_GroupMambaBlock(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Block_mamba(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:14319-14322 ----
class C3k2_GroupMambaBlock(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_GroupMambaBlock(self.c, self.c, 2, shortcut, g) if c3k else Block_mamba(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/mamba_yolo.py:418-436 ----
class LSBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=3, padding=3 // 2, groups=hidden_features)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, padding=0)
        self.act = act_layer()
        self.fc3 = nn.Conv2d(hidden_features, in_features, kernel_size=1, padding=0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        input = x
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = input + self.drop(x)
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:13467-13471 ----
class C3k_LSBlock(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(LSBlock(c_, depth=1) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:13473-13476 ----
class C3k2_LSBlock(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_LSBlock(self.c, self.c, n, shortcut, g) if c3k else LSBlock(self.c, depth=1) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:4644-4648 ----
class C3k_LVMB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(VSSBlock(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:4650-4653 ----
class C3k2_LVMB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_LVMB(self.c, self.c, 2, shortcut, g) if c3k else VSSBlock(self.c) for _ in range(n))

# ---- 原样迁移自 nn/backbone/MambaOut.py:207-240 ----
class GatedCNNBlock_BCHW(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args: 
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve practical efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and 
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(LayerNormGeneral,eps=1e-6,normalized_dim=(1, 2, 3)), 
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        self.norm = norm_layer((dim, 1, 1))
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 1)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x # [B, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        # c = c.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        # c = c.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=1))
        x = self.drop_path(x)
        return x + shortcut

# ---- 原样迁移自 nn/extra_modules/block.py:12747-12751 ----
class C3k_MambaOut(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GatedCNNBlock_BCHW(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:12753-12756 ----
class C3k2_MambaOut(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MambaOut(self.c, self.c, n, shortcut, g) if c3k else GatedCNNBlock_BCHW(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/MambaVision.py:223-236 ----
class MambaVisionBlock(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.attention_block = Block(dim, is_attention=True)
        self.mamba_block = Block(dim)
    
    def forward(self, x):
        N, C, H, W = x.size()
        x = x.flatten(2).permute(0, 2, 1)
        x = self.mamba_block(x)
        x = self.attention_block(x)
        x = x.permute(0, 2, 1).view([N, C, H, W])
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:14328-14332 ----
class C3k_MambaVision(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MambaVisionBlock(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:14334-14337 ----
class C3k2_MambaVision(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MambaVision(self.c, self.c, 2, shortcut, g) if c3k else MambaVisionBlock(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/mobileMamba/mobilemamba.py:334-344 ----
class MobileMambaBlockWindow(torch.nn.Module):
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.25,
                 kernels=5, ssm_ratio=1, forward_type="v052d",):
        super().__init__()
        self.dim = dim
        self.attn = MobileMambaModule(dim, global_ratio=global_ratio, local_ratio=local_ratio,
                                           kernels=kernels, ssm_ratio=ssm_ratio, forward_type=forward_type,)

    def forward(self, x):
        x = self.attn(x)
        return x

# ---- 原样迁移自 nn/extra_modules/mobileMamba/mobilemamba.py:347-367 ----
class MobileMambaBlock(torch.nn.Module):
    def __init__(self, ed, global_ratio=0.25, local_ratio=0.25,
                 kernels=5,  drop_path=0., has_skip=True, ssm_ratio=1, forward_type="v052d"):
        super().__init__()

        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.))
        self.ffn0 = Residual(FFN(ed, int(ed * 2)))

        self.mixer = Residual(MobileMambaBlockWindow(ed, global_ratio=global_ratio, local_ratio=local_ratio, kernels=kernels, ssm_ratio=ssm_ratio,forward_type=forward_type))

        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.,))
        self.ffn1 = Residual(FFN(ed, int(ed * 2)))

        self.has_skip = has_skip
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))
        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:12732-12736 ----
class C3k_MobileMamba(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MobileMambaBlock(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:12738-12741 ----
class C3k2_MobileMamba(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MobileMamba(self.c, self.c, n, shortcut, g) if c3k else MobileMambaBlock(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/savss.py:38-75 ----
class GBC(nn.Module):
    def __init__(self, in_channels, norm_type='GN'):
        super(GBC, self).__init__()

        self.block1 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, 16),
            nn.ReLU()
        )

    def forward(self, x):
        residual = x

        x1 = self.block1(x)
        x1 = self.block2(x1)
        x2 = self.block3(x)
        x = x1 * x2
        x = self.block4(x)

        return x + residual

# ---- 原样迁移自 nn/extra_modules/savss.py:77-115 ----
class PAF(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 after_relu: bool = False,
                 mid_norm: nn.Module = nn.BatchNorm2d,
                 in_norm: nn.Module = nn.BatchNorm2d):
        super().__init__()
        self.after_relu = after_relu

        self.feature_transform = nn.Sequential(
            BottConv(in_channels, mid_channels, mid_channels=16, kernel_size=1),
            mid_norm(mid_channels)
        )

        self.channel_adapter = nn.Sequential(
            BottConv(mid_channels, in_channels, mid_channels=16, kernel_size=1),
            in_norm(in_channels)
        )

        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, base_feat: torch.Tensor, guidance_feat: torch.Tensor) -> torch.Tensor:
        base_shape = base_feat.size()

        if self.after_relu:
            base_feat = self.relu(base_feat)
            guidance_feat = self.relu(guidance_feat)

        guidance_query = self.feature_transform(guidance_feat)
        base_key = self.feature_transform(base_feat)
        guidance_query = F.interpolate(guidance_query, size=[base_shape[2], base_shape[3]], mode='bilinear', align_corners=False)
        similarity_map = torch.sigmoid(self.channel_adapter(base_key * guidance_query))
        resized_guidance = F.interpolate(guidance_feat, size=[base_shape[2], base_shape[3]], mode='bilinear', align_corners=False)

        fused_feature = (1 - similarity_map) * base_feat + similarity_map * resized_guidance

        return fused_feature

# ---- 原样迁移自 nn/extra_modules/mamba/mamba_ssm/ops/triton/layernorm_gated.py:415-437 ----
class RMSNorm(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-5, group_size=None, norm_before_gate=True, device=None, dtype=None):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))
        """
        return rmsnorm_fn(x, self.weight, self.bias, z=z, eps=self.eps, group_size=self.group_size,
                          norm_before_gate=self.norm_before_gate)

# ---- 原样迁移自 nn/extra_modules/savss.py:118-353 ----
class SAVSS_2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_size=7,
            bias=False,
            conv_bias=False,
            init_layer_scale=None,
            default_hw_shape=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.default_hw_shape = default_hw_shape
        self.default_permute_order = None
        self.default_permute_order_inverse = None
        self.n_directions = 4

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        assert conv_size % 2 == 1
        self.conv2d = BottConv(in_channels=self.d_inner, out_channels=self.d_inner, mid_channels=self.d_inner // 16, kernel_size=3, padding=1, stride=1)
        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True
        )

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.direction_Bs = nn.Parameter(torch.zeros(self.n_directions + 1, self.d_state))
        trunc_normal_(self.direction_Bs, std=0.02)

    def sass(self, hw_shape):
        H, W = hw_shape
        L = H * W
        o1, o2, o3, o4 = [], [], [], []
        d1, d2, d3, d4 = [], [], [], []
        o1_inverse = [-1 for _ in range(L)]
        o2_inverse = [-1 for _ in range(L)]
        o3_inverse = [-1 for _ in range(L)]
        o4_inverse = [-1 for _ in range(L)]

        if H % 2 == 1:
            i, j = H - 1, W - 1
            j_d = "left"
        else:
            i, j = H - 1, 0
            j_d = "right"

        while i > -1:
            assert j_d in ["right", "left"]
            idx = i * W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if j_d == "right":
                if j < W - 1:
                    j = j + 1
                    d1.append(1)
                else:
                    i = i - 1
                    d1.append(3)
                    j_d = "left"
            else:
                if j > 0:
                    j = j - 1
                    d1.append(2)
                else:
                    i = i - 1
                    d1.append(3)
                    j_d = "right"
        d1 = [0] + d1[:-1]

        i, j = 0, 0
        i_d = "down"
        while j < W:
            assert i_d in ["down", "up"]
            idx = i * W + j
            o2_inverse[idx] = len(o2)
            o2.append(idx)
            if i_d == "down":
                if i < H - 1:
                    i = i + 1
                    d2.append(4)
                else:
                    j = j + 1
                    d2.append(1)
                    i_d = "up"
            else:
                if i > 0:
                    i = i - 1
                    d2.append(3)
                else:
                    j = j + 1
                    d2.append(1)
                    i_d = "down"
        d2 = [0] + d2[:-1]

        for diag in range(H + W - 1):
            if diag % 2 == 0:
                for i in range(min(diag + 1, H)):
                    j = diag - i
                    if j < W:
                        idx = i * W + j
                        o3.append(idx)
                        o3_inverse[idx] = len(o1) - 1
                        d3.append(1 if j == diag else 4)
            else:
                for j in range(min(diag + 1, W)):
                    i = diag - j
                    if i < H:
                        idx = i * W + j
                        o3.append(idx)
                        o3_inverse[idx] = len(o1) - 1
                        d3.append(4 if i == diag else 1)
        d3 = [0] + d3[:-1]

        for diag in range(H + W - 1):
            if diag % 2 == 0:
                for i in range(min(diag + 1, H)):
                    j = diag - i
                    if j < W:
                        idx = i * W + (W - j - 1)
                        o4.append(idx)
                        o4_inverse[idx] = len(o4) - 1
                        d4.append(1 if j == diag else 4)
            else:
                for j in range(min(diag + 1, W)):
                    i = diag - j
                    if i < H:
                        idx = i * W + (W - j - 1)
                        o4.append(idx)
                        o4_inverse[idx] = len(o4) - 1
                        d4.append(4 if i == diag else 1)
        d4 = [0] + d4[:-1]

        return (tuple(o1), tuple(o2), tuple(o3), tuple(o4)), \
            (tuple(o1_inverse), tuple(o2_inverse), tuple(o3_inverse), tuple(o4_inverse)), \
            (tuple(d1), tuple(d2), tuple(d3), tuple(d4))

    def forward(self, x, hw_shape):
        batch_size, L, _ = x.shape
        H, W = hw_shape
        E = self.d_inner

        conv_state, ssm_state = None, None
        xz = self.in_proj(x)
        A = -torch.exp(self.A_log.float())

        x, z = xz.chunk(2, dim=-1)
        x_2d = x.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
        x_2d = self.act(self.conv2d(x_2d))
        x_conv = x_2d.permute(0, 2, 3, 1).reshape(batch_size, L, E)

        x_dbl = self.x_proj(x_conv)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)
        dt = dt.permute(0, 2, 1).contiguous()
        B = B.permute(0, 2, 1).contiguous()
        C = C.permute(0, 2, 1).contiguous()

        assert self.activation in ["silu", "swish"]

        orders, inverse_orders, directions = self.sass(hw_shape)
        direction_Bs = [self.direction_Bs[d, :] for d in directions]
        direction_Bs = [dB[None, :, :].expand(batch_size, -1, -1).permute(0, 2, 1).to(dtype=B.dtype) for dB in
                        direction_Bs]

        y_scan = [
            selective_scan_fn(
                x_conv[:, o, :].permute(0, 2, 1).contiguous(),
                dt,
                A,
                (B + dB).contiguous(),
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            ).permute(0, 2, 1)[:, inv_order, :]
            for o, inv_order, dB in zip(orders, inverse_orders, direction_Bs)
        ]

        y = sum(y_scan) * self.act(z.contiguous())
        out = self.out_proj(y)
        if self.init_layer_scale is not None:
            out = out * self.gamma

        return out

# ---- 原样迁移自 nn/extra_modules/savss.py:355-416 ----
class SAVSS_Layer(nn.Module):
    def __init__(
            self,
            embed_dims,
            use_rms_norm=False,
            with_dwconv=False,
            drop_path_rate=0.0,
    ):

        super(SAVSS_Layer, self).__init__()
        if use_rms_norm:
            self.norm = RMSNorm(embed_dims)
        else:
            self.norm = nn.LayerNorm(embed_dims)

        self.with_dwconv = with_dwconv
        if self.with_dwconv:
            self.dw = nn.Sequential(
                nn.Conv2d(
                    embed_dims,
                    embed_dims,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    bias=False,
                    groups=embed_dims
                ),
                nn.BatchNorm2d(embed_dims),
                nn.GELU(),
            )

        self.SAVSS_2D = SAVSS_2D(d_model=embed_dims)
        # self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))
        self.drop_path = DropPath(drop_prob=drop_path_rate)
        self.linear_256 = nn.Linear(in_features=embed_dims, out_features=embed_dims, bias=True)
        self.GN_256 = nn.GroupNorm(num_channels=embed_dims, num_groups=16)
        self.GBC_C = GBC(embed_dims)
        self.PAF_256 = PAF(embed_dims, embed_dims // 2)

    def forward(self, x):
        # B, L, C = x.shape
        # H = W = int(math.sqrt(L))
        B, C, H, W = x.size()
        hw_shape = (H, W)
        # x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        for i in range(2):
            x = self.GBC_C(x)

        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        mixed_x = self.drop_path(self.SAVSS_2D(self.norm(x), hw_shape))
        mixed_x = self.PAF_256(x.permute(0, 2, 1).reshape(B, C, H, W),
                               mixed_x.permute(0, 2, 1).reshape(B, C, H, W))
        mixed_x = self.GN_256(mixed_x).reshape(B, C, H * W).permute(0, 2, 1)

        if self.with_dwconv:
            mixed_x = mixed_x.reshape(B, H, W, C).permute(0, 3, 1, 2)
            mixed_x = self.GBC_C(mixed_x)
            mixed_x = mixed_x.reshape(B, C, H * W).permute(0, 2, 1)

        mixed_x_res = self.linear_256(self.GN_256(mixed_x.permute(0, 2, 1)).permute(0, 2, 1))
        output = mixed_x + mixed_x_res
        return output.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

# ---- 原样迁移自 nn/extra_modules/block.py:12717-12721 ----
class C3k_SAVSS(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(SAVSS_Layer(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:12723-12726 ----
class C3k2_SAVSS(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_SAVSS(self.c, self.c, n, shortcut, g) if c3k else SAVSS_Layer(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/transMamba.py:528-541 ----
class TransMambaBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=1.5, bias=False, LayerNorm_type='BiasFree'):
        super(TransMambaBlock, self).__init__()

        self.trans_block = TransformerBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.mamba_block = MambaBlock(dim, LayerNorm_type)
        self.conv = nn.Conv2d(int(dim*2), dim, kernel_size=1, bias=bias) 

    def forward(self, x):
        x1 = self.trans_block(x)
        x2 = self.mamba_block(x)
        out = torch.cat((x1, x2), 1)
        out = self.conv(out)
        return out

# ---- 原样迁移自 nn/extra_modules/block.py:13529-13533 ----
class C3k_TransMamba(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(TransMambaBlock(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:13535-13538 ----
class C3k2_TransMamba(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_TransMamba(self.c, self.c, n, shortcut, g) if c3k else TransMambaBlock(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:4633-4637 ----
class C3k_VSS(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_VSS(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:4639-4642 ----
class C3k2_VSS(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_VSS(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_VSS(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- C3k2_TransMamba ----
    try:
        module = C3k2_TransMamba(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_TransMamba  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_TransMamba  自测跳过: {e}' + RESET)
    # ---- C3k_VSS ----
    try:
        module = C3k_VSS(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k_VSS  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k_VSS  自测跳过: {e}' + RESET)
    # ---- C3k2_VSS ----
    try:
        module = C3k2_VSS(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_VSS  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_VSS  自测跳过: {e}' + RESET)

