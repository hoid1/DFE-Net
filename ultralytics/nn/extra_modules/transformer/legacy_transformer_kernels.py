'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/backbone/TransNeXt/TransNext_native.py:46-71 , nn/backbone/TransNeXt/TransNext_native.py:72-78 , nn/backbone/TransNeXt/TransNext_native.py:80-185 , nn/extra_modules/PlainUSR.py:122-130 , nn/extra_modules/PlainUSR.py:132-154 , nn/extra_modules/attention.py:1038-1050 , nn/extra_modules/attention.py:1385-1482 , nn/extra_modules/attention.py:1530-1553 , nn/extra_modules/attention.py:762-771 , nn/extra_modules/attention.py:773-850 , nn/extra_modules/block.py:1130-1140 , nn/extra_modules/block.py:1142-1146 , nn/extra_modules/block.py:1148-1151 , nn/extra_modules/block.py:15476-15489 , nn/extra_modules/block.py:15504-15509 , nn/extra_modules/block.py:15511-15515 , nn/extra_modules/block.py:2504-2509 , nn/extra_modules/block.py:2511-2515 , nn/extra_modules/block.py:2517-2520 , nn/extra_modules/block.py:3096-3105 , nn/extra_modules/block.py:3113-3116 , nn/extra_modules/block.py:3783-3792 , nn/extra_modules/block.py:3794-3798 , nn/extra_modules/block.py:3800-3803
A2 组：0526 版 CSP 包装壳 + 其内核（内核在 2107 版中不存在，一并迁入）。
壳签名 (c1, c2, n=1, c3k=False, ...) 与 base_modules + repeat_modules 契约一致，旧 yaml 可一字不改。
本组包含：C3k2_AggregatedAtt, C3k2_CloAtt, C3k2_DLKA, C3k2_FocusedLinearAttention, C3k2_RMBC_LA
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn.functional as F
from efficientnet_pytorch.model import MemoryEfficientSwish
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch
from ultralytics.nn.extra_modules.attention.DeformableLKA import DeformConv
from ultralytics.nn.extra_modules.conv_module.RMBC import RepMBConv
from ultralytics.nn.extra_modules.module.FATBlock import img2windows, windows2img
from ultralytics.nn.modules.block import Bottleneck, C3k, C3k2
from ultralytics.nn.modules.conv import Conv


# ---- 原样迁移自 nn/backbone/TransNeXt/TransNext_native.py:72-78 ----
@torch.no_grad()
def get_seqlen_and_mask(input_resolution, window_size):
    attn_map = F.unfold(torch.ones([1, 1, input_resolution[0], input_resolution[1]]), window_size,
                        dilation=1, padding=(window_size // 2, window_size // 2), stride=1)
    attn_local_length = attn_map.sum(-2).squeeze().unsqueeze(-1)
    attn_mask = (attn_map.squeeze(0).permute(1, 0)) == 0
    return attn_local_length, attn_mask

# ---- 原样迁移自 nn/backbone/TransNeXt/TransNext_native.py:80-185 ----
class AggregatedAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, window_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.sr_ratio = sr_ratio

        assert window_size % 2 == 1, "window size must be odd"
        self.window_size = window_size
        self.local_len = window_size ** 2

        self.pool_H, self.pool_W = input_resolution[0] // self.sr_ratio, input_resolution[1] // self.sr_ratio
        self.pool_len = self.pool_H * self.pool_W

        self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2, stride=1)
        self.temperature = nn.Parameter(torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1)) #Initialize softplus(temperature) to 1/0.24.

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        #Components to generate pooled features.
        self.pool = nn.AdaptiveAvgPool2d((self.pool_H, self.pool_W))
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

        # relative bias for local features
        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len), mean=0,
                                  std=0.0004))

        # Generate padding_mask && sequnce length scale
        local_seq_length, padding_mask = get_seqlen_and_mask(input_resolution, window_size)
        self.register_buffer("seq_length_scale", torch.as_tensor(np.log(local_seq_length.numpy() + self.pool_len)),
                             persistent=False)
        self.register_buffer("padding_mask", padding_mask, persistent=False)

        # dynamic_local_bias:
        self.learnable_tokens = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.head_dim, self.local_len), mean=0, std=0.02))
        self.learnable_bias = nn.Parameter(torch.zeros(num_heads, 1, self.local_len))

    def forward(self, x, H, W, relative_pos_index, relative_coords_table):
        B, N, C = x.shape

        #Generate queries, normalize them with L2, add query embedding, and then magnify with sequence length scale and temperature.
        #Use softplus function ensuring that the temperature is not lower than 0.
        q_norm=F.normalize(self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3),dim=-1)
        q_norm_scaled = (q_norm + self.query_embedding) * F.softplus(self.temperature) * self.seq_length_scale

        # Generate unfolded keys and values and l2-normalize them
        k_local, v_local = self.kv(x).chunk(2, dim=-1)
        k_local = F.normalize(k_local.reshape(B, N, self.num_heads, self.head_dim), dim=-1).reshape(B, N, -1)
        kv_local = torch.cat([k_local, v_local], dim=-1).permute(0, 2, 1).reshape(B, -1, H, W)
        k_local, v_local = self.unfold(kv_local).reshape(
            B, 2 * self.num_heads, self.head_dim, self.local_len, N).permute(0, 1, 4, 2, 3).chunk(2, dim=1)

        # Compute local similarity
        attn_local = ((q_norm_scaled.unsqueeze(-2) @ k_local).squeeze(-2) \
                      + self.relative_pos_bias_local.unsqueeze(1)).masked_fill(self.padding_mask, float('-inf'))

        # Generate pooled features
        x_ = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        x_ = self.pool(self.act(self.sr(x_))).reshape(B, -1, self.pool_len).permute(0, 2, 1)
        x_ = self.norm(x_)

        # Generate pooled keys and values
        kv_pool = self.kv(x_).reshape(B, self.pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_pool, v_pool = kv_pool.chunk(2, dim=1)

        #Use MLP to generate continuous relative positional bias for pooled features.
        pool_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                    relative_pos_index.view(-1)].view(-1, N, self.pool_len)
        # Compute pooled similarity
        attn_pool = q_norm_scaled @ F.normalize(k_pool, dim=-1).transpose(-2, -1) + pool_bias

        # Concatenate local & pooled similarity matrices and calculate attention weights through the same Softmax
        attn = torch.cat([attn_local, attn_pool], dim=-1).softmax(dim=-1)
        attn = self.attn_drop(attn)

        #Split the attention weights and separately aggregate the values of local & pooled features
        attn_local, attn_pool = torch.split(attn, [self.local_len, self.pool_len], dim=-1)
        x_local = (((q_norm @ self.learnable_tokens) + self.learnable_bias + attn_local).unsqueeze(-2) @ v_local.transpose(-2, -1)).squeeze(-2)
        x_pool = attn_pool @ v_pool
        x = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)

        #Linear projection and output
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# ---- 原样迁移自 nn/extra_modules/attention.py:762-771 ----
class AttnMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
                            nn.Conv2d(dim, dim, 1, 1, 0),
                            MemoryEfficientSwish(),
                            nn.Conv2d(dim, dim, 1, 1, 0)
                         )
    def forward(self, x):
        return self.act_block(x)

# ---- 原样迁移自 nn/backbone/TransNeXt/TransNext_native.py:46-71 ----
@torch.no_grad()
def get_relative_position_cpb(query_size, key_size, pretrain_size=None):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrain_size = pretrain_size or query_size
    axis_qh = torch.arange(query_size[0], dtype=torch.float32)
    axis_kh = F.adaptive_avg_pool1d(axis_qh.unsqueeze(0), key_size[0]).squeeze(0)
    axis_qw = torch.arange(query_size[1], dtype=torch.float32)
    axis_kw = F.adaptive_avg_pool1d(axis_qw.unsqueeze(0), key_size[1]).squeeze(0)
    axis_kh, axis_kw = torch.meshgrid(axis_kh, axis_kw)
    axis_qh, axis_qw = torch.meshgrid(axis_qh, axis_qw)

    axis_kh = torch.reshape(axis_kh, [-1])
    axis_kw = torch.reshape(axis_kw, [-1])
    axis_qh = torch.reshape(axis_qh, [-1])
    axis_qw = torch.reshape(axis_qw, [-1])

    relative_h = (axis_qh[:, None] - axis_kh[None, :]) / (pretrain_size[0] - 1) * 8
    relative_w = (axis_qw[:, None] - axis_kw[None, :]) / (pretrain_size[1] - 1) * 8
    relative_hw = torch.stack([relative_h, relative_w], dim=-1).view(-1, 2)

    relative_coords_table, idx_map = torch.unique(relative_hw, return_inverse=True, dim=0)

    relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
        torch.abs(relative_coords_table) + 1.0) / torch.log2(torch.tensor(8, dtype=torch.float32))

    return idx_map, relative_coords_table

# ---- 原样迁移自 nn/extra_modules/attention.py:1530-1553 ----
class TransNeXt_AggregatedAttention(nn.Module):
    def __init__(self, dim, input_resolution, sr_ratio=8, num_heads=8, window_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0.) -> None:
        super().__init__()
        
        if type(input_resolution) == int:
            input_resolution = (input_resolution, input_resolution)
        relative_pos_index, relative_coords_table = get_relative_position_cpb(
                query_size=input_resolution,
                key_size=(20, 20),
                pretrain_size=input_resolution)
        
        self.register_buffer(f"relative_pos_index", relative_pos_index, persistent=False)
        self.register_buffer(f"relative_coords_table", relative_coords_table, persistent=False)
        self.attention = AggregatedAttention(dim, input_resolution, num_heads, window_size, qkv_bias, attn_drop, proj_drop, sr_ratio)
    
    def forward(self, x):
        B, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        relative_pos_index = getattr(self, f"relative_pos_index")
        relative_coords_table = getattr(self, f"relative_coords_table")
        x = self.attention(x, H, W, relative_pos_index.to(x.device), relative_coords_table.to(x.device))
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:3783-3792 ----
class Bottleneck_AggregatedAttention(Bottleneck):
    """Standard bottleneck With CloAttention."""

    def __init__(self, c1, c2, input_resolution, sr_ratio, shortcut=True, g=1, k=..., e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        self.attention = TransNeXt_AggregatedAttention(c2, input_resolution, sr_ratio)
    
    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))

# ---- 原样迁移自 nn/extra_modules/attention.py:773-850 ----
class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, group_split=[4, 4], kernel_sizes=[5], window_size=4, 
                 attn_drop=0., proj_drop=0., qkv_bias=True):
        super().__init__()
        assert sum(group_split) == num_heads
        assert len(kernel_sizes) + 1 == len(group_split)
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scalor = self.dim_head ** -0.5
        self.kernel_sizes = kernel_sizes
        self.window_size = window_size
        self.group_split = group_split
        convs = []
        act_blocks = []
        qkvs = []
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            group_head = group_split[i]
            if group_head == 0:
                continue
            convs.append(nn.Conv2d(3*self.dim_head*group_head, 3*self.dim_head*group_head, kernel_size,
                         1, kernel_size//2, groups=3*self.dim_head*group_head))
            act_blocks.append(AttnMap(self.dim_head*group_head))
            qkvs.append(nn.Conv2d(dim, 3*group_head*self.dim_head, 1, 1, 0, bias=qkv_bias))
        if group_split[-1] != 0:
            self.global_q = nn.Conv2d(dim, group_split[-1]*self.dim_head, 1, 1, 0, bias=qkv_bias)
            self.global_kv = nn.Conv2d(dim, group_split[-1]*self.dim_head*2, 1, 1, 0, bias=qkv_bias)
            self.avgpool = nn.AvgPool2d(window_size, window_size) if window_size!=1 else nn.Identity()

        self.convs = nn.ModuleList(convs)
        self.act_blocks = nn.ModuleList(act_blocks)
        self.qkvs = nn.ModuleList(qkvs)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def high_fre_attntion(self, x: torch.Tensor, to_qkv: nn.Module, mixer: nn.Module, attn_block: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()
        qkv = to_qkv(x) #(b (3 m d) h w)
        qkv = mixer(qkv).reshape(b, 3, -1, h, w).transpose(0, 1).contiguous() #(3 b (m d) h w)
        q, k, v = qkv #(b (m d) h w)
        attn = attn_block(q.mul(k)).mul(self.scalor)
        attn = self.attn_drop(torch.tanh(attn))
        res = attn.mul(v) #(b (m d) h w)
        return res
        
    def low_fre_attention(self, x : torch.Tensor, to_q: nn.Module, to_kv: nn.Module, avgpool: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()
        
        q = to_q(x).reshape(b, -1, self.dim_head, h*w).transpose(-1, -2).contiguous() #(b m (h w) d)
        kv = avgpool(x) #(b c h w)
        kv = to_kv(kv).view(b, 2, -1, self.dim_head, (h*w)//(self.window_size**2)).permute(1, 0, 2, 4, 3).contiguous() #(2 b m (H W) d)
        k, v = kv #(b m (H W) d)
        attn = self.scalor * q @ k.transpose(-1, -2) #(b m (h w) (H W))
        attn = self.attn_drop(attn.softmax(dim=-1))
        res = attn @ v #(b m (h w) d)
        res = res.transpose(2, 3).reshape(b, -1, h, w).contiguous()
        return res

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        res = []
        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue
            res.append(self.high_fre_attntion(x, self.qkvs[i], self.convs[i], self.act_blocks[i]))
        if self.group_split[-1] != 0:
            res.append(self.low_fre_attention(x, self.global_q, self.global_kv, self.avgpool))
        return self.proj_drop(self.proj(torch.cat(res, dim=1)))

# ---- 原样迁移自 nn/extra_modules/block.py:1130-1140 ----
class Bottleneck_CloAtt(Bottleneck):
    """Standard bottleneck With CloAttention."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=..., e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        self.attention = EfficientAttention(c2)
        # self.attention = LSKBlock(c2)
    
    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))

# ---- 原样迁移自 nn/extra_modules/attention.py:1038-1050 ----
class deformable_LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DeformConv(dim, kernel_size=(5, 5), padding=2, groups=dim)
        self.conv_spatial = DeformConv(dim, kernel_size=(7, 7), stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn

# ---- 原样迁移自 nn/extra_modules/block.py:2504-2509 ----
class Bottleneck_DLKA(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = deformable_LKA(c2)

# ---- 原样迁移自 nn/extra_modules/attention.py:1385-1482 ----
class FocusedLinearAttention(nn.Module):
    def __init__(self, dim, resolution, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None, focusing_factor=3, kernel_size=5):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # self.scale = qk_scale or head_dim ** -0.5
        H_sp, W_sp = self.resolution[0], self.resolution[1]
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.conv_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, self.H_sp * self.W_sp, dim)))

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        # x = x.reshape(-1, self.H_sp * self.W_sp, C).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, C // self.num_heads, H_sp * W_sp).permute(0, 2, 1).contiguous()

        x = x.reshape(-1, C, self.H_sp * self.W_sp).permute(0, 2, 1).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B C H W
        """
        qkv = self.conv_qkv(qkv)
        q, k, v = torch.chunk(qkv.flatten(2).transpose(1, 2), 3, dim=-1)

        ### Img2Window
        H, W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        k = k + self.positional_encoding
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        feature_map = rearrange(v, "b (h w) c -> b c h w", h=self.H_sp, w=self.W_sp)
        feature_map = rearrange(self.dwc(feature_map), "b c h w -> b (h w) c")
        x = x + feature_map
        x = x + lepe
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = windows2img(x, self.H_sp, self.W_sp, H, W).permute(0, 3, 1, 2)
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:3096-3105 ----
class Bottleneck_FocusedLinearAttention(Bottleneck):
    """Standard bottleneck with FocusedLinearAttention."""

    def __init__(self, c1, c2, fmapsize, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.attention = FocusedLinearAttention(c2, fmapsize)
    
    def forward(self, x):
        return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))

# ---- 原样迁移自 nn/extra_modules/PlainUSR.py:122-130 ----
class SoftPooling2D(torch.nn.Module):
    def __init__(self,kernel_size,stride=None,padding=0):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size,stride,padding, count_include_pad=False)
    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool 

# ---- 原样迁移自 nn/extra_modules/PlainUSR.py:132-154 ----
class LocalAttention(nn.Module):
    ''' attention based on local importance'''
    def __init__(self, channels, f=16):
        super().__init__()
        self.body = nn.Sequential(
            # sample importance
            nn.Conv2d(channels, f, 1),
            SoftPooling2D(7, stride=3),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(f, channels, 3, padding=1),
            # to heatmap
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )            
    def forward(self, x):
        ''' forward '''
        # interpolate the heat map
        g = self.gate(x[:,:1].clone())
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        return x * w * g #(w + g) #self.gate(x, w) 

# ---- 原样迁移自 nn/extra_modules/block.py:15476-15489 ----
class Bottleneck_RepMBConv_LA(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        self.cv1 = RepMBConv(c1)
        self.cv2 = RepMBConv(c2)
        self.add = shortcut and c1 == c2
        self.attention = LocalAttention(c2)

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return self.attention(x + self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))

# ---- 原样迁移自 nn/extra_modules/block.py:3794-3798 ----
class C3k_AggregatedAtt(C3k):
    def __init__(self, c1, c2, n=1, input_resolution=None, sr_ratio=None, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_AggregatedAttention(c_, c_, input_resolution, sr_ratio, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:3800-3803 ----
class C3k2_AggregatedAtt(C3k2):
    def __init__(self, c1, c2, n=1, input_resolution=None, sr_ratio=None, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_AggregatedAtt(self.c, self.c, 2, input_resolution, sr_ratio, shortcut, g) if c3k else Bottleneck_AggregatedAttention(self.c, self.c, input_resolution, sr_ratio, shortcut, g) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:1142-1146 ----
class C3k_CloAtt(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_CloAtt(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:1148-1151 ----
class C3k2_CloAtt(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_CloAtt(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_CloAtt(self.c, self.c, shortcut, g) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:2511-2515 ----
class C3k_DLKA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DLKA(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:2517-2520 ----
class C3k2_DLKA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DLKA(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DLKA(self.c, self.c, shortcut, g) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:3113-3116 ----
class C3k2_FocusedLinearAttention(C3k2):
    def __init__(self, c1, c2, n=1, fmapsize=None, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(Bottleneck_FocusedLinearAttention(self.c, self.c, fmapsize, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:15504-15509 ----
class C3k_RMBC_LA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        # kernel_size可选7,11,23,35
        self.m = nn.Sequential(*(Bottleneck_RepMBConv_LA(c_, c_, shortcut=shortcut, g=g) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:15511-15515 ----
class C3k2_RMBC_LA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        # kernel_size可选7,11,23,35
        self.m = nn.ModuleList(C3k_RMBC_LA(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_RepMBConv_LA(self.c, self.c, shortcut=shortcut, g=g) for _ in range(n))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- C3k2_FocusedLinearAttention ----
    try:
        module = C3k2_FocusedLinearAttention(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_FocusedLinearAttention  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_FocusedLinearAttention  自测跳过: {e}' + RESET)
    # ---- C3k_RMBC_LA ----
    try:
        module = C3k_RMBC_LA(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k_RMBC_LA  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k_RMBC_LA  自测跳过: {e}' + RESET)
    # ---- C3k2_RMBC_LA ----
    try:
        module = C3k2_RMBC_LA(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_RMBC_LA  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_RMBC_LA  自测跳过: {e}' + RESET)

