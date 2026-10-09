'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:11146-11159 , nn/extra_modules/block.py:11270-11318 , nn/extra_modules/block.py:1682-1747 , nn/extra_modules/mambaIR.py:365-511 , nn/extra_modules/mambaIR.py:66-78 , nn/extra_modules/mambaIR.py:81-107 , nn/extra_modules/transformer.py:40-45 , nn/extra_modules/transformer.py:47-76 , nn/extra_modules/tsdn.py:108-139 , nn/extra_modules/tsdn.py:142-190 , nn/extra_modules/tsdn.py:209-230
C 组：0526 版独立模块，注册方式见 tasks.py 中 legacy0526 分支。
包含：AIFI_RepBN, DTAB, ETB, FDT, AttentiveLayer, FocalModulation
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
from ultralytics.nn.extra_modules.norm.repbn import RepBN, LinearNorm
from functools import partial
ln = nn.LayerNorm
linearnorm = partial(LinearNorm, norm1=ln, norm2=RepBN, step=60000)
from einops import rearrange
from timm.layers import to_2tuple
import torch
from ultralytics.nn.extra_modules.mamba.ASSM import ASSM
from ultralytics.nn.extra_modules.mamba.PatchMamba import FeedForward
from ultralytics.nn.extra_modules.mamba.TransMixer import LayerNorm
from ultralytics.nn.extra_modules.module.FATBlock import WindowAttention
from ultralytics.nn.extra_modules.module.LEGM import window_partition, window_reverse
from ultralytics.nn.extra_modules.transformer.DilatedGCSA import FixedPosEmb, RelPosEmb
from ultralytics.nn.extra_modules.transformer.FSSA import Attention_F, Attention_S
from ultralytics.nn.extra_modules.transformer.GSA import GSA
from ultralytics.nn.extra_modules.transformer.RSA import RSA
from ultralytics.nn.modules.transformer import TransformerEncoderLayer


# ---- 原样迁移自 nn/extra_modules/transformer.py:40-45 ----
class TransformerEncoderLayer_RepBN(TransformerEncoderLayer):
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=..., normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        
        self.norm1 = linearnorm(c1)
        self.norm2 = linearnorm(c1)

# ---- 原样迁移自 nn/extra_modules/transformer.py:47-76 ----
class AIFI_RepBN(TransformerEncoderLayer_RepBN):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding."""
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]

# ---- 原样迁移自 nn/extra_modules/mambaIR.py:66-78 ----
class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)  # DW Conv

    def forward(self, x, H, W):
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W)).flatten(2).transpose(-1,
                                                                                                              -2).contiguous()

        return x1 * x2

# ---- 原样迁移自 nn/extra_modules/mambaIR.py:81-107 ----
class GatedMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = Gate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, x_size):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        H, W = x_size
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x

# ---- 原样迁移自 nn/extra_modules/mambaIR.py:365-511 ----
class AttentiveLayer(nn.Module):
    def __init__(self,
                 dim,
                 input_size,
                 d_state=8,
                 num_heads=4,
                 window_size=4,
                 shift_size=2,
                 inner_rank=32,
                 num_tokens=64,
                 convffn_kernel_size=5,
                 mlp_ratio=1,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.num_tokens = num_tokens
        self.softmax = nn.Softmax(dim=-1)
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.inner_rank = inner_rank

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)

        layer_scale = 1e-4
        self.scale1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        self.scale2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self.win_mhsa = WindowAttention(
            self.dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.assm = ASSM(
            self.dim,
            d_state,
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            mlp_ratio=mlp_ratio
        )

        mlp_hidden_dim = int(dim * self.mlp_ratio)

        self.convffn1 = GatedMLP(in_features=dim,hidden_features=mlp_hidden_dim,out_features=dim)
        self.convffn2 =  GatedMLP(in_features=dim,hidden_features=mlp_hidden_dim,out_features=dim)

        self.embeddingA = nn.Embedding(self.inner_rank, d_state)
        self.embeddingA.weight.data.uniform_(-1 / self.inner_rank, 1 / self.inner_rank)

        # self.attn_mask = self.calculate_mask(input_size)
        # self.rpi = self.calculate_rpi_sa()

        self.register_buffer('attn_mask', self.calculate_mask(input_size))
        self.register_buffer('rpi', self.calculate_rpi_sa())

    def calculate_rpi_sa(self):
        # calculate relative position index for SW-MSA
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        # h, w = x_size
        # b, n, c = x.shape
        # c3 = 3 * c

        b, c, h, w = x.size()
        x_size = (h, w)
        n = h * w
        x = x.flatten(2).permute(0, 2, 1).contiguous() # b h*w c
        c3 = 3 * c

        # part1: Window-MHSA
        shortcut = x
        x = self.norm1(x)
        qkv = self.wqkv(x)
        qkv = qkv.reshape(b, h, w, c3)
        if self.shift_size > 0:
            shifted_qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self.attn_mask
        else:
            shifted_qkv = qkv
            attn_mask = None
        x_windows = window_partition(shifted_qkv, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c3)
        attn_windows = self.win_mhsa(x_windows, rpi=self.rpi, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        x_win = attn_x.view(b, n, c) + shortcut
        x_win = self.convffn1(self.norm2(x_win), x_size) + x_win
        x = shortcut * self.scale1 + x_win

        # part2: Attentive State Space
        shortcut = x
        x_aca = self.assm(self.norm3(x), x_size, self.embeddingA) + x
        x = x_aca + self.convffn2(self.norm4(x_aca), x_size)
        x = shortcut * self.scale2 + x

        # print(x.size(), b, h, w, c)
        return x.permute(0, 2, 1).reshape(b, c, h, w).contiguous()

# ---- 原样迁移自 nn/extra_modules/tsdn.py:108-139 ----
class DilatedMDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(DilatedMDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

# ---- 原样迁移自 nn/extra_modules/tsdn.py:142-190 ----
class DilatedOCA(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_heads, dim_head, bias):
        super(DilatedOCA, self).__init__()
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_spatial_heads
        self.scale = self.dim_head**-0.5

        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)
        self.qkv = nn.Conv2d(self.dim, self.inner_dim*3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        self.rel_pos_emb = RelPosEmb(
            block_size = window_size,
            rel_size = window_size + (self.overlap_win_size - window_size),
            dim_head = self.dim_head
        )
        self.fixed_pos_emb = FixedPosEmb(window_size, self.overlap_win_size)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qs, ks, vs = qkv.chunk(3, dim=1)

        # spatial attention
        qs = rearrange(qs, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = self.window_size, p2 = self.window_size)
        ks, vs = map(lambda t: self.unfold(t), (ks, vs))
        ks, vs = map(lambda t: rearrange(t, 'b (c j) i -> (b i) j c', c = self.inner_dim), (ks, vs))

        # print(f'qs.shape:{qs.shape}, ks.shape:{ks.shape}, vs.shape:{vs.shape}')
        #split heads
        qs, ks, vs = map(lambda t: rearrange(t, 'b n (head c) -> (b head) n c', head = self.num_spatial_heads), (qs, ks, vs))

        # attention
        qs = qs * self.scale
        spatial_attn = (qs @ ks.transpose(-2, -1))
        spatial_attn += self.rel_pos_emb(qs)
        spatial_attn += self.fixed_pos_emb()
        spatial_attn = spatial_attn.softmax(dim=-1)

        out = (spatial_attn @ vs)

        out = rearrange(out, '(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)', head = self.num_spatial_heads, h = h // self.window_size, w = w // self.window_size, p1 = self.window_size, p2 = self.window_size)

        # merge spatial and channel
        out = self.project_out(out)

        return out

# ---- 原样迁移自 nn/extra_modules/tsdn.py:209-230 ----
class DTAB(nn.Module):
    def __init__(self, dim, window_size=4, overlap_ratio=0.5, num_channel_heads=4, num_spatial_heads=2, spatial_dim_head=16, ffn_expansion_factor=1, bias=False, LayerNorm_type='BiasFree'):
        super(DTAB, self).__init__()

        self.spatial_attn = DilatedOCA(dim, window_size, overlap_ratio, num_spatial_heads, spatial_dim_head, bias)
        self.channel_attn = DilatedMDTA(dim, num_channel_heads, bias)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.norm4 = LayerNorm(dim, LayerNorm_type)

        self.channel_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.spatial_ffn = FeedForward(dim, ffn_expansion_factor, bias)


    def forward(self, x):
        x = x + self.channel_attn(self.norm1(x))
        x = x + self.channel_ffn(self.norm2(x))
        x = x + self.spatial_attn(self.norm3(x))
        x = x + self.spatial_ffn(self.norm4(x))
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:11146-11159 ----
class ETB(nn.Module):
    def __init__(self, dim=128, num_heads=4, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super(ETB, self).__init__()
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn_S = Attention_S(dim, num_heads, bias)
        self.attn_F = Attention_F(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + torch.add(self.attn_F(self.norm1(x)),self.attn_S(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:11270-11318 ----
class FDT(nn.Module):
    def __init__(self, inp_channels, num_heads=4, window_sizes=4, shifts=0, shared_depth=1, ffn_expansion_factor=2.66):
        super(FDT, self).__init__()
        self.shared_depth = shared_depth
        
        modules_ffd = {}
        modules_att = {} 
        modules_norm = {} 
        for i in range(shared_depth):
            modules_ffd['ffd{}'.format(i)] = FeedForward(inp_channels, ffn_expansion_factor, bias=False)
            modules_att['att_{}'.format(i)] = RSA(channels=inp_channels, num_heads=num_heads, shifts=shifts, window_sizes=window_sizes)
            modules_norm['norm_{}'.format(i)] = LayerNorm(inp_channels, 'WithBias')
            modules_norm['norm_{}'.format(i+2)] = LayerNorm(inp_channels, 'WithBias')
        self.modules_ffd = nn.ModuleDict(modules_ffd)
        self.modules_att = nn.ModuleDict(modules_att)
        self.modules_norm = nn.ModuleDict(modules_norm)

        modulec_ffd = {}
        modulec_att = {} 
        modulec_norm = {} 
        for i in range(shared_depth):
            modulec_ffd['ffd{}'.format(i)] = FeedForward(inp_channels, ffn_expansion_factor, bias=False)
            modulec_att['att_{}'.format(i)] = GSA(channels=inp_channels, num_heads=num_heads)
            modulec_norm['norm_{}'.format(i)] = LayerNorm(inp_channels, 'WithBias')
            modulec_norm['norm_{}'.format(i+2)] = LayerNorm(inp_channels, 'WithBias')
        self.modulec_ffd = nn.ModuleDict(modulec_ffd)
        self.modulec_att = nn.ModuleDict(modulec_att)
        self.modulec_norm = nn.ModuleDict(modulec_norm)

    def forward(self, x):
        atn = None
        B, C, H, W = x.size()
        for i in range(self.shared_depth):
            if i == 0: ## only calculate attention for the 1-st module
                x_, atn = self.modules_att['att_{}'.format(i)](self.modules_norm['norm_{}'.format(i)](x), None)
                x = self.modules_ffd['ffd{}'.format(i)](self.modules_norm['norm_{}'.format(i+2)](x_ + x)) + x_
            else:
                x_ = self.modules_att['att_{}'.format(i)](self.modules_norm['norm_{}'.format(i)](x), atn)
                x = self.modules_ffd['ffd{}'.format(i)](self.modules_norm['norm_{}'.format(i+2)](x_ + x)) + x_

        for i in range(self.shared_depth):
            if i == 0: ## only calculate attention for the 1-st module
                x_, atn = self.modulec_att['att_{}'.format(i)](self.modulec_norm['norm_{}'.format(i)](x), None)
                x = self.modulec_ffd['ffd{}'.format(i)](self.modulec_norm['norm_{}'.format(i+2)](x_ + x)) + x_
            else:
                x = self.modulec_att['att_{}'.format(i)](self.modulec_norm['norm_{}'.format(i)](x), atn)
                x = self.modulec_ffd['ffd{}'.format(i)](self.modulec_norm['norm_{}'.format(i+2)](x_ + x)) + x_
        
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:1682-1747 ----
class FocalModulation(nn.Module):
    def __init__(self, dim, focal_window=3, focal_level=2, focal_factor=2, bias=True, proj_drop=0., use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f_linear = nn.Conv2d(dim, 2 * dim + (self.focal_level + 1), kernel_size=1, bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()
                
        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, 
                    groups=dim, padding=kernel_size//2, bias=False),
                    nn.GELU(),
                    )
                )              
            self.kernel_sizes.append(kernel_size)          
        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        C = x.shape[1]

        # pre linear projection
        x = self.f_linear(x).contiguous()
        q, ctx, gates = torch.split(x, (C, C, self.focal_level+1), 1)
        
        # context aggreation
        ctx_all = 0.0
        for l in range(self.focal_level):         
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * gates[:, l:l+1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        # focal modulation
        x_out = q * self.h(ctx_all)
        x_out = x_out.contiguous()
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)
        
        # post linear porjection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- Gate ----
    try:
        module = Gate(in_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'Gate  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'Gate  自测跳过: {e}' + RESET)
    # ---- DTAB ----
    try:
        module = DTAB(in_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'DTAB  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'DTAB  自测跳过: {e}' + RESET)
    # ---- FocalModulation ----
    try:
        module = FocalModulation(in_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'FocalModulation  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'FocalModulation  自测跳过: {e}' + RESET)

