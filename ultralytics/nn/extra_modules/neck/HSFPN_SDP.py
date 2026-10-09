'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/HSFPN.py:179-203
0526 版 HSFPN 的 SDP_Improved 模块。
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
import numpy as np
from einops import rearrange
import torch



# ---- 原样迁移自 nn/extra_modules/HSFPN.py:179-203 ----
class SDP_Improved(nn.Module):
    def __init__(self,
                dim=256,
                inter_dim=None):
        super(SDP_Improved, self).__init__()
        self.inter_dim=inter_dim
        if self.inter_dim == None:
            self.inter_dim = dim
        self.conv_q = nn.Sequential(*[nn.Conv2d(dim, self.inter_dim, 3, padding=1, bias=False), nn.GroupNorm(32,self.inter_dim)])
        self.conv_k = nn.Sequential(*[nn.Conv2d(dim, self.inter_dim, 3, padding=1, bias=False), nn.GroupNorm(32,self.inter_dim)])
        self.conv = nn.Sequential(*[nn.Conv2d(self.inter_dim, dim, 3, padding=1, bias=False), nn.GroupNorm(32, dim)])
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x_low, x_high, patch_size):
        b_, _, h_, w_ = x_low.size()
        q = rearrange(self.conv_q(x_low), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=patch_size[0], p2=patch_size[1])
        q = q.transpose(1,2) # 1,4096,128
        k = rearrange(self.conv_k(x_high), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=patch_size[0], p2=patch_size[1])
        attn = torch.matmul(q, k) # 1, 4096, 1024
        attn = attn / np.power(self.inter_dim, 0.5)
        attn = self.softmax(attn)
        v = k.transpose(1,2)# 1, 1024, 128
        output = torch.matmul(attn,v)# 1, 4096, 128
        output = rearrange(output.transpose(1, 2).contiguous(), '(b h w) c (p1 p2) -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=h_//patch_size[0], w=w_//patch_size[1])
        output = self.conv(output + x_low)
        return output
