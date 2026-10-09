'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/MobileUViT.py:16-45 , nn/extra_modules/MobileUViT.py:48-71 , nn/extra_modules/ast.py:210-241 , nn/extra_modules/filc.py:145-167
0526 版其余独立 Block（MobileUViT 的 ConvUtr / AST 的 LeFF / FILC 的 WFM-FMFFN）。
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import math
import torch.nn as nn
from einops import rearrange
import torch


class Residual(nn.Module):
    """通用残差包装（迁移自旧版 nn/extra_modules/block.py，勿与 ultralytics.nn.modules.block.Residual 混淆）。"""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x



# ---- 原样迁移自 nn/extra_modules/MobileUViT.py:16-45 ----
class ConvUtr(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, kernel=3):
        super(ConvUtr, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(ch_in, ch_in, kernel_size=(kernel, kernel), groups=ch_in, padding=(kernel // 2, kernel // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                Residual(nn.Sequential(
                    nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in * 4),
                    nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
            ) for i in range(depth)]
        )
        self.up = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x

# ---- 原样迁移自 nn/extra_modules/MobileUViT.py:48-71 ----
class Embeddings(nn.Module):
    def __init__(self, inch=3, dims=[8, 16, 32], depths=[1, 1, 3], kernels=[3, 3, 7]):
        super(Embeddings, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(inch, dims[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ConvUtr(dims[0], dims[0], depth=depths[0], kernel=kernels[0])
        self.layer2 = ConvUtr(dims[0], dims[1], depth=depths[1], kernel=kernels[1])
        self.layer3 = ConvUtr(dims[1], dims[2], depth=depths[2], kernel=kernels[2])
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x0 = self.stem(x)
        x0 = self.layer1(x0)

        x1 = self.down(x0)
        x1 = self.layer2(x1)

        x2 = self.down(x1)
        x2 = self.layer3(x2)

        return x2, (x0, x1, x2)

# ---- 原样迁移自 nn/extra_modules/ast.py:210-241 ----
class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = nn.Identity()

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)

        x = self.linear2(x)
        x = self.eca(x)

        return x

# ---- 原样迁移自 nn/extra_modules/filc.py:145-167 ----
class WindowFrequencyModulation_FMFFN(nn.Module):
    def __init__(self, dim, window_size):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.ratio = 1
        self.complex_weight= nn.Parameter(torch.cat((torch.ones(self.window_size, self.window_size//2+1, self.ratio*dim, 1, dtype=torch.float32),\
        torch.zeros(self.window_size, self.window_size//2+1, self.ratio*dim, 1, dtype=torch.float32)),dim=-1))

    def forward(self, x):
        x = rearrange(x, 'b c (w1 p1) (w2 p2) -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)

        x_dtype = x.dtype
        x = x.to(torch.float32)
        
        x= torch.fft.rfft2(x,dim=(3, 4), norm='ortho')
      
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(self.window_size, self.window_size), dim=(3, 4), norm='ortho')

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b c (w1 p1) (w2 p2)')
        return x.to(x_dtype)
