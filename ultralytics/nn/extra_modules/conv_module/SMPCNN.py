'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/SMPConv.py:118-125 , nn/extra_modules/SMPConv.py:140-161 , nn/extra_modules/SMPConv.py:164-181 , nn/extra_modules/SMPConv.py:184-203
0526 版 SMPConv 的 Block 级封装。
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

from timm.layers import DropPath
import torch.nn as nn
from ultralytics.nn.extra_modules.conv_module.SMPConv import get_bn, conv_bn
from ultralytics.nn.modules.conv import Conv


# ---- 原样迁移自 nn/extra_modules/SMPConv.py:140-161 ----
class SMPCNN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups, n_points=None, n_points_divide=4):
        super().__init__()
        self.kernel_size = kernel_size
        if n_points == None:
            n_points = int((kernel_size**2) // n_points_divide)

        padding = kernel_size // 2
        self.smp = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, dilation=1, groups=groups, n_points=n_points)
        
        self.small_kernel = 5
        # self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=self.small_kernel,
        #                            stride=stride, padding=self.small_kernel//2, groups=groups)
        self.small_conv = Conv(in_channels, out_channels, self.small_kernel, stride, self.small_kernel // 2, groups, act=False)

    def forward(self, inputs):
        out = self.smp(inputs)
        out += self.small_conv(inputs)
        return out

# ---- 原样迁移自 nn/extra_modules/SMPConv.py:118-125 ----
def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, n_points=None):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=groups, dilation=dilation,
                                         n_points=n_points)
    result.add_module('nonlinear', nn.ReLU())
    return result

# ---- 原样迁移自 nn/extra_modules/SMPConv.py:184-203 ----
class SMPBlock(nn.Module):

    def __init__(self, in_channels, dw_channels, lk_size, drop_path, n_points=None, n_points_divide=4):
        super().__init__()
        self.pw1 = conv_bn_relu(in_channels, dw_channels, 1, 1, 0, groups=1)
        self.pw2 = conv_bn(dw_channels, in_channels, 1, 1, 0, groups=1)
        self.large_kernel = SMPCNN(in_channels=dw_channels, out_channels=dw_channels, kernel_size=lk_size,
                                  stride=1, groups=dw_channels, n_points=n_points, n_points_divide=n_points_divide)
        self.lk_nonlinear = nn.ReLU()
        self.prelkb_bn = get_bn(in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # print('drop path:', self.drop_path)

    def forward(self, x):
        out = self.prelkb_bn(x)
        out = self.pw1(out)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)

# ---- 原样迁移自 nn/extra_modules/SMPConv.py:164-181 ----
class SMPCNN_ConvFFN(nn.Module):

    def __init__(self, in_channels, internal_channels, out_channels, drop_path):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.preffn_bn = get_bn(in_channels)
        # self.pw1 = conv_bn(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1, padding=0, groups=1)
        # self.pw2 = conv_bn(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.pw1 = Conv(in_channels, internal_channels, act=False)
        self.pw2 = Conv(internal_channels, out_channels, act=False)
        self.nonlinear = nn.GELU()

    def forward(self, x):
        out = self.preffn_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)
