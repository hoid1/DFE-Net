'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/mhafyolo.py:375-407 , nn/extra_modules/mhafyolo.py:410-454 , nn/extra_modules/mhafyolo.py:52-70 , nn/extra_modules/mhafyolo.py:72-102
0526 版 MAF-YOLO 的 RepHDW 系组件。
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
import torch
from ultralytics.nn.extra_modules.block.RepHMS import UniRepLKNetBlock
from ultralytics.nn.modules.conv import Conv, DWConv


# ---- 原样迁移自 nn/extra_modules/mhafyolo.py:375-407 ----
class DepthBottleneckv2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 kersize=5,
                 expansion_depth=1,
                 small_kersize=3,
                 use_depthwise=True):
        super(DepthBottleneckv2, self).__init__()

        mid_channel = int(in_channels * expansion_depth)
        mid_channel2 = mid_channel
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut
        if use_depthwise:
            self.conv2 = DWConv(mid_channel, mid_channel, kersize)
            # self.act = nn.SiLU()
            self.one_conv = Conv(mid_channel, mid_channel2, kernel_size=1)

            self.conv3 = DWConv(mid_channel2, mid_channel2, kersize)
            # self.act1 = nn.SiLU()
            self.one_conv2 = Conv(mid_channel2, out_channels, kernel_size=1)
        else:
            self.conv2 = Conv(out_channels, out_channels, 3, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.one_conv(y)
        y = self.conv3(y)
        y = self.one_conv2(y)
        return y

# ---- 原样迁移自 nn/extra_modules/mhafyolo.py:410-454 ----
class ConvMS(nn.Module):
    def __init__(self, in_channels, out_channels, width=3, depth=1, depth_expansion=2, kersize=5, shortcut=True,
                 expansion=0.5,
                 small_kersize=3, use_depthwise=True):
        super(ConvMS, self).__init__()
        self.width = width
        self.depth = depth
        c1 = int(out_channels * expansion) * width
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.RepElanMSBlock = nn.ModuleList()
        for _ in range(width - 1):
            DepthBlock = nn.ModuleList([
                DepthBottleneckv2(self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise)
                for _ in range(depth)
            ])
            self.RepElanMSBlock.append(DepthBlock)

        self.conv2 = Conv(c_ * 1 + c_ * (width - 1) * depth, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = [x[:, i * self.c_:(i + 1) * self.c_] for i in range(self.width)]
        x_out[1] = x_out[1] + x_out[0]
        cascade = []
        elan = [x_out[0]]
        for i in range(self.width - 1):
            for j in range(self.depth):
                if i > 0:
                    x_out[i + 1] = x_out[i + 1] + cascade[j]
                    if j == self.depth - 1:
                        # cascade = [cascade[-1]]
                        if self.depth > 1:
                            cascade = [cascade[-1]]
                        else:
                            cascade = []
                x_out[i + 1] = self.RepElanMSBlock[i][j](x_out[i + 1])
                elan.append(x_out[i + 1])
                if i < self.width - 2:
                    cascade.append(x_out[i + 1])

        y_out = torch.cat(elan, 1)
        y_out = self.conv2(y_out)
        return y_out

# ---- 原样迁移自 nn/extra_modules/mhafyolo.py:72-102 ----
class DepthBottleneckUni(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 kersize = 5,
                 expansion_depth = 1,
                 small_kersize = 3,
                 use_depthwise=True):
        super(DepthBottleneckUni, self).__init__()


        mid_channel = int(in_channels * expansion_depth)
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut
        if use_depthwise:

            self.conv2 = UniRepLKNetBlock(mid_channel, kernel_size=kersize)
            self.act = nn.SiLU()
            self.one_conv = Conv(mid_channel,out_channels,kernel_size = 1)

        else:
            self.conv2 = Conv(out_channels, out_channels, 3, 1)

    def forward(self, x):
        y = self.conv1(x)

        y = self.act(self.conv2(y))

        y = self.one_conv(y)
        return y

# ---- 原样迁移自 nn/extra_modules/mhafyolo.py:52-70 ----
class RepHDW(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, shortcut = True, expansion = 0.5, kersize = 5,depth_expansion = 1,small_kersize = 3,use_depthwise = True):
        super(RepHDW, self).__init__()
        c1 = int(out_channels * expansion) * 2
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.m = nn.ModuleList(DepthBottleneckUni(self.c_, self.c_, shortcut,kersize,depth_expansion,small_kersize,use_depthwise) for _ in range(depth))
        self.conv2 = Conv(c_ * (depth+2), out_channels, 1, 1)

    def forward(self,x):
        x = self.conv1(x)
        x_out = list(x.split((self.c_, self.c_), 1))
        for conv in self.m:
            y = conv(x_out[-1])
            x_out.append(y)
        y_out = torch.cat(x_out, axis=1)
        y_out = self.conv2(y_out)
        return  y_out
