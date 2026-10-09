'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/attention.py:1829-1845 , nn/extra_modules/attention.py:1847-1860 , nn/extra_modules/attention.py:1862-1878 , nn/extra_modules/block.py:8865-8875 , nn/extra_modules/block.py:8903-8928 , nn/extra_modules/block.py:8941-8945 , nn/extra_modules/block.py:8947-8950
二次创新(1)：自研 CSP-MSEIE + ICCV2023 DSM 双域选择机制（旧库 md #214）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn.functional as F
import torch.nn as nn
import torch
from ultralytics.nn.extra_modules.module.CSSC import ChannelPool
from ultralytics.nn.modules.block import C3k, C3k2
from ultralytics.nn.modules.conv import Conv


# ---- 原样迁移自 nn/extra_modules/attention.py:1847-1860 ----
class DSM_LocalAttention(nn.Module):
    def __init__(self, channel, p) -> None:
        super().__init__()
        self.channel = channel

        self.num_patch = 2 ** p
        self.sig = nn.Sigmoid()

        self.a = nn.Parameter(torch.zeros(channel,1,1))
        self.b = nn.Parameter(torch.ones(channel,1,1))

    def forward(self, x):
        out = x - torch.mean(x, dim=(2,3), keepdim=True)
        return self.a*out*x + self.b*x

# ---- 原样迁移自 nn/extra_modules/attention.py:1829-1845 ----
class DSM_SpatialGate(nn.Module):
    def __init__(self, channel):
        super(DSM_SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, kernel_size, act=False)
        self.dw1 = nn.Sequential(
            Conv(channel, channel, 5, s=1, d=2, g=channel, act=nn.GELU()),
            Conv(channel, channel, 7, s=1, d=3, g=channel, act=nn.GELU())
        )
        self.dw2 = Conv(channel, channel, kernel_size, g=channel, act=nn.GELU())

    def forward(self, x):
        out = self.compress(x)
        out = self.spatial(out)
        out = self.dw1(x) * out + self.dw2(x)
        return out

# ---- 原样迁移自 nn/extra_modules/attention.py:1862-1878 ----
class DualDomainSelectionMechanism(nn.Module):
    # https://openaccess.thecvf.com/content/ICCV2023/papers/Cui_Focal_Network_for_Image_Restoration_ICCV_2023_paper.pdf
    # https://github.com/c-yn/FocalNet
    # Dual-DomainSelectionMechanism
    def __init__(self, channel) -> None:
        super().__init__()
        pyramid = 1
        self.spatial_gate = DSM_SpatialGate(channel)
        layers = [DSM_LocalAttention(channel, p=i) for i in range(pyramid-1,-1,-1)]
        self.local_attention = nn.Sequential(*layers)
        self.a = nn.Parameter(torch.zeros(channel,1,1))
        self.b = nn.Parameter(torch.ones(channel,1,1))
        
    def forward(self, x):
        out = self.spatial_gate(x)
        out = self.local_attention(out)
        return self.a*out + self.b*x

# ---- 原样迁移自 nn/extra_modules/block.py:8865-8875 ----
class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = Conv(in_dim, in_dim, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride= 1, padding = 1)
    
    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge

# ---- 原样迁移自 nn/extra_modules/block.py:8903-8928 ----
class MutilScaleEdgeInformationSelect(nn.Module):
    def __init__(self, inc, bins):
        super().__init__()
        
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                Conv(inc, inc // len(bins), 1),
                Conv(inc // len(bins), inc // len(bins), 3, g=inc // len(bins))
            ))
        self.ees = []
        for _ in bins:
            self.ees.append(EdgeEnhancer(inc // len(bins)))
        self.features = nn.ModuleList(self.features)
        self.ees = nn.ModuleList(self.ees)
        self.local_conv = Conv(inc, inc, 3)
        self.dsm = DualDomainSelectionMechanism(inc * 2)
        self.final_conv = Conv(inc * 2, inc)
    
    def forward(self, x):
        x_size = x.size()
        out = [self.local_conv(x)]
        for idx, f in enumerate(self.features):
            out.append(self.ees[idx](F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True)))
        return self.final_conv(self.dsm(torch.cat(out, 1)))

# ---- 原样迁移自 nn/extra_modules/block.py:8941-8945 ----
class C3k_MutilScaleEdgeInformationSelect(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MutilScaleEdgeInformationSelect(c_, [3, 6, 9, 12]) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:8947-8950 ----
class C3k2_MutilScaleEdgeInformationSelect(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MutilScaleEdgeInformationSelect(self.c, self.c, 2, shortcut, g) if c3k else MutilScaleEdgeInformationSelect(self.c, [3, 6, 9, 12]) for _ in range(n))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- EdgeEnhancer ----
    try:
        module = EdgeEnhancer(in_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'EdgeEnhancer  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'EdgeEnhancer  自测跳过: {e}' + RESET)
    # ---- C3k_MutilScaleEdgeInformationSelect ----
    try:
        module = C3k_MutilScaleEdgeInformationSelect(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k_MutilScaleEdgeInformationSelect  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k_MutilScaleEdgeInformationSelect  自测跳过: {e}' + RESET)
    # ---- C3k2_MutilScaleEdgeInformationSelect ----
    try:
        module = C3k2_MutilScaleEdgeInformationSelect(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_MutilScaleEdgeInformationSelect  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_MutilScaleEdgeInformationSelect  自测跳过: {e}' + RESET)

