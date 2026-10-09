'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:8865-8875 , nn/extra_modules/block.py:8877-8901 , nn/extra_modules/block.py:8930-8934 , nn/extra_modules/block.py:8936-8939
自研模块：CSP-MutilScaleEdgeInformationEnhance（旧库 md #208）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn.functional as F
import torch.nn as nn
import torch
from ultralytics.nn.modules.block import C3k, C3k2
from ultralytics.nn.modules.conv import Conv


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

# ---- 原样迁移自 nn/extra_modules/block.py:8877-8901 ----
class MutilScaleEdgeInformationEnhance(nn.Module):
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
        self.final_conv = Conv(inc * 2, inc)
    
    def forward(self, x):
        x_size = x.size()
        out = [self.local_conv(x)]
        for idx, f in enumerate(self.features):
            out.append(self.ees[idx](F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True)))
        return self.final_conv(torch.cat(out, 1))

# ---- 原样迁移自 nn/extra_modules/block.py:8930-8934 ----
class C3k_MutilScaleEdgeInformationEnhance(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MutilScaleEdgeInformationEnhance(c_, [3, 6, 9, 12]) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:8936-8939 ----
class C3k2_MutilScaleEdgeInformationEnhance(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MutilScaleEdgeInformationEnhance(self.c, self.c, 2, shortcut, g) if c3k else MutilScaleEdgeInformationEnhance(self.c, [3, 6, 9, 12]) for _ in range(n))


if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = C3k2_MutilScaleEdgeInformationEnhance(in_channel, out_channel).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    try:
        from calflops import calculate_flops
        flops, macs, _ = calculate_flops(model=module,
                                         input_shape=(batch_size, in_channel, height, width),
                                         output_as_string=True,
                                         output_precision=4,
                                         print_detailed=False)
    except Exception as e:
        print('calflops 不可用:', e)
    print(RESET)
