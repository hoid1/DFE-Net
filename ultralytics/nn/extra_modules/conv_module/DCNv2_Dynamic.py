'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/attention.py:985-1009 , nn/extra_modules/block.py:1808-1820 , nn/extra_modules/block.py:1822-1877 , nn/extra_modules/block.py:1879-1885 , nn/extra_modules/block.py:1887-1891 , nn/extra_modules/block.py:1893-1896
自研模块：MPCA(MultiPath Coordinate Attention) 强化 DCNV2 的 offset/mask（旧库 md #42）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import math
import torch.nn as nn
import torch
from ultralytics.nn.modules.block import Bottleneck, C3k, C3k2
from ultralytics.nn.modules.conv import Conv, autopad


# ---- 原样迁移自 nn/extra_modules/attention.py:985-1009 ----
class MPCA(nn.Module):
    # MultiPath Coordinate Attention
    def __init__(self, channels) -> None:
        super().__init__()
        
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv(channels, channels)
        )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_hw = Conv(channels, channels, (3, 1))
        self.conv_pool_hw = Conv(channels, channels, 1)
    
    def forward(self, x):
        _, _, h, w = x.size()
        x_pool_h, x_pool_w, x_pool_ch = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2), self.gap(x)
        x_pool_hw = torch.cat([x_pool_h, x_pool_w], dim=2)
        x_pool_hw = self.conv_hw(x_pool_hw)
        x_pool_h, x_pool_w = torch.split(x_pool_hw, [h, w], dim=2)
        x_pool_hw_weight = self.conv_pool_hw(x_pool_hw).sigmoid()
        x_pool_h_weight, x_pool_w_weight = torch.split(x_pool_hw_weight, [h, w], dim=2)
        x_pool_h, x_pool_w = x_pool_h * x_pool_h_weight, x_pool_w * x_pool_w_weight
        x_pool_ch = x_pool_ch * torch.mean(x_pool_hw_weight, dim=2, keepdim=True)
        return x * x_pool_h.sigmoid() * x_pool_w.permute(0, 1, 3, 2).sigmoid() * x_pool_ch.sigmoid()

# ---- 原样迁移自 nn/extra_modules/block.py:1808-1820 ----
class DCNv2_Offset_Attention(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, deformable_groups=1) -> None:
        super().__init__()
        
        padding = autopad(kernel_size, None, 1)
        self.out_channel = (deformable_groups * 3 * kernel_size * kernel_size)
        self.conv_offset_mask = nn.Conv2d(in_channels, self.out_channel, kernel_size, stride, padding, bias=True)
        self.attention = MPCA(self.out_channel)
        
    def forward(self, x):
        conv_offset_mask = self.conv_offset_mask(x)
        conv_offset_mask = self.attention(conv_offset_mask)
        return conv_offset_mask

# ---- 原样迁移自 nn/extra_modules/block.py:1822-1877 ----
class DCNv2_Dynamic(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, groups=1, dilation=1, act=True, deformable_groups=1):
        super(DCNv2_Dynamic, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        padding = autopad(kernel_size, padding, dilation)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        self.conv_offset_mask = DCNv2_Offset_Attention(in_channels, kernel_size, stride, deformable_groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.conv_offset_mask.bias.data.zero_()

# ---- 原样迁移自 nn/extra_modules/block.py:1879-1885 ----
class Bottleneck_DCNV2_Dynamic(Bottleneck):
    """Standard bottleneck with DCNV2."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DCNv2_Dynamic(c_, c2, k[1], 1)

# ---- 原样迁移自 nn/extra_modules/block.py:1887-1891 ----
class C3k_DCNv2_Dynamic(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DCNV2_Dynamic(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:1893-1896 ----
class C3k2_DCNv2_Dynamic(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DCNv2_Dynamic(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DCNV2_Dynamic(self.c, self.c, shortcut, g) for _ in range(n))


if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = C3k2_DCNv2_Dynamic(in_channel, out_channel).to(device)

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
