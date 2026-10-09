'''
本文件由BiliBili：魔傀面具整理
ultralytics/nn/module_images/PartialBlock.png
论文链接：https://arxiv.org/pdf/2303.03667

二次创新：为 PartialBlock 增加 module 关键字槽，契约 (in_dim, out_dim) -> Module，
与 ResidualBlock / C3k2_Block 的计算槽写法一致，经 partial 工厂注入。
默认 module=partial(Conv, k=3)，与改动前行为逐位等价，原有 yaml 不受影响。
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../..')

import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from ultralytics.nn.modules import Conv
from ultralytics.nn.extra_modules.conv_module.dynamic_snake_conv import DySnakeConv
from ultralytics.nn.extra_modules.module.ADIE import ADIE
from ultralytics.nn.extra_modules.mamba.SFMB import SFMB

class PartialBlock(nn.Module):
    def __init__(self, inc, ouc, module=partial(Conv, k=3), n_div=4):
        super().__init__()

        self.partial_channels = inc // n_div
        self.identity_channels = inc - self.partial_channels

        self.partial_module = module(self.partial_channels, self.partial_channels)
        # self.partial_module = DySnakeConv(self.partial_channels, self.partial_channels, 3)
        # self.partial_module = ADIE(self.partial_channels, self.partial_channels)
        # self.partial_module = SFMB(self.partial_channels)

        self.conv_adjust = Conv(inc, ouc, 1) if inc != ouc else nn.Identity()

    def forward(self, x):
        x1, x2 = torch.split(x, (self.partial_channels, self.identity_channels), 1)
        x1 = self.partial_module(x1)
        y = torch.cat([x1, x2], 1)
        y = self.conv_adjust(y)
        return y

class PartialBlock_SFMB(nn.Module):
    def __init__(self, inc, ouc, n_div=4):
        super().__init__()

        self.partial_channels = inc // n_div
        self.identity_channels = inc - self.partial_channels

        self.partial_module = SFMB(self.partial_channels)

        self.conv_adjust = Conv(inc, ouc, 1) if inc != ouc else nn.Identity()

    def forward(self, x):
        x1, x2 = torch.split(x, (self.partial_channels, self.identity_channels), 1)
        x1 = self.partial_module(x1)
        y = torch.cat([x1, x2], 1)
        y = self.conv_adjust(y)
        return y

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    from ultralytics.utils.torch_utils import select_device
    from ultralytics.nn.extra_modules.module.HPFGA import HPFGA

    device_id = '0'
    batch_size, in_channel, out_channel, height, width = 1, 128, 256, 160, 160

    torch_device = select_device(device_id)
    inputs_tensor = torch.randn((batch_size, in_channel, height, width)).to(torch_device)

    module = PartialBlock(in_channel, out_channel, n_div=4).to(torch_device)
    # module = PartialBlock(in_channel, out_channel, module=partial(HPFGA), n_div=4).to(torch_device)
    # module = PartialBlock_SFMB(in_channel, out_channel, 4).to(torch_device)
    module.eval()

    outputs = module(inputs_tensor)
    print(GREEN + f'inputs.size:{inputs_tensor.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)
