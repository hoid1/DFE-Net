'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:14173-14194 , nn/extra_modules/block.py:14196-14218 , nn/extra_modules/block.py:14220-14243 , nn/extra_modules/block.py:14271-14296
C 组：0526 版独立模块，注册方式见 tasks.py 中 legacy0526 分支。
包含：FCM_1, FCM_2, FCM_3, Pzconv
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
import torch
from ultralytics.nn.extra_modules.module.FCM import Channel, Spatial
from ultralytics.nn.modules.conv import Conv


# ---- 原样迁移自 nn/extra_modules/block.py:14220-14243 ----
class FCM_1(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.one = dim // 4
        self.two = dim - dim // 4
        self.conv1 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim // 4, dim, 1, 1)
        self.conv2 = Conv(dim - dim // 4, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44

        return x5

# ---- 原样迁移自 nn/extra_modules/block.py:14196-14218 ----
class FCM_2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4
        self.conv1 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim - dim // 4, dim, 1, 1)
        self.conv2 = Conv(dim // 4, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44

        return x5

# ---- 原样迁移自 nn/extra_modules/block.py:14173-14194 ----
class FCM_3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4
        self.conv1 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim - dim // 4, dim, 1, 1)
        self.conv2 = Conv(dim // 4, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44
        return x5

# ---- 原样迁移自 nn/extra_modules/block.py:14271-14296 ----
class Pzconv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(
            dim, dim, 3,
            1, 1, groups=dim
        )
        self.conv2 = Conv(dim, dim, k=1, s=1, )
        self.conv3 = nn.Conv2d(
            dim, dim, 5,
            1, 2, groups=dim
        )
        self.conv4 = Conv(dim, dim, 1, 1)
        self.conv5 = nn.Conv2d(
            dim, dim, 7,
            1, 3, groups=dim
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = x5 + x
        return x6

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- FCM_2 ----
    try:
        module = FCM_2(in_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'FCM_2  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'FCM_2  自测跳过: {e}' + RESET)
    # ---- FCM_3 ----
    try:
        module = FCM_3(in_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'FCM_3  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'FCM_3  自测跳过: {e}' + RESET)
    # ---- Pzconv ----
    try:
        module = Pzconv(in_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'Pzconv  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'Pzconv  自测跳过: {e}' + RESET)

