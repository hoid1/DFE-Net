'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:6138-6156
自研模块公共零件：Sobel 边缘卷积（被 EIEStem / EIEM / GEIT 共用，旧库 md #157/#158/#224）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
import numpy as np
import torch



# ---- 原样迁移自 nn/extra_modules/block.py:6138-6156 ----
class SobelConv(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        
        sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_kernel_y = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        sobel_kernel_x = torch.tensor(sobel.T, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        
        self.sobel_kernel_x_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.sobel_kernel_y_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        
        self.sobel_kernel_x_conv3d.weight.data = sobel_kernel_x.clone()
        self.sobel_kernel_y_conv3d.weight.data = sobel_kernel_y.clone()
        
        self.sobel_kernel_x_conv3d.requires_grad = False
        self.sobel_kernel_y_conv3d.requires_grad = False

    def forward(self, x):
        return (self.sobel_kernel_x_conv3d(x[:, :, None, :, :]) + self.sobel_kernel_y_conv3d(x[:, :, None, :, :]))[:, :, 0]


if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 64, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = SobelConv(in_channel).to(device)

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
