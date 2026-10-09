'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:5254-5281
0526 版 FocusFeature（旧库 md #135）。新版 neck/FDPN.py 的同名类类体已有改动。
yaml 中请使用 FocusFeature_0526。
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
import torch
from ultralytics.nn.modules.block import ADown
from ultralytics.nn.modules.conv import Conv, autopad


# ---- 原样迁移自 nn/extra_modules/block.py:5254-5281 ----
class FocusFeature(nn.Module):
    def __init__(self, inc, kernel_sizes=(5, 7, 9, 11), e=0.5) -> None:
        super().__init__()
        hidc = int(inc[1] * e)
        
        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(inc[0], hidc, 1)
        )
        self.conv2 = Conv(inc[1], hidc, 1) if e != 1 else nn.Identity()
        self.conv3 = ADown(inc[2], hidc)
        
        
        self.dw_conv = nn.ModuleList(nn.Conv2d(hidc * 3, hidc * 3, kernel_size=k, padding=autopad(k), groups=hidc * 3) for k in kernel_sizes)
        self.pw_conv = Conv(hidc * 3, hidc * 3)
    
    def forward(self, x):
        x1, x2, x3 = x
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        
        x = torch.cat([x1, x2, x3], dim=1)
        feature = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv], dim=0), dim=0)
        feature = self.pw_conv(feature)
        
        x = x + feature
        return x

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- FocusFeature ----
    try:
        module = FocusFeature(in_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'FocusFeature  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'FocusFeature  自测跳过: {e}' + RESET)

