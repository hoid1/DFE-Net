'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:13918-13930 , nn/extra_modules/block.py:13933-13956
二次创新(1)：ECCV2024 SNI + GSConvE 融入自研 SOEP（旧库 md #343/#345）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
import torch
from ultralytics.nn.modules.conv import Conv


# ---- 原样迁移自 nn/extra_modules/block.py:13933-13956 ----
class GSConvE(nn.Module):
    '''
    GSConv enhancement for representation learning: generate various receptive-fields and
    texture-features only in one Conv module
    https://github.com/AlanLi1997/slim-neck-by-gsconv
    '''
    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, d, act)
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_, c_, 3, 1, 1, bias=False),
            nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, bias=False),
            nn.GELU()
        )

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        y = torch.cat((x1, x2), dim=1)
        # shuffle
        y = y.reshape(y.shape[0], 2, y.shape[1] // 2, y.shape[2], y.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

# ---- 原样迁移自 nn/extra_modules/block.py:13918-13930 ----
class SNI(nn.Module):
    '''
    https://github.com/AlanLi1997/rethinking-fpn
    soft nearest neighbor interpolation for up-sampling
    secondary features aligned
    '''
    def __init__(self, up_f=2):
        super(SNI, self).__init__()
        self.us = nn.Upsample(None, up_f, 'nearest')
        self.alpha = 1/(up_f**2)

    def forward(self, x):
        return self.alpha*self.us(x)

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- GSConvE ----
    try:
        module = GSConvE(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'GSConvE  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'GSConvE  自测跳过: {e}' + RESET)
    # ---- SNI ----
    try:
        module = SNI().to(device)
        outputs = module(inputs)
        print(GREEN + f'SNI  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'SNI  自测跳过: {e}' + RESET)

