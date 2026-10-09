'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:9873-9887 , nn/extra_modules/block.py:9889-9900
自研模块：GlobalEdgeInformationTransfer（旧库 md #224）
MutilScaleEdgeInfoGenetator 为多输出模块，yaml 中配合 GetIndexOutput 使用
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
import torch
from ultralytics.nn.extra_modules.conv_module.SobelConv import SobelConv
from ultralytics.nn.modules.conv import Conv


# ---- 原样迁移自 nn/extra_modules/block.py:9889-9900 ----
class ConvEdgeFusion(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()
        
        self.conv_channel_fusion = Conv(sum(inc), ouc // 2, k = 1)
        self.conv_3x3_feature_extract = Conv(ouc // 2, ouc // 2, 3)
        self.conv_1x1 = Conv(ouc // 2, ouc, 1)
    
    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = self.conv_1x1(self.conv_3x3_feature_extract(self.conv_channel_fusion(x)))
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:9873-9887 ----
class MutilScaleEdgeInfoGenetator(nn.Module):
    def __init__(self, inc, oucs) -> None:
        super().__init__()
        
        self.sc = SobelConv(inc)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1x1s = nn.ModuleList(Conv(inc, ouc, 1) for ouc in oucs)
    
    def forward(self, x):
        outputs = [self.sc(x)]
        outputs.extend(self.maxpool(outputs[-1]) for _ in self.conv_1x1s)
        outputs = outputs[1:]
        for i in range(len(self.conv_1x1s)):
            outputs[i] = self.conv_1x1s[i](outputs[i])
        return outputs

if __name__ == '__main__':
    RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    try:
        module = ConvEdgeFusion([64, 128], 128).to(device)
        inputs = [torch.randn(1,64,32,32).to(device), torch.randn(1,128,32,32).to(device)]
        outputs = module(inputs)
        print(GREEN + f'inputs:{[tuple(x.shape) for x in inputs]} -> outputs:{tuple(outputs.shape)}' + RESET)
    except Exception as e:
        print(RED + f'自测跳过: {e}' + RESET)
