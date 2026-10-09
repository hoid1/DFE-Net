'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:3620-3652
二次创新(2)：ASF 的 ScalSeq + DySample 动态上采样（旧库 md #99）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
import torch
from ultralytics.nn.extra_modules.upsample.DySample import DySample
from ultralytics.nn.modules.conv import Conv


# ---- 原样迁移自 nn/extra_modules/block.py:3620-3652 ----
class DynamicScalSeq(nn.Module):
    def __init__(self, inc, channel):
        super(DynamicScalSeq, self).__init__()
        if channel != inc[0]:
            self.conv0 = Conv(inc[0], channel,1)
        self.conv1 =  Conv(inc[1], channel,1)
        self.conv2 =  Conv(inc[2], channel,1)
        self.conv3d = nn.Conv3d(channel,channel,kernel_size=(1,1,1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3,1,1))
        
        self.dysample1 = DySample(channel, 2, 'lp')
        self.dysample2 = DySample(channel, 4, 'lp')

    def forward(self, x):
        p3, p4, p5 = x[0],x[1],x[2]
        if hasattr(self, 'conv0'):
            p3 = self.conv0(p3)
        p4_2 = self.conv1(p4)
        p4_2 = self.dysample1(p4_2)
        p5_2 = self.conv2(p5)
        p5_2 = self.dysample2(p5_2)
        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)
        combine = torch.cat([p3_3d, p4_3d, p5_3d],dim = 2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x

if __name__ == '__main__':
    RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    try:
        module = DynamicScalSeq([64,128,256], 64).to(device)
        inputs = [torch.randn(1,64,32,32).to(device), torch.randn(1,128,16,16).to(device), torch.randn(1,256,8,8).to(device)]
        outputs = module(inputs)
        print(GREEN + f'inputs:{[tuple(x.shape) for x in inputs]} -> outputs:{tuple(outputs.shape)}' + RESET)
    except Exception as e:
        print(RED + f'自测跳过: {e}' + RESET)
