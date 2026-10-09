'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:6514-6518 , nn/extra_modules/block.py:6520-6553
二次创新(2)：DuAT 的 SBA 选择性边界聚合，构成 Re-CalibrationFPN（旧库 md #164）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
import torch
from ultralytics.nn.modules.conv import Conv


# ---- 原样迁移自 nn/extra_modules/block.py:6514-6518 ----
def Upsample(x, size, align_corners = False):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)

# ---- 原样迁移自 nn/extra_modules/block.py:6520-6553 ----
class SBA(nn.Module):

    def __init__(self, inc, input_dim=64):
        super().__init__()

        self.input_dim = input_dim

        self.d_in1 = Conv(input_dim//2, input_dim//2, 1)
        self.d_in2 = Conv(input_dim//2, input_dim//2, 1)       
                
        self.conv = Conv(input_dim, input_dim, 3)
        self.fc1 = nn.Conv2d(inc[1], input_dim//2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(inc[0], input_dim//2, kernel_size=1, bias=False)
        
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        H_feature, L_feature = x

        L_feature = self.fc1(L_feature)
        H_feature = self.fc2(H_feature)
        
        g_L_feature =  self.Sigmoid(L_feature)
        g_H_feature = self.Sigmoid(H_feature)
        
        L_feature = self.d_in1(L_feature)
        H_feature = self.d_in2(H_feature)

        L_feature = L_feature + L_feature * g_L_feature + (1 - g_L_feature) * Upsample(g_H_feature * H_feature, size= L_feature.size()[2:], align_corners=False)
        H_feature = H_feature + H_feature * g_H_feature + (1 - g_H_feature) * Upsample(g_L_feature * L_feature, size= H_feature.size()[2:], align_corners=False) 
        
        H_feature = Upsample(H_feature, size = L_feature.size()[2:])
        out = self.conv(torch.cat([H_feature, L_feature], dim=1))
        return out

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- SBA ----
    try:
        module = SBA(in_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'SBA  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'SBA  自测跳过: {e}' + RESET)

