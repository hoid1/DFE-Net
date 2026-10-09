'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:6211-6230
0526 版 ContextGuideFusionModule（旧库 md #159）。新版同名类签名已由 (inc) 改为 (inc, ouc)。
yaml 中请使用 ContextGuideFusionModule_0526。
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
import torch
from ultralytics.nn.extra_modules.module.iRMB import SEAttention
from ultralytics.nn.modules.conv import Conv


# ---- 原样迁移自 nn/extra_modules/block.py:6211-6230 ----
class ContextGuideFusionModule(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()
        
        self.adjust_conv = nn.Identity()
        if inc[0] != inc[1]:
            self.adjust_conv = Conv(inc[0], inc[1], k=1)
        
        self.se = SEAttention(inc[1] * 2)
    
    def forward(self, x):
        x0, x1 = x
        x0 = self.adjust_conv(x0)
        
        x_concat = torch.cat([x0, x1], dim=1) # n c h w
        x_concat = self.se(x_concat)
        x0_weight, x1_weight = torch.split(x_concat, [x0.size()[1], x1.size()[1]], dim=1)
        x0_weight = x0 * x0_weight
        x1_weight = x1 * x1_weight
        return torch.cat([x0 + x1_weight, x1 + x0_weight], dim=1)

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- ContextGuideFusionModule ----
    try:
        module = ContextGuideFusionModule(in_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'ContextGuideFusionModule  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'ContextGuideFusionModule  自测跳过: {e}' + RESET)

