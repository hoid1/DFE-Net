''' 
本文件由BiliBili：魔傀面具整理
ultralytics/nn/module_images/IF2023-SDFM.png
ultralytics/nn/module_images/IF2023-SDFM.md  
论文链接：https://www.sciencedirect.com/science/article/abs/pii/S1566253523001860
'''     

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')   

import warnings     
warnings.filterwarnings('ignore')
from calflops import calculate_flops
    
import torch   
import torch.nn as nn

from ultralytics.nn.modules import Conv    

class SDFM(nn.Module): 
    """Superficial detail fusion module."""
  
    def __init__(self, in_channel, out_channels=64, r=4):     
        super().__init__()
        inter_channels = int(out_channels // r)    

        self.recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    
            Conv(2 * out_channels, 2 * inter_channels),
            Conv(2 * inter_channels, 2 * out_channels, act=nn.Sigmoid()),
        )     
        self.channel_agg = Conv(2 * out_channels, out_channels)
        self.local_att = nn.Sequential(     
            Conv(out_channels, inter_channels, 1),
            Conv(inter_channels, out_channels, 1, act=False), 
        )
        self.global_att = nn.Sequential(  
            nn.AdaptiveAvgPool2d(1),    
            Conv(out_channels, inter_channels, 1),
            Conv(inter_channels, out_channels, 1),
        )     
        self.sigmoid = nn.Sigmoid()
     
        self.conv_adjust = nn.ModuleList([])    
        for i in in_channel:
            if i != out_channels: 
                self.conv_adjust.append(Conv(i, out_channels, 1))    
            else:
                self.conv_adjust.append(nn.Identity())
     
    def forward(self, data):
        x1, x2 = data  
        x1 = self.conv_adjust[0](x1)
        x2 = self.conv_adjust[1](x2)
        _, c, _, _ = x1.shape
        fused = torch.cat([x1, x2], dim=1)  
        recal = self.recalibrate(fused) 
        recal = recal * fused + fused    
        x1, x2 = torch.split(recal, c, dim=1)
        agg = self.channel_agg(recal)
        local_w = self.local_att(agg)    
        global_w = self.global_att(agg)
        w = self.sigmoid(local_w * global_w)     
        return w * x1 + (1 - w) * x2
 
if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32
    ouc_channel = 32
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device)  
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device) 

    module = SDFM([channel_1, channel_2], ouc_channel).to(device)    
    module.eval()  

    outputs = module([inputs_1, inputs_2])
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)    
 
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,     
                                     args=[[inputs_1, inputs_2]],
                                     output_as_string=True,    
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)  
