'''
本文件由BiliBili：魔傀面具整理
ultralytics/nn/module_images/TIP2024-CGAFusion.png  
ultralytics/nn/module_images/TIP2024-CGAFusion.md  
论文链接：https://arxiv.org/pdf/2301.04805
'''    

import os, sys     
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')     
 
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops     

import torch
import torch.nn as nn  

from ultralytics.nn.modules.conv import Conv  

class SpatialAttention_CGA(nn.Module):    
    def __init__(self):
        super().__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode="reflect", bias=True)
   
    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)     
        x2 = torch.cat([x_avg, x_max], dim=1)
        return self.sa(x2)


class ChannelAttention_CGA(nn.Module):    
    def __init__(self, dim, reduction=8):
        super().__init__()
        hidden = max(dim // reduction, 1) 
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, padding=0, bias=True),     
        )  

    def forward(self, x):
        return self.ca(self.gap(x)) 

    
class PixelAttention_CGA(nn.Module): 
    def __init__(self, dim):  
        super().__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode="reflect", groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()   
     
    def forward(self, x, pattn1):
        b, c, h, w = x.shape  
        x_pair = torch.cat([x.unsqueeze(2), pattn1.unsqueeze(2)], dim=2).reshape(b, 2 * c, h, w) 
        return self.sigmoid(self.pa2(x_pair))
 

class CGAFusion(nn.Module): 
    def __init__(self, in_dim, out_dim, reduction=8):  
        super().__init__()
        self.sa = SpatialAttention_CGA()
        self.ca = ChannelAttention_CGA(out_dim, reduction)   
        self.pa = PixelAttention_CGA(out_dim)  
        self.conv = nn.Conv2d(out_dim, out_dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

        self.conv_adjust = nn.ModuleList([])
        for i in in_dim:    
            if i != out_dim:    
                self.conv_adjust.append(Conv(i, out_dim, 1)) 
            else:     
                self.conv_adjust.append(nn.Identity())  
  
    def forward(self, data):
        x, y = data
        x = self.conv_adjust[0](x) 
        y = self.conv_adjust[1](y)     
        initial = x + y    
        pattn1 = self.sa(initial) + self.ca(initial)
        pattn2 = self.sigmoid(self.pa(initial, pattn1))  
        out = initial + pattn2 * x + (1 - pattn2) * y    
        return self.conv(out)     
     
if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32 
    ouc_channel = 32 
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device)    
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device)    
     
    module = CGAFusion([channel_1, channel_2], ouc_channel).to(device)
 
    outputs = module([inputs_1, inputs_2])   
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)
 
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     args=[[inputs_1, inputs_2]],
                                     output_as_string=True,   
                                     output_precision=4,
                                     print_detailed=True) 
    print(RESET)