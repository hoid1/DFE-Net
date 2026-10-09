"""    
本文件由BiliBili：魔傀面具整理  
ultralytics/nn/module_images/CSP-MSCB.png     
ultralytics/nn/module_images/EMBSFPN.md
"""  

import math    

import torch
import torch.nn as nn   

from ultralytics.nn.modules.block import C2f   
from ultralytics.nn.modules.conv import Conv  


class MSDC(nn.Module):    
    """Multi-scale depthwise conv stage used by MSCB."""    

    def __init__(self, in_channels, kernel_sizes, stride=1, dw_parallel=True):   
        super().__init__()
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList( 
            [nn.Sequential(Conv(in_channels, in_channels, k, stride, g=in_channels)) for k in kernel_sizes]
        ) 

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:  
            dw_out = dwconv(x)
            outputs.append(dw_out)     
            if not self.dw_parallel:
                x = x + dw_out 
        return outputs 
  

class MSCB(nn.Module):
    """Multi-scale convolution block used by CSP_MSCB."""  
     
    def __init__(self, in_channels, out_channels, kernel_sizes=None, stride=1, expansion_factor=2, dw_parallel=True, add=True):
        super().__init__()   
        kernel_sizes = [1, 3, 5] if kernel_sizes is None else kernel_sizes
        assert stride in {1, 2}
   
        self.use_skip_connection = stride == 1
        ex_channels = int(in_channels * expansion_factor)   
        self.pconv1 = Conv(in_channels, ex_channels, 1)
        self.msdc = MSDC(ex_channels, kernel_sizes, stride=stride, dw_parallel=dw_parallel)
        combined_channels = ex_channels if add else ex_channels * len(kernel_sizes)    
        self.pconv2 = Conv(combined_channels, out_channels, 1, act=False)
        self.add = add     
        self.combined_channels = combined_channels     
        self.out_channels = out_channels  
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False) if self.use_skip_connection and in_channels != out_channels else None    

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)   
   
        if self.add:
            dout = 0    
            for dw_out in msdc_outs:
                dout = dout + dw_out
        else:   
            dout = torch.cat(msdc_outs, dim=1)  

        dout = self.channel_shuffle(dout, math.gcd(self.combined_channels, self.out_channels)) 
        out = self.pconv2(dout)
  
        if self.use_skip_connection:     
            residual = x if self.conv1x1 is None else self.conv1x1(x)
            return residual + out
        return out 

    @staticmethod
    def channel_shuffle(x, groups):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)  
        x = x.transpose(1, 2).contiguous()  
        return x.view(batch_size, -1, height, width)  

  
class CSP_MSCB(C2f):     
    """C2f-style wrapper with MSCB inner blocks for EMBSFPN node stages."""    
  
    def __init__(self, c1, c2, n=1, kernel_sizes=None, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)     
        kernel_sizes = [1, 3, 5] if kernel_sizes is None else kernel_sizes   
        self.m = nn.ModuleList(MSCB(self.c, self.c, kernel_sizes=kernel_sizes) for _ in range(n))  


__all__ = ("CSP_MSCB",)  
