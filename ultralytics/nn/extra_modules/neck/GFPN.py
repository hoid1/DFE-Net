'''
本文件由BiliBili：魔傀面具整理
ultralytics/nn/module_images/GFPN.png   
ultralytics/nn/module_images/GFPN.md
论文链接：https://arxiv.org/abs/2211.15444
'''

import torch     
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, RepConv   
  
__all__ = ("CSPStage",)    
     
 
class BasicBlock_3x3_Reverse(nn.Module):   
    """DAMO-YOLO GFPN basic block."""
   
    def __init__(self, ch_in, ch_hidden_ratio, ch_out, shortcut=True):     
        super().__init__()
        assert ch_in == ch_out 
        ch_hidden = int(ch_in * ch_hidden_ratio)
        self.conv1 = Conv(ch_hidden, ch_out, 3, 1)
        self.conv2 = RepConv(ch_in, ch_hidden, 3, 1)
        self.shortcut = shortcut     

    def forward(self, x):
        y = self.conv2(x)    
        y = self.conv1(y)
        return x + y if self.shortcut else y
   

class SPP(nn.Module):  
    """Local GFPN SPP wrapper matching the source neck block contract."""

    def __init__(self, ch_in, ch_out, k, pool_size):
        super().__init__()    
        self.pool = nn.ModuleList(    
            [nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2, ceil_mode=False) for size in pool_size]    
        )
        self.conv = Conv(ch_in, ch_out, k)

    def forward(self, x):
        return self.conv(torch.cat([x, *[pool(x) for pool in self.pool]], dim=1))
 

class CSPStage(nn.Module):
    """DAMO-YOLO GFPN CSP stage."""     

    def __init__(  
        self,  
        ch_in,     
        ch_out,
        n=1,  
        block_fn="BasicBlock_3x3_Reverse",  
        ch_hidden_ratio=1.0, 
        act="silu",
        spp=False,   
    ):
        super().__init__()  
        split_ratio = 2     
        ch_first = int(ch_out // split_ratio)   
        ch_mid = int(ch_out - ch_first)   
        self.conv1 = Conv(ch_in, ch_first, 1)
        self.conv2 = Conv(ch_in, ch_mid, 1)
        self.convs = nn.Sequential()
        num_mid_outputs = n + int(bool(spp))    

        next_ch_in = ch_mid     
        for i in range(n):     
            if block_fn != "BasicBlock_3x3_Reverse":
                raise NotImplementedError(f"Unsupported GFPN block_fn: {block_fn}")     
            self.convs.add_module(
                str(i), BasicBlock_3x3_Reverse(next_ch_in, ch_hidden_ratio, ch_mid, shortcut=True)    
            )
            if i == (n - 1) // 2 and spp:
                self.convs.add_module("spp", SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13]))
            next_ch_in = ch_mid

        self.conv3 = Conv(ch_mid * num_mid_outputs + ch_first, ch_out, 1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)

        mid_out = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mid_out.append(y2) 
   
        return self.conv3(torch.cat(mid_out, dim=1))     
