'''
本文件由BiliBili：魔傀面具整理   
ultralytics/nn/module_images/SlimNeck.md     
论文链接：https://arxiv.org/pdf/2206.02424
'''     
  
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, DWConv     

__all__ = ("GSConv", "GSBottleneck", "GSBottleneckC", "VoVGSCSP")    
    

class GSConv(nn.Module):
    """GSConv from Slim-neck by GSConv."""
  
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__() 
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, act) 
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, act)  

    def forward(self, x):   
        x1 = self.cv1(x)  
        x2 = torch.cat((x1, self.cv2(x1)), 1) 
  
        b, n, h, w = x2.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)   
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        return torch.cat((y[0], y[1]), 1)


class GSBottleneck(nn.Module):
    """GS bottleneck used by SlimNeck CSP blocks."""   

    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()   
        c_ = int(c2 * e) 
        self.conv_lighting = nn.Sequential(GSConv(c1, c_, 1, 1), GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = Conv(c1, c2, 1, 1, act=False)
  
    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)  

    
class GSBottleneckC(GSBottleneck):
    """Cheap GS bottleneck variant with depthwise shortcut."""

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__(c1, c2, k, s)   
        self.shortcut = DWConv(c1, c2, k, s, act=False)    
 
 
class VoVGSCSP(nn.Module):  
    """VoV-GSCSP neck block composed of GS bottlenecks."""    
  
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()   
        c_ = int(c2 * e)     
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)    
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
        # Kept for source-state compatibility with the upstream SlimNeck implementation.
        self.res = Conv(c_, c_, 3, 1, act=False)
        self.cv3 = Conv(2 * c_, c2, 1) 

    def forward(self, x):  
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))
