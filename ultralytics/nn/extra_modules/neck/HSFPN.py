'''  
本文件由BiliBili：魔傀面具整理    
ultralytics/nn/module_images/HSFPN.png    
ultralytics/nn/module_images/HSFPN.md
ultralytics/nn/module_images/HSPAN.md 
论文链接：https://arxiv.org/pdf/2401.00926    
'''     
    
import torch.nn as nn
 
class ChannelAttention_HSFPN(nn.Module):     
    """HSFPN channel attention used by the gated top-down neck variants."""

    def __init__(self, in_planes, ratio=4, flag=True):     
        super().__init__()    
        hidden_planes = max(in_planes // ratio, 1)  
        self.avg_pool = nn.AdaptiveAvgPool2d(1)   
        self.max_pool = nn.AdaptiveMaxPool2d(1)   
        self.conv1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)  
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_planes, in_planes, 1, bias=False)
        self.flag = flag
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight) 

    def forward(self, x):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)  
        return out * x if self.flag else out
 

class Multiply(nn.Module):  
    """Elementwise multiply helper for HSFPN gating branches."""     

    def forward(self, x):    
        return x[0] * x[1]

__all__ = ("ChannelAttention_HSFPN", "Multiply", "HFP", "SDP")  
