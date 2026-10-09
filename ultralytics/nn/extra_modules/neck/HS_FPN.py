'''
本文件由BiliBili：魔傀面具整理  
ultralytics/nn/module_images/AAAI2025-HS-FPN.png  
ultralytics/nn/module_images/AAAI2025-HS-FPN.md     
论文链接：https://arxiv.org/pdf/2412.10116
'''  

import numpy as np

import torch  
import torch.nn as nn 
import torch.nn.functional as F
   
try:
    import torch_dct as DCT
except Exception:
    DCT = None    

from einops import rearrange
   
from ultralytics.nn.modules import Conv

class DctSpatialInteraction(nn.Module): 
    def __init__(self, in_channels, ratio, isdct=True):
        super().__init__()   
        self.ratio = ratio  
        self.use_dct = bool(isdct and DCT is not None)
        if not self.use_dct: 
            self.spatial1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)    

    def forward(self, x):   
        _, _, h0, w0 = x.size()
        if not self.use_dct:
            return x * torch.sigmoid(self.spatial1x1(x))    

        idct = DCT.dct_2d(x, norm="ortho")   
        weight = self._compute_weight(h0, w0, self.ratio).to(device=x.device, dtype=x.dtype)    
        weight = weight.view(1, h0, w0).expand_as(idct)    
        dct = idct * weight
        dct_ = DCT.idct_2d(dct, norm="ortho")    
        return x * dct_ 
 
    @staticmethod
    def _compute_weight(h, w, ratio):     
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])    
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0     
        return weight
   

class DctChannelInteraction(nn.Module):    
    def __init__(self, in_channels, patch, ratio, isdct=True):     
        super().__init__()    
        self.h = patch[0]    
        self.w = patch[1]
        self.ratio = ratio
        self.use_dct = bool(isdct and DCT is not None)  
        self.channel1x1 = nn.Conv2d(in_channels, in_channels, 1, groups=32)  
        self.channel2x1 = nn.Conv2d(in_channels, in_channels, 1, groups=32)    
        self.relu = nn.ReLU()
    
    def forward(self, x):   
        n, c, h, w = x.size()
        if not self.use_dct:     
            amaxp = F.adaptive_max_pool2d(x, output_size=(1, 1))
            aavgp = F.adaptive_avg_pool2d(x, output_size=(1, 1))     
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp))
            return x * torch.sigmoid(self.channel2x1(channel))  

        idct = DCT.dct_2d(x, norm="ortho") 
        weight = self._compute_weight(h, w, self.ratio).to(device=x.device, dtype=x.dtype)   
        weight = weight.view(1, h, w).expand_as(idct)
        dct = idct * weight
        dct_ = DCT.idct_2d(dct, norm="ortho")
  
        amaxp = F.adaptive_max_pool2d(dct_, output_size=(self.h, self.w))    
        aavgp = F.adaptive_avg_pool2d(dct_, output_size=(self.h, self.w))  
        amaxp = torch.sum(self.relu(amaxp), dim=[2, 3]).view(n, c, 1, 1)  
        aavgp = torch.sum(self.relu(aavgp), dim=[2, 3]).view(n, c, 1, 1)

        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)
        return x * torch.sigmoid(self.channel2x1(channel))     
 
    @staticmethod 
    def _compute_weight(h, w, ratio):
        h0 = int(h * ratio[0]) 
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)     
        weight[:h0, :w0] = 0
        return weight

  
class HFP(nn.Module): 
    """High-frequency perception module for HS-FPN."""    
  
    def __init__(self, in_channels, ratio=(0.25, 0.25), patch=(8, 8), isdct=True):    
        super().__init__()
        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct=isdct)  
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct=isdct)  
        self.out = nn.Sequential( 
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False), 
            nn.GroupNorm(32, in_channels), 
        )

    def forward(self, x):  
        spatial = self.spatial(x)  
        channel = self.channel(x)
        return self.out(spatial + channel)
  

class SDP(nn.Module):  
    """Spatial dependency perception module for HS-FPN."""

    def __init__(self, in_dim, dim=256, patch_size=None, inter_dim=None): 
        super().__init__()   
        self.conv1x1_0 = Conv(in_dim[0], dim) if in_dim[0] != dim else nn.Identity()  
        self.conv1x1_1 = Conv(in_dim[1], dim) if in_dim[1] != dim else nn.Identity()     
 
        self.inter_dim = dim if inter_dim is None else inter_dim
        self.conv_q = nn.Sequential( 
            nn.Conv2d(dim, self.inter_dim, 1, padding=0, bias=False),
            nn.GroupNorm(32, self.inter_dim), 
        )    
        self.conv_k = nn.Sequential(  
            nn.Conv2d(dim, self.inter_dim, 1, padding=0, bias=False),
            nn.GroupNorm(32, self.inter_dim),
        )   
        self.softmax = nn.Softmax(dim=-1)  
        self.patch_size = patch_size
    
    def forward(self, x):  
        x_low, x_high = x 
        x_low = self.conv1x1_0(x_low) 
        x_high = self.conv1x1_1(x_high)
        _, _, h_, w_ = x_low.size()     
        p1, p2 = self._fit_patch(h_, w_)    
   
        q = rearrange(    
            self.conv_q(x_low),    
            "b c (h p1) (w p2) -> (b h w) c (p1 p2)", 
            p1=p1,   
            p2=p2,
        ).transpose(1, 2) 
        k = rearrange(
            self.conv_k(x_high),
            "b c (h p1) (w p2) -> (b h w) c (p1 p2)", 
            p1=p1,
            p2=p2,    
        )
     
        attn = torch.matmul(q, k) / np.power(self.inter_dim, 0.5)    
        attn = self.softmax(attn) 
        v = k.transpose(1, 2)
        output = torch.matmul(attn, v) 
        output = rearrange(   
            output.transpose(1, 2).contiguous(),     
            "(b h w) c (p1 p2) -> b c (h p1) (w p2)",
            p1=p1,
            p2=p2,   
            h=h_ // p1,
            w=w_ // p2,
        )  
        return output + x_low

    def _fit_patch(self, h, w):   
        p1 = min(self.patch_size[0], h)    
        p2 = min(self.patch_size[1], w)  
        while p1 > 1 and h % p1 != 0:  
            p1 -= 1    
        while p2 > 1 and w % p2 != 0:  
            p2 -= 1   
        return p1, p2    
    
__all__ = ("HFP", "SDP")
