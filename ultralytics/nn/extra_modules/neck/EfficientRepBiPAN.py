''' 
本文件由BiliBili：魔傀面具整理
ultralytics/nn/module_images/EfficientRepBiPAN.png 
ultralytics/nn/module_images/EfficientRepBiPAN.md
论文链接：https://arxiv.org/pdf/2301.05586
'''

import torch   
import torch.nn as nn

   
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
    """Rep-style helper block with Conv2d + BatchNorm2d."""
   
    result = nn.Sequential() 
    result.add_module(  
        "conv",
        nn.Conv2d(    
            in_channels=in_channels,  
            out_channels=out_channels,     
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,     
            groups=groups, 
            bias=bias,   
        ), 
    )  
    result.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    """Minimal RepVGG block used by EfficientRepBiPAN neck blocks.""" 
 
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):     
        super().__init__()
        assert kernel_size == 3    
        assert padding == 1
     
        padding_11 = padding - kernel_size // 2
        self.nonlinearity = nn.ReLU() 
        self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        self.rbr_dense = conv_bn(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,     
            groups=groups,
        )     
        self.rbr_1x1 = conv_bn(   
            in_channels=in_channels,    
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding_11,  
            groups=groups,   
        )    
     
    def forward(self, x):  
        identity = 0 if self.rbr_identity is None else self.rbr_identity(x)
        return self.nonlinearity(self.rbr_dense(x) + self.rbr_1x1(x) + identity)
 
   
class Transpose(nn.Module): 
    """Transpose upsampling block used by BiFusion."""

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):     
        super().__init__()     
        self.upsample_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,   
            out_channels=out_channels,
            kernel_size=kernel_size,  
            stride=stride,    
            bias=True,
        )   
    
    def forward(self, x):
        return self.upsample_transpose(x)   

    
class BiFusion(nn.Module):  
    """Three-branch EfficientRepBiPAN fusion block."""

    def __init__(self, in_channels, out_channels):  
        super().__init__()  
        from ultralytics.nn.modules.conv import Conv   

        self.cv1 = Conv(in_channels[1], out_channels, 1, 1)
        self.cv2 = Conv(in_channels[2], out_channels, 1, 1)
        self.cv3 = Conv(out_channels * 3, out_channels, 1, 1)     
        self.upsample = Transpose(in_channels=out_channels, out_channels=out_channels)  
        self.downsample = Conv(out_channels, out_channels, 3, 2)
 
    def forward(self, x):
        x0 = self.upsample(x[0])     
        x1 = self.cv1(x[1]) 
        x2 = self.downsample(self.cv2(x[2]))    
        return self.cv3(torch.cat((x0, x1, x2), dim=1))


class BottleRep(nn.Module):   
    """Two-stage bottle rep block with optional residual scaling."""    
   
    def __init__(self, in_channels, out_channels, basic_block=RepVGGBlock, weight=False):
        super().__init__()     
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        self.shortcut = in_channels == out_channels     
        self.alpha = nn.Parameter(torch.ones(1)) if weight else 1.0   
     
    def forward(self, x):
        outputs = self.conv1(x)  
        outputs = self.conv2(outputs)  
        return outputs + self.alpha * x if self.shortcut else outputs
     

class RepBlock(nn.Module):     
    """Stage block composed of RepVGG-style units."""

    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, basic_block=RepVGGBlock):    
        super().__init__()
        self.conv1 = block(in_channels, out_channels)     
        self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None
     
        if block is BottleRep:  
            self.conv1 = BottleRep(in_channels, out_channels, basic_block=basic_block, weight=True)
            n = n // 2
            self.block = (    
                nn.Sequential(  
                    *(BottleRep(out_channels, out_channels, basic_block=basic_block, weight=True) for _ in range(n - 1)) 
                )   
                if n > 1
                else None
            )  
    
    def forward(self, x):
        x = self.conv1(x)    
        if self.block is not None: 
            x = self.block(x)     
        return x
    
  
__all__ = ("BiFusion", "RepBlock")
