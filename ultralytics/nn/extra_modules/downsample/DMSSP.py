'''     
本文件由BiliBili：魔傀面具整理     
ultralytics/nn/module_images/TGRS2026-DMSSP.png 
ultralytics/nn/module_images/TGRS2026-DMSSP.md
''' 

import os, sys  
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')  

import warnings
warnings.filterwarnings('ignore') 
from calflops import calculate_flops 

import torch
import torch.nn as nn    
import torch.nn.functional as F
   
from ultralytics.nn.modules.conv import Conv

class SpatialAttention(nn.Module):
    def __init__(self): 
        super().__init__()
    
    def forward(self, x):
        x_ = x.clone()  
 
        mean_values = torch.mean(x_, dim=(2, 3), keepdim=True)     

        max_values, _ = torch.max(x_, dim=2, keepdim=True)     
        max_values, _ = torch.max(max_values, dim=3, keepdim=True)    

        min_values, _ = torch.min(x_, dim=2, keepdim=True)
        min_values, _ = torch.min(min_values, dim=3, keepdim=True)
 
        var = torch.var(x_, dim=(-2, -1), keepdim=True)    
        u_ij = (x_ - mean_values) ** 2  
        q_ij = (u_ij / (max_values - min_values + 1e-8)) * var
        q = torch.sigmoid(q_ij)

        return q * x  
    
 
def convrelubn(in_channel, out_channel, kernel_size, dirate):  
    return nn.Sequential(
        nn.Conv2d( 
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,     
            stride=1, 
            padding=dirate, 
            dilation=dirate,
        ),  
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),     
    )
  

class DMSSP(nn.Module): 
    def __init__(self, in_channels, out_channels, channel_split=(1, 1, 2)):   
        super().__init__()

        assert in_channels % sum(channel_split) == 0

        self.split_ratio = [i / sum(channel_split) for i in channel_split]     
        self.embed_dims_1 = int(self.split_ratio[1] * in_channels)  
        self.embed_dims_2 = int(self.split_ratio[2] * in_channels)
        self.embed_dims_0 = in_channels - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = in_channels    
  
        self.SpatialAtt = SpatialAttention()
        self.belt = nn.Parameter(torch.zeros((1, in_channels, 1, 1)))     

        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channels, in_channels, 1, 1)     

        self.atrous_block6 = Conv(
            self.embed_dims_2,
            self.embed_dims_2,
            k=3,
            d=6,
        )    
        self.atrous_block12 = Conv(
            self.embed_dims_1,
            self.embed_dims_1,   
            k=3,
            d=12, 
        )
        self.atrous_block18 = Conv(
            self.embed_dims_0, 
            self.embed_dims_0,
            k=3,
            d=18,     
        )    
     
        self.PW_conv = nn.Conv2d(
            in_channels=in_channels,  
            out_channels=in_channels,
            kernel_size=1,
        )
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels, 
            kernel_size=3,
            stride=2,
            padding=1,
        )
    
    def forward(self, x):   
        size = x.shape[2:]
    
        image_features = self.mean(x)    
        image_features = self.conv(image_features)  
        image_features = F.interpolate(image_features, size=size, mode="bilinear") 

        image_features2 = self.SpatialAtt(x)
        x_s = image_features + self.belt * image_features2

        x_6 = self.atrous_block6(x[:, : self.embed_dims_2, ...])
        x_12 = self.atrous_block12(     
            x[:, self.embed_dims_2 : self.embed_dims_2 + self.embed_dims_1, ...]
        ) 
        x_18 = self.atrous_block18(x[:, self.embed_dims - self.embed_dims_0 :, ...])     
     
        x = self.PW_conv(torch.cat([x_6, x_12, x_18], dim=1))     
        x = x + x_s     

        return self.proj(x)    
  
   
if __name__ == '__main__':     
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for DMSSP half precision validation.')     
    device = torch.device('cuda')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width), device=device).half()

    module = DMSSP(in_channel, out_channel).to(device).half()
 
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    module = module.float()    

    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module, 
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4,     
                                     print_detailed=True)     
    print(RESET)
