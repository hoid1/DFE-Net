'''
本文件由BiliBili：魔傀面具整理     
ultralytics/nn/module_images/自研模块-DGCM.png     
ultralytics/nn/module_images/自研模块-DGCM.md 
'''   
   
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings   
warnings.filterwarnings('ignore')  
from calflops import calculate_flops
  
import torch
import torch.nn as nn   
 
from ultralytics.nn.modules.conv import Conv

# ============ 双池化通道注意力 ============
class DualPoolChannelAttention(nn.Module):
    """Avg + Max 双池化通道注意力，统计描述比单一平均池化更丰富。"""
    def __init__(self, dim, reduction=8):
        super().__init__() 
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)     
        hidden = max(dim // reduction, 4)  
        self.fc = nn.Sequential( 
            nn.Conv2d(dim, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=True)
        )
        self.sigmoid = nn.Sigmoid()
  
    def forward(self, x):
        return self.sigmoid(self.fc(self.gap(x)) + self.fc(self.gmp(x)))

# ============ 多尺度条带空间注意力 ============
class StripSpatialAttention(nn.Module):
    """以水平/垂直条带卷积捕获各向异性长程上下文的空间注意力。"""
    def __init__(self, kernel=7):     
        super().__init__()  
        pad = kernel // 2
        # 输入为 [avg, max] 两通道描述子
        self.conv_h = nn.Conv2d(2, 1, (1, kernel), padding=(0, pad), bias=False)
        self.conv_v = nn.Conv2d(2, 1, (kernel, 1), padding=(pad, 0), bias=False)
        self.conv_sq = nn.Conv2d(2, 1, 3, padding=1, bias=False)  
        self.sigmoid = nn.Sigmoid()   
  
    def forward(self, x):     
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        desc = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv_h(desc) + self.conv_v(desc) + self.conv_sq(desc)  
        return self.sigmoid(attn)

# ============ 核心模块：DGCM ============
class DGCM(nn.Module):
    """
    Dual-Pooling Gated Complementary Module (DGCM)   
    纯学习、轻量路线，在 FCM 双流互补结构上：
      - 用 Avg+Max 双池化通道注意力替换单一平均池化；     
      - 用多尺度条带空间注意力替换单 1×1 空间注意力；  
      - 用逐像素门控凸组合替换加性融合。    
    """
    def __init__(self, dim, dim_out): 
        super().__init__()
        self.one = dim // 4  
        self.two = dim - dim // 4     
        self.conv1 = Conv(self.one, self.one, 3, 1, 1)    
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)
        self.conv2 = Conv(self.two, dim, 1, 1)

        self.spatial = StripSpatialAttention()            # 由外观流生成，调制语义流
        self.channel = DualPoolChannelAttention(dim)      # 由语义流生成，调制外观流
     
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=True),
            nn.Sigmoid()     
        ) 
        self.conv3 = Conv(dim, dim_out, 1, 1)    
 
    def forward(self, x):   
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv123(self.conv12(self.conv1(x1)))    
        x4 = self.conv2(x2)    
    
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4

        g = self.gate(torch.cat([x33, x44], dim=1))   
        x5 = g * x33 + (1 - g) * x44
        return self.conv3(x5) 

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32    
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
    
    module = DGCM(in_channel, out_channel).to(device)  
   
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    

    if not torch.cuda.is_available():    
        raise RuntimeError('CUDA is required for half validation.') 
    half_inputs = torch.randn((batch_size, in_channel, height, width), device='cuda').half()     
    half_module = DGCM(in_channel, out_channel).cuda().half()     
    half_outputs = half_module(half_inputs)
    print(GREEN + f'half inputs.size:{half_inputs.size()} outputs.size:{half_outputs.size()} dtype:{half_outputs.dtype}' + RESET)    
   
    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module,   
                                     input_shape=(batch_size, in_channel, height, width),     
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)   
    print(RESET)
