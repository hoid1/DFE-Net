'''   
本文件由BiliBili：魔傀面具整理    
ultralytics/nn/module_images/自研模块-EFCM.png   
ultralytics/nn/module_images/自研模块-EFCM.md     
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

# ============ 辅助算子：物理先验（免参数）============ 
class Sobelxy(nn.Module):    
    """Sobel 算子，提取目标边界结构信息。"""
    def __init__(self):
        super().__init__() 
        kernelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)   
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False) 
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)
   
    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)   
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)   

class Laplacian(nn.Module):
    """拉普拉斯算子，提取高频纹理信息。"""    
    def __init__(self):
        super().__init__()
        kernel = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  
        self.weight = nn.Parameter(data=kernel, requires_grad=False) 

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=1)
 
def _minmax_norm(x):   
    """逐样本 min-max 归一化，half 安全。"""     
    b = x.shape[0] 
    x_flat = x.reshape(b, -1) 
    x_min = x_flat.min(dim=1, keepdim=True)[0].reshape(b, 1, 1, 1)
    x_max = x_flat.max(dim=1, keepdim=True)[0].reshape(b, 1, 1, 1)
    return (x - x_min) / (x_max - x_min + 1e-6)   
     
# ============ 边缘引导空间注意力（作用于语义流）============ 
class EdgeSpatialAttention(nn.Module):     
    """以 Sobel 边缘先验引导的空间注意力，锐化目标边界。"""  
    def __init__(self):
        super().__init__()  
        self.sobel = Sobelxy()
        self.conv = nn.Conv2d(1, 1, kernel_size=7, padding=3, padding_mode='reflect', bias=True)
        self.sigmoid = nn.Sigmoid()  
   
    def forward(self, x):
        x_gray = torch.mean(x, dim=1, keepdim=True)     
        x_edge = self.sobel(x_gray)    
        x_edge = _minmax_norm(x_edge)
        return self.sigmoid(self.conv(x_edge))

# ============ 高频引导空间注意力（作用于外观流）============    
class FreqSpatialAttention(nn.Module):
    """以 Laplacian 高频先验引导的空间注意力，强调纹理细节。"""
    def __init__(self):     
        super().__init__()
        self.laplacian = Laplacian()
        self.conv = nn.Conv2d(1, 1, kernel_size=7, padding=3, padding_mode='reflect', bias=True)    
        self.sigmoid = nn.Sigmoid()
     
    def forward(self, x):   
        x_gray = torch.mean(x, dim=1, keepdim=True)   
        x_hf = torch.abs(self.laplacian(x_gray))
        x_hf = _minmax_norm(x_hf)  
        return self.sigmoid(self.conv(x_hf)) 
    
# ============ 双池化通道注意力 ============     
class DualPoolChannelAttention(nn.Module):   
    """融合 Avg/Max 双池化的通道注意力，统计描述更丰富。"""  
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
        attn = self.fc(self.gap(x)) + self.fc(self.gmp(x))
        return self.sigmoid(attn)     

# ============ 核心模块：EFCM ============
class EFCM(nn.Module): 
    """     
    Edge-Frequency Complementary Module (EFCM)
    在 FCM 非对称双流互补结构上注入物理算子先验与逐像素门控融合：
      - 语义流 (x3)：由外观流生成的边缘空间注意力调制；
      - 外观流 (x4)：由语义流生成的双池化通道注意力调制 + 自身高频空间注意力增强；    
      - 融合：以可学习逐像素门控完成两流凸组合，取代僵硬的加性融合。     
    """
    def __init__(self, dim, dim_out):   
        super().__init__()
        self.one = dim // 4
        self.two = dim - dim // 4
        # 语义流：深层小通道精炼
        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)  
        # 外观流：浅层升维保留外观     
        self.conv2 = Conv(self.two, dim, 1, 1)
   
        # 物理先验注意力
        self.edge_spatial = EdgeSpatialAttention()       # 作用于语义流 x3  
        self.freq_spatial = FreqSpatialAttention()        # 作用于外观流 x4   
        self.channel = DualPoolChannelAttention(dim)      # 由语义流生成，调制外观流    
 
        # 逐像素门控融合 
        self.gate = nn.Sequential(    
            nn.Conv2d(dim * 2, dim, 1, bias=True), 
            nn.Sigmoid() 
        )  
        self.conv3 = Conv(dim, dim_out, 1, 1) 
  
    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)   
        # 语义流
        x3 = self.conv123(self.conv12(self.conv1(x1)))     
        # 外观流
        x4 = self.conv2(x2)  

        # 交叉互补 + 物理先验增强     
        x33 = self.edge_spatial(x4) * x3          # 外观→边缘空间注意力调制语义流
        x44 = self.channel(x3) * x4               # 语义→通道注意力调制外观流
        x44 = self.freq_spatial(x44) * x44        # 高频空间注意力增强外观流
 
        # 逐像素门控凸组合融合
        g = self.gate(torch.cat([x33, x44], dim=1))   
        x5 = g * x33 + (1 - g) * x44    
        return self.conv3(x5) 
 
if __name__ == '__main__':     
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m" 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)   

    module = EFCM(in_channel, out_channel).to(device) 
    
    outputs = module(inputs)  
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for half validation.') 
    half_inputs = torch.randn((batch_size, in_channel, height, width), device='cuda').half()
    half_module = EFCM(in_channel, out_channel).cuda().half()
    half_outputs = half_module(half_inputs)
    print(GREEN + f'half inputs.size:{half_inputs.size()} outputs.size:{half_outputs.size()} dtype:{half_outputs.dtype}' + RESET)     
   
    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,  
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True, 
                                     output_precision=4,    
                                     print_detailed=True)
    print(RESET)  
