'''
本文件由BiliBili：魔傀面具整理
ultralytics/nn/module_images/自研模块-BCIM.png    
ultralytics/nn/module_images/自研模块-BCIM.md    
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
    """Sobel 算子，提取边界结构信息。"""   
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
    b = x.shape[0]  
    x_flat = x.reshape(b, -1) 
    x_min = x_flat.min(dim=1, keepdim=True)[0].reshape(b, 1, 1, 1)
    x_max = x_flat.max(dim=1, keepdim=True)[0].reshape(b, 1, 1, 1)  
    return (x - x_min) / (x_max - x_min + 1e-6)
    
class EdgeSpatialAttention(nn.Module):  
    """边缘引导空间注意力。"""     
    def __init__(self):  
        super().__init__() 
        self.sobel = Sobelxy()
        self.conv = nn.Conv2d(1, 1, 7, padding=3, padding_mode='reflect', bias=True)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        x_gray = torch.mean(x, dim=1, keepdim=True)   
        x_edge = _minmax_norm(self.sobel(x_gray)) 
        return self.sigmoid(self.conv(x_edge))
  
class FreqChannelAttention(nn.Module): 
    """高频感知的双池化通道注意力。"""   
    def __init__(self, dim, reduction=8):
        super().__init__()  
        self.laplacian = Laplacian()    
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
        B, C, H, W = x.size()   
        weight = self.laplacian.weight.repeat(C, 1, 1, 1).to(x.dtype)   
        hf = torch.abs(F.conv2d(x, weight, padding=1, groups=C)) 
        attn = self.fc(self.gap(hf)) + self.fc(self.gmp(hf)) 
        return self.sigmoid(attn) 
  
# ============ 双向跨流互注意力单元 ============
class BidirectionalCrossUnit(nn.Module):
    """单轮双向互调制：x3 由边缘空间注意力调制，x4 由高频通道注意力调制。"""
    def __init__(self, dim): 
        super().__init__()   
        self.edge_spatial = EdgeSpatialAttention()       # 由 x4 生成，调制 x3
        self.freq_channel = FreqChannelAttention(dim)     # 由 x3 生成，调制 x4

    def forward(self, x3, x4):    
        x3_new = x3 + self.edge_spatial(x4) * x3     
        x4_new = x4 + self.freq_channel(x3) * x4  
        return x3_new, x4_new  
     
# ============ 核心模块：BCIM ============ 
class BCIM(nn.Module):
    """
    Bidirectional Cross-Stream Iteration Module (BCIM)
    将 FCM 单向单次交叉引导升级为双向多轮迭代互精炼：
      - 把语义流 x3 与外观流 x4 视为"伪双模态"；
      - 每轮双向互注意力调制后，以逐像素门控聚合到下一轮；
      - 渐进式增强互补特征对齐。     
    """
    def __init__(self, dim, dim_out, iters=2):   
        super().__init__()  
        self.iters = iters    
        self.one = dim // 4
        self.two = dim - dim // 4   
        self.conv1 = Conv(self.one, self.one, 3, 1, 1)     
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)    
        self.conv123 = Conv(self.one, dim, 1, 1)
        self.conv2 = Conv(self.two, dim, 1, 1)
   
        self.units = nn.ModuleList([BidirectionalCrossUnit(dim) for _ in range(iters)]) 
        # 每轮的跨流门控聚合
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim * 2, dim, 1, bias=True), nn.Sigmoid())    
            for _ in range(iters)
        ])  
        self.conv3 = Conv(dim, dim_out, 1, 1)
    
    def forward(self, x):   
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv123(self.conv12(self.conv1(x1)))
        x4 = self.conv2(x2)
    
        fused = None
        for unit, gate in zip(self.units, self.gates):  
            x3, x4 = unit(x3, x4)
            g = gate(torch.cat([x3, x4], dim=1)) 
            fused = g * x3 + (1 - g) * x4   
        return self.conv3(fused)

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32  
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device) 
     
    module = BCIM(in_channel, out_channel).to(device)    

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    if not torch.cuda.is_available():    
        raise RuntimeError('CUDA is required for half validation.')     
    half_inputs = torch.randn((batch_size, in_channel, height, width), device='cuda').half()     
    half_module = BCIM(in_channel, out_channel).cuda().half()     
    half_outputs = half_module(half_inputs)     
    print(GREEN + f'half inputs.size:{half_inputs.size()} outputs.size:{half_outputs.size()} dtype:{half_outputs.dtype}' + RESET)

    print(ORANGE)    
    flops, macs, _ = calculate_flops(model=module,     
                                     input_shape=(batch_size, in_channel, height, width),    
                                     output_as_string=True,     
                                     output_precision=4,
                                     print_detailed=True) 
    print(RESET)
