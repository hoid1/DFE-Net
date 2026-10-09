'''     
本文件由BiliBili：魔傀面具整理     
ultralytics/nn/module_images/TCSVT2026-MAFusion.png
ultralytics/nn/module_images/TCSVT2026-MAFusion.md  
论文链接：https://ieeexplore.ieee.org/document/11218867   
''' 

import torch     
import torch.nn as nn
import torch.nn.functional as F
 
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops     
 
class Sobelxy(nn.Module):
    """Sobel 算子，用于边缘检测。"""
    def __init__(self):   
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],   
                   [-1, 0, 1]]  
        kernely = [[1, 2, 1],
                   [0, 0, 0], 
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)    
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)  
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False) 
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)
   
    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)  
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)    

class ChannelAttentionIR(nn.Module):  
    """针对红外图像的通道注意力模块，更关注全局信息。"""    
    def __init__(self, dim, reduction=8):     
        super(ChannelAttentionIR, self).__init__() 
        self.gmp = nn.AdaptiveMaxPool2d(1)  # 使用最大池化
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=True),    
            nn.ReLU(inplace=True),    
            nn.Linear(dim // reduction, dim, bias=True)
        )     
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):    
        b, c, _, _ = x.size()
        x_mp = self.gmp(x).view(b, c)     # [B, C]
        attn = self.fc(x_mp)              # [B, C]
        attn = self.sigmoid(attn).view(b, c, 1, 1)
        return attn   
     
class Laplacian(nn.Module):
    """拉普拉斯算子，用于边缘检测。"""  
    def __init__(self):
        super(Laplacian, self).__init__()   
        # 定义拉普拉斯卷积核    
        kernel = [[0, 1, 0],     
                  [1, -4, 1],   
                  [0, 1, 0]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # 升维以适应卷积操作
        self.weight = nn.Parameter(data=kernel, requires_grad=False)  # 不更新权重  
  
    def forward(self, x): 
        # x 的形状应为 [B, C, H, W]，但由于卷积核是单通道的，我们需要对每个通道进行卷积
        # 可以使用 group 参数进行分组卷积
        B, C, H, W = x.size()     
        weight = self.weight.repeat(C, 1, 1, 1)  # 扩展卷积核以匹配输入通道数  
        laplacian = F.conv2d(x, weight, padding=1, groups=C)   
        return laplacian
   
class SpatialAttentionIR(nn.Module):
    """针对红外图像的空间注意力模块，强调边缘和形状信息。"""
    def __init__(self):   
        super(SpatialAttentionIR, self).__init__()
        self.sobel = Sobelxy()  
        self.conv = nn.Conv2d(1, 1, kernel_size=7, padding=3, padding_mode='reflect', bias=True) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_gray = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        x_edge = self.sobel(x_gray)   
        x_edge = (x_edge - x_edge.min()) / (x_edge.max() - x_edge.min() + 1e-6)
        attn = self.conv(x_edge)
        #check_tensor(attn, "edge attn before Sigmoid")
        attn = self.sigmoid(attn)   
        return attn

class ChannelAttentionVIS(nn.Module):    
    """针对可见光图像的通道注意力模块，更关注细节信息。"""
    def __init__(self, dim, reduction=16):     
        super(ChannelAttentionVIS, self).__init__()     
        self.gap = nn.AdaptiveAvgPool2d(1)  # 使用平均池化 
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=True)
        )  
        self.sigmoid = nn.Sigmoid()  
   
    def forward(self, x):
        b, c, _, _ = x.size()  
        x_ap = self.gap(x).view(b, c)     # [B, C] 
        attn = self.fc(x_ap)              # [B, C]
        attn = self.sigmoid(attn).view(b, c, 1, 1) 
        return attn
     
class SpatialAttentionVIS(nn.Module):   
    """针对可见光图像的空间注意力模块，强调纹理和细节信息。"""
    def __init__(self):    
        super(SpatialAttentionVIS, self).__init__()
        self.laplacian = Laplacian()   
        self.conv = nn.Conv2d(1, 1, kernel_size=7, padding=3, padding_mode='reflect', bias=True)    
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): 
        # 将输入转换为灰度图 
        x_gray = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]     
        # 使用拉普拉斯算子提取高频信息
        x_laplacian = self.laplacian(x_gray) 
        # 归一化处理
        x_laplacian = torch.abs(x_laplacian)  # 取绝对值
        x_laplacian = (x_laplacian - x_laplacian.min()) / (x_laplacian.max() - x_laplacian.min() + 1e-6)  
        # 卷积和激活     
        attn = self.conv(x_laplacian)     
        #check_tensor(attn, "laplacian attn before Sigmoid")
        attn = self.sigmoid(attn)   
        return attn

class MAFusion(nn.Module):
    """模态感知融合模块，分别处理红外和可见光特征，再进行融合。"""
    def __init__(self, in_dims, out_dim, reduction_ir=8, reduction_vis=16):
        super(MAFusion, self).__init__()   
        # 红外注意力模块
        self.channel_attn_ir = ChannelAttentionIR(out_dim, reduction_ir)    
        self.spatial_attn_ir = SpatialAttentionIR()
        # 可见光注意力模块 
        self.channel_attn_vis = ChannelAttentionVIS(out_dim, reduction_vis)
        self.spatial_attn_vis = SpatialAttentionVIS()
        # 像素注意力模块 
        self.pa = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=1, bias=True),     
            nn.Sigmoid()
        )
        # 最终融合卷积    
        self.conv = nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=True)   

        self.conv_vis = nn.Conv2d(in_dims[0], out_dim, kernel_size=1, bias=True)
        self.conv_ir = nn.Conv2d(in_dims[1], out_dim, kernel_size=1, bias=True)

    def forward(self, inputs):   
        x_vis, x_ir = inputs    
        x_vis = self.conv_vis(x_vis)
        x_ir = self.conv_ir(x_ir)    
        # 红外特征的通道注意力
        attn_ir_channel = self.channel_attn_ir(x_ir)           # [B, C, 1, 1]   
        x_ir_channel = x_ir * attn_ir_channel                  # 通道加权 
        # 红外特征的空间注意力
        attn_ir_spatial = self.spatial_attn_ir(x_ir_channel)   # [B, 1, H, W]
        x_ir_attn = x_ir_channel * attn_ir_spatial             # 空间加权

        # 可见光特征的通道注意力
        attn_vis_channel = self.channel_attn_vis(x_vis)        # [B, C, 1, 1]   
        x_vis_channel = x_vis * attn_vis_channel               # 通道加权    
        # 可见光特征的空间注意力   
        attn_vis_spatial = self.spatial_attn_vis(x_vis_channel)  # [B, 1, H, W]  
        x_vis_attn = x_vis_channel * attn_vis_spatial          # 空间加权

        # 像素注意力融合     
        x_cat = torch.cat([x_ir_attn, x_vis_attn], dim=1)      # [B, 2C, H, W]   
        attn_pixel = self.pa(x_cat)                            # [B, C, H, W]
        result = attn_pixel * x_ir_attn + (1 - attn_pixel) * x_vis_attn  # 融合
   
        result = self.conv(result)                             # 1x1 卷积融合
        return result
  
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32   
    ouc_channel = 32
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device)
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device)

    module = MAFusion([channel_1, channel_2], ouc_channel).to(device)    

    outputs = module([inputs_1, inputs_2])    
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)
 
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,  
                                     args=[[inputs_1, inputs_2]],     
                                     output_as_string=True,  
                                     output_precision=4,
                                     print_detailed=True)    
    print(RESET)    
