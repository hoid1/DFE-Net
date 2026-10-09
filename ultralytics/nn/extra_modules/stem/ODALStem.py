'''     
本文件由BiliBili：魔傀面具整理
ultralytics/nn/module_images/自研模块-ODALStem.png    
ultralytics/nn/module_images/自研模块-ODALStem.md 
''' 

import warnings    
warnings.filterwarnings('ignore') 
from calflops import calculate_flops

import torch, math
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv   
from ultralytics.nn.extra_modules.stem.LoG import Gaussian, DRFD_LoG


class FixedLoGConv(nn.Module):    
    def __init__(self, dim, kernel_size, sigma):
        super().__init__()    
        kernel = self.log_kernel(kernel_size, sigma)    
        self.filter = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2), groups=dim, bias=False)    
        self.filter.weight.data = kernel.repeat(dim, 1, 1, 1)
        self.filter.weight.requires_grad_(False)
    
    def forward(self, x):
        return self.filter(x) 

    @staticmethod
    def log_kernel(kernel_size: int, sigma: float):
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij') 
        kernel = (xx**2 + yy**2 - 2 * sigma**2) / (2 * math.pi * sigma**4) * torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel - kernel.mean()   
        kernel = kernel / kernel.abs().sum().clamp_min(1e-6)  
        return kernel.unsqueeze(0).unsqueeze(0)

     
class DirectionalLoGFilter(nn.Module):
    def __init__(self, dim, kernel_size=5, sigma=0.8): 
        super().__init__()  
        self.filters = nn.ModuleList(
            [   
                nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2), groups=dim, bias=False)    
                for _ in range(4)  
            ] 
        ) 
        kernels = self.directional_kernels(kernel_size, sigma)
        for conv, kernel in zip(self.filters, kernels):   
            conv.weight.data = kernel.repeat(dim, 1, 1, 1)
            conv.weight.requires_grad_(False)  
    
    def forward(self, x): 
        return torch.stack([conv(x) for conv in self.filters], dim=1) 

    @staticmethod
    def directional_kernels(kernel_size: int, sigma: float):   
        if kernel_size < 3 or kernel_size % 2 == 0:  
            raise ValueError('kernel_size must be an odd integer greater than or equal to 3')   
  
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')  
        gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        center = kernel_size // 2    
        second = torch.zeros(kernel_size, dtype=torch.float32)    
        second[center - 1], second[center], second[center + 1] = 1.0, -2.0, 1.0
 
        horizontal = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
        horizontal[center, :] = second 
 
        vertical = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)   
        vertical[:, center] = second 

        diag_main = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
        diag_anti = torch.zeros((kernel_size, kernel_size), dtype=torch.float32) 
        for idx, value in enumerate(second):
            diag_main[idx, idx] = value 
            diag_anti[idx, kernel_size - 1 - idx] = value

        kernels = []
        for base in [horizontal, vertical, diag_main, diag_anti]:  
            kernel = base * gaussian   
            kernel = kernel - kernel.mean()   
            kernel = kernel / kernel.abs().sum().clamp_min(1e-6)
            kernels.append(kernel.unsqueeze(0).unsqueeze(0))   
        return kernels


class OrientationGate(nn.Module):    
    def __init__(self, dim, num_paths=4, reduction=4):    
        super().__init__() 
        hidden = max(dim // reduction, num_paths)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1, stride=1)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(hidden, num_paths, kernel_size=1, stride=1)    

    def forward(self, x):
        weights = self.fc2(self.act(self.fc1(self.pool(x)))).flatten(1)
        return torch.softmax(weights, dim=1)  
     

class ODALBlock(nn.Module):    
    def __init__(self, in_c, out_c, iso_kernel_size=7, iso_sigma=1.0, dir_kernel_size=5, dir_sigma=0.8):   
        super().__init__()  
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3)
        self.iso_filter = FixedLoGConv(out_c, iso_kernel_size, iso_sigma)     
        self.dir_filter = DirectionalLoGFilter(out_c, dir_kernel_size, dir_sigma)   
        self.orientation_gate = OrientationGate(out_c) 
        self.iso_norm = nn.BatchNorm2d(out_c)  
        self.dir_norm = nn.BatchNorm2d(out_c)
        self.out_norm = nn.BatchNorm2d(out_c)  
        self.act = nn.SiLU()  
        self.direction_scale = nn.Parameter(torch.tensor(0.0))   
     
    def forward(self, x):
        x = self.conv_init(x)
        iso = self.act(self.iso_norm(self.iso_filter(x)))    
        direction_weights = self.orientation_gate(x) 
        directional = self.dir_filter(x)     
        directional = torch.sum(directional * direction_weights[:, :, None, None, None], dim=1) 
        directional = self.act(self.dir_norm(directional))
        scale = torch.sigmoid(self.direction_scale)
        out = self.act(self.out_norm(x + iso + scale * directional))     
        return out, direction_weights 


class ODALStem(nn.Module):     
    
    def __init__(self, in_chans, stem_dim):   
        super().__init__()  
        out_c14 = int(stem_dim / 4)  
        out_c12 = int(stem_dim / 2)
        self.Conv_D = nn.Sequential(
            nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14),
            Conv(out_c12, out_c12, 3, 2, g=out_c12)    
        ) 
        self.odal = ODALBlock(in_chans, out_c14, 7, 1.0, 5, 0.8)   
        self.gaussian = Gaussian(out_c12, 9, 0.5)
        self.norm = nn.BatchNorm2d(out_c12)
        self.drfd = DRFD_LoG(out_c12)
        self.last_orientation_weights = None
     
    def forward(self, x):   
        x, direction_weights = self.odal(x)
        self.last_orientation_weights = direction_weights.detach()
        x = self.Conv_D(x)
        x = self.norm(x + self.gaussian(x))     
        x = self.drfd(x)
        return x


if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
   
    module = ODALStem(in_channel, out_channel).to(device)

    outputs = module(inputs)     
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET) 
   
    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),    
                                     output_as_string=True,
                                     output_precision=4,   
                                     print_detailed=True)
    print(RESET)  
