'''    
本文件由BiliBili：魔傀面具整理
ultralytics/nn/module_images/自研模块-MSIDWC.png
ultralytics/nn/module_images/自研模块-MSIDWC.md  
'''
     
import warnings
warnings.filterwarnings('ignore')   
import os, sys     
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')    
from calflops import calculate_flops
import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
 

class MultiScaleStripDW(nn.Module):  
    """Multi-scale strip depthwise conv with learnable softmax-weighted fusion."""   
    def __init__(self, channels, kernel_sizes=(7, 11, 15), horizontal=True):
        super().__init__() 
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels,     
                      kernel_size=(1, ks) if horizontal else (ks, 1),   
                      padding=(0, ks // 2) if horizontal else (ks // 2, 0),
                      groups=channels) 
            for ks in kernel_sizes
        ])     
        # Learnable scale weights; zeros → uniform (1/N) at init via softmax
        self.scale = nn.Parameter(torch.zeros(len(kernel_sizes)))
 
    def forward(self, x):
        weights = torch.softmax(self.scale.to(x.dtype), dim=0)
        out = None   
        for i, conv in enumerate(self.convs):  
            feat = weights[i] * conv(x)    
            out = feat if out is None else out + feat
        return out

   
class MSIDWC(nn.Module):    
    """Multi-Scale Strip Inception Depthwise Conv (MSIDWC).    

    Upgrades the single-scale 1×K / K×1 strip branches of InceptionDWConv2d    
    to multi-scale strip aggregation: three strip kernels of different sizes
    (default 7, 11, 15) are applied in parallel and fused via learnable
    softmax weights, capturing short-, medium-, and long-range directional context.     
    The square DW branch and identity branch remain unchanged.
    """
    def __init__(self, in_channels, out_channels,     
                 square_kernel_size=3, strip_kernel_sizes=(7, 11, 15),  
                 branch_ratio=0.125):     
        super().__init__()    
        gc = int(in_channels * branch_ratio)  
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc) 
   
        self.dwconv_hw  = nn.Conv2d(gc, gc, square_kernel_size,
                                    padding=square_kernel_size // 2, groups=gc)
        self.msstrip_w  = MultiScaleStripDW(gc, strip_kernel_sizes, horizontal=True)    
        self.msstrip_h  = MultiScaleStripDW(gc, strip_kernel_sizes, horizontal=False)
 
        self.conv1x1 = Conv(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        out = torch.cat([   
            x_id,
            self.dwconv_hw(x_hw),
            self.msstrip_w(x_w), 
            self.msstrip_h(x_h),
        ], dim=1)
        return self.conv1x1(out)

     
if __name__ == '__main__':    
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for MSIDWC half precision validation.')
    device = torch.device('cuda')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32     
    inputs = torch.randn((batch_size, in_channel, height, width), device=device).half() 

    module = MSIDWC(in_channel, out_channel).to(device).half()   
     
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
