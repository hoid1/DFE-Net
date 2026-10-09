'''
本文件由BiliBili：魔傀面具整理
ultralytics/nn/module_images/自研模块-AGIDWC.png
ultralytics/nn/module_images/自研模块-AGIDWC.md
'''     
 
import warnings
warnings.filterwarnings('ignore')    
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')  
from calflops import calculate_flops
import torch
import torch.nn as nn 
    
from ultralytics.nn.modules.conv import Conv


class AGIDWC(nn.Module):
    """Adaptive-Gated Inception Depthwise Conv (AGIDWC).  
  
    Extends InceptionDWConv2d with content-adaptive branch gating: 
    a lightweight GAP→MLP generates 4 per-branch dynamic weights,   
    enabling input-conditioned routing across all Inception branches.
    Gates are initialised to ≈1.0 so training starts from the IDWC baseline. 
    """   
    def __init__(self, in_channels, out_channels,    
                 square_kernel_size=3, band_kernel_size=11, 
                 branch_ratio=0.125, reduction=4):  
        super().__init__()    
        gc = int(in_channels * branch_ratio)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)    
   
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size,    
                                   padding=square_kernel_size // 2, groups=gc)    
        self.dwconv_w  = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size),   
                                   padding=(0, band_kernel_size // 2), groups=gc)  
        self.dwconv_h  = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), 
                                   padding=(band_kernel_size // 2, 0), groups=gc)

        # Adaptive branch gate: GAP → MLP → 4 branch weights ∈ (0, 2)  
        hidden = max(in_channels // reduction, 16)
        self.branch_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(1),
            nn.Linear(in_channels, hidden, bias=False),     
            nn.ReLU(inplace=True),     
            nn.Linear(hidden, 4, bias=True)   # bias=True, initialised to 0 → gate≈1.0
        )

        self.conv1x1 = Conv(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
 
    def forward(self, x):
        # gates ∈ (0, 2); at init sigmoid(0)×2 = 1.0 → behaves like plain IDWC
        gates = 2.0 * torch.sigmoid(self.branch_gate(x))  # (B, 4)
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        out = torch.cat([
            x_id                * gates[:, 0].view(-1, 1, 1, 1),   
            self.dwconv_hw(x_hw) * gates[:, 1].view(-1, 1, 1, 1),
            self.dwconv_w(x_w)  * gates[:, 2].view(-1, 1, 1, 1),   
            self.dwconv_h(x_h)  * gates[:, 3].view(-1, 1, 1, 1),
        ], dim=1)
        return self.conv1x1(out)     
     

if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for AGIDWC half precision validation.')
    device = torch.device('cuda')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width), device=device).half()    
     
    module = AGIDWC(in_channel, out_channel).to(device).half()
    
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
