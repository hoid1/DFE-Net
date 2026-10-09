'''
本文件由BiliBili：魔傀面具整理
ultralytics/nn/module_images/自研模块-CBIMS.png
ultralytics/nn/module_images/自研模块-CBIMS.md
'''    

import warnings
warnings.filterwarnings('ignore')
import os, sys 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
from calflops import calculate_flops 
import torch   
import torch.nn as nn
    
from ultralytics.nn.modules.conv import Conv
from ultralytics.plugins.torch_utils import model_fuse_test   
from ultralytics.nn.extra_modules.block.RepHMS import DepthBottleneckUniv2

    
class CrossBranchMixer(nn.Module):
    """Lateral cross-branch interaction via grouped channel mixing and asymmetric DW spatial convs."""
    def __init__(self, channels, n_branches):    
        super().__init__()  
        self.n = n_branches
        total = n_branches * channels
        # Grouped 1×1 conv: for each channel position, mixes across n_branches     
        self.mix = nn.Conv2d(total, total, 1, groups=channels, bias=False) 
        self.norm = nn.BatchNorm2d(total)
        self.act = nn.SiLU()     
        # Asymmetric DW spatial refinement (1×3 and 3×1)   
        self.dw_h = nn.Conv2d(total, total, (1, 3), padding=(0, 1), groups=total, bias=False)    
        self.dw_v = nn.Conv2d(total, total, (3, 1), padding=(1, 0), groups=total, bias=False)
 
    def forward(self, features):     
        x = torch.cat(features, dim=1)  # (B, n*C, H, W)
        residual = x  
        x = self.act(self.norm(self.mix(x)))  
        x = x + self.dw_h(x) + self.dw_v(x)     
        x = x + residual
        return x

     
class CBIMS(nn.Module):    
    def __init__(self, in_channels, out_channels, width=3, depth=1, depth_expansion=2, kersize=5, shortcut=True,
                 expansion=0.5, small_kersize=3, use_depthwise=True): 
        super(CBIMS, self).__init__()  
        self.width = width
        self.depth = depth     
        c1 = int(out_channels * expansion) * width     
        c_ = int(out_channels * expansion)   
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1)   
        self.RepElanMSBlock = nn.ModuleList()
        for _ in range(width - 1):
            DepthBlock = nn.ModuleList([
                DepthBottleneckUniv2(c_, c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise)  
                for _ in range(depth)
            ]) 
            self.RepElanMSBlock.append(DepthBlock)
   
        n_elan = 1 + (width - 1) * depth
        self.cbm = CrossBranchMixer(c_, n_elan)
   
        self.conv2 = Conv(c_ * n_elan, out_channels, 1)     
    
    def forward(self, x):
        x = self.conv1(x)
        x_out = [x[:, i * self.c_:(i + 1) * self.c_] for i in range(self.width)]     
        x_out[1] = x_out[1] + x_out[0]  
        cascade = []  
        elan = [x_out[0]] 

        for i in range(self.width - 1):
            for j in range(self.depth):
                if i > 0:
                    x_out[i + 1] = x_out[i + 1] + cascade[j]
                    if j == self.depth - 1:     
                        if self.depth > 1:
                            cascade = [cascade[-1]]  
                        else:
                            cascade = [] 
                x_out[i + 1] = self.RepElanMSBlock[i][j](x_out[i + 1])
                elan.append(x_out[i + 1])
                if i < self.width - 2:    
                    cascade.append(x_out[i + 1])
  
        y_out = self.cbm(elan)
        y_out = self.conv2(y_out)
        return y_out
  
  
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    if not torch.cuda.is_available():  
        raise RuntimeError('CUDA is required for CBIMS half precision validation.')
    device = torch.device('cuda') 
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32   
    inputs = torch.randn((batch_size, in_channel, height, width), device=device).half()  

    module = CBIMS(in_channel, out_channel, width=3, depth=1).to(device).half()

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
  
    print(GREEN + 'test reparameterization.' + RESET)     
    module = model_fuse_test(module)
    outputs = module(inputs)
    print(GREEN + 'test reparameterization done.' + RESET)   
  
    module = module.float()     

    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module,   
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,
                                     output_precision=4, 
                                     print_detailed=True)
    print(RESET)
