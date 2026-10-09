import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
  
import warnings
warnings.filterwarnings('ignore')     

import torch
import torch.nn as nn   
from ultralytics.nn.modules.conv import Conv  
from ultralytics.nn.modules.block import Bottleneck

import thop   
from functools import partial 
from copy import deepcopy
from ultralytics.utils.torch_utils import get_num_params
from ultralytics.plugins.torch_utils import CUDABenchmark     

from ultralytics.nn.extra_modules.block.MANet import MANet
from ultralytics.nn.extra_modules.block.MetaFormer import MetaFormer_Block, MetaFormer_SEFN, MetaFormer_Mona, MetaFormer_Mona_SEFN
from ultralytics.nn.extra_modules.module.DRG import DRG  
    
class C3_Block(nn.Module):   
    """CSP Bottleneck with 3 convolutions."""   
 
    def __init__(self, c1, c2, module=partial(Bottleneck, k=(1, 3), shortcut=True, e=0.5), n=1, e=0.5, selfatt=False):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""     
        super().__init__()   
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)  
        if selfatt:
            self.m = nn.Sequential(*(module(c_) for _ in range(n)))  
        else:   
            self.m = nn.Sequential(*(module(c_, c_) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""    
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))  
    
class C2f_Block(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
   
    def __init__(self, c1, c2, module=partial(Bottleneck, k=(3, 3), shortcut=True, e=0.5), n=1, e=0.5, selfatt=False):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels 
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2) 
        if selfatt:
            self.m = nn.ModuleList(module(self.c) for _ in range(n))
        else:  
            self.m = nn.ModuleList(module(self.c, self.c) for _ in range(n))     

    def forward(self, x):  
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)   
        return self.cv2(torch.cat(y, 1))
     
class C3k_Block(nn.Module):    
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, module=partial(Bottleneck, k=(3, 3), shortcut=True, e=1.0), n=1, e=0.5, selfatt=False):  
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""  
        super().__init__()  
        c_ = int(c2 * e)  # hidden channels   
        self.cv1 = Conv(c1, c_, 1, 1) 
        self.cv2 = Conv(c1, c_, 1, 1)    
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        if selfatt:
            self.m = nn.Sequential(*(module(c_) for _ in range(n))) 
        else:     
            self.m = nn.Sequential(*(module(c_, c_) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""   
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))  

class C3k2_Block(nn.Module):    
    def __init__(self, c1, c2, module=partial(Bottleneck, k=(3, 3), shortcut=True, e=0.5), n=1, c3k=True, e=0.5, selfatt=False):  
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)     
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        if selfatt:
            self.m = nn.ModuleList(
                    C3k_Block(self.c, self.c, module, 2, selfatt=selfatt) if c3k else module(self.c) for _ in range(n)
                )
        else:    
            self.m = nn.ModuleList(   
                    C3k_Block(self.c, self.c, module, 2) if c3k else module(self.c, self.c) for _ in range(n)
                )     
    
    def forward(self, x):    
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)    
        return self.cv2(torch.cat(y, 1))     
   
if __name__ == '__main__':
    from ultralytics.utils.torch_utils import select_device
    
    device_id = '0'
    in_channel, out_channel, height, width = 128, 256, 160, 160
    repeats = 2  
 
    torch_device = select_device(device_id)
    inputs_tensor = torch.randn((1, in_channel, height, width)).to(torch_device)
    
    module_instance = partial(Bottleneck, k=(3, 3), shortcut=True, e=0.5)  
    # module_instance = partial(DRG, kernel_size=3, n_resblocks=2)    
    selfatt = False     
     
    module_list = [
        # ------ CSP
        C2f_Block(in_channel, out_channel, module=module_instance, n=repeats, selfatt=selfatt),     
        C3_Block(in_channel, out_channel, module=module_instance, n=repeats, selfatt=selfatt),    
        C3k_Block(in_channel, out_channel, module=module_instance, n=repeats, selfatt=selfatt),     
        C3k2_Block(in_channel, out_channel, module=module_instance, n=repeats, c3k=True, selfatt=selfatt),     
        MANet(in_channel, out_channel, module=module_instance, n=repeats),

        # ------ MetaFormer
        MetaFormer_Block(in_channel, out_channel, token_mixer=module_instance, selfatt=selfatt),    
        MetaFormer_Mona(in_channel, out_channel, token_mixer=module_instance, selfatt=selfatt),   
        MetaFormer_SEFN(in_channel, out_channel, token_mixer=module_instance, selfatt=selfatt),  
        MetaFormer_Mona_SEFN(in_channel, out_channel, token_mixer=module_instance, selfatt=selfatt)
    ]    
  
    for module_ in module_list:    
        module_.to(torch_device)
  
        module_name = module_.__class__.__name__ 
        n_p = get_num_params(module_)
        n_flops = thop.profile(deepcopy(module_), inputs=[inputs_tensor], verbose=False)[0] / 1e9 * 2
        n_flops_str, n_p_str = thop.clever_format([n_flops, n_p], format="%.3f")

        benchmark = CUDABenchmark(name=f'Module:{module_name} Flops:{n_flops_str} Parameters:{n_p_str}', warmup=500, repeat=1000)
        benchmark.run(lambda: module_(inputs_tensor))  
        benchmark.print_stats()   
