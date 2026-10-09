'''
本文件由BiliBili：魔傀面具整理    
ultralytics/nn/module_images/ICIP2026-SDTA.png  
ultralytics/nn/module_images/ICIP2026-SDTA.md
论文链接：https://arxiv.org/pdf/2605.10148v1   
'''  
  
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')  
     
import warnings    
warnings.filterwarnings('ignore')   
from calflops import calculate_flops

import torch    
import torch.nn as nn
from ultralytics.nn.modules.conv import RepConv, Conv
  
from ultralytics.plugins.torch_utils import model_fuse_test    
    
class SDTA(nn.Module):
    def __init__(self, dim, qk_dim=16, n_div=4):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim   
        self.pdim = dim // n_div
        self.split_index = (qk_dim, qk_dim, self.pdim, dim-self.pdim)  
        self.pre_norm = nn.GroupNorm(1, dim)     
        hid = (qk_dim*2)+dim  
        self.in_proj = nn.Sequential(    
                        RepConv(dim, dim, g=dim),
                        Conv(dim, hid, act=False))  
 
        self.out_proj = nn.Sequential(nn.GELU(), Conv(dim, dim, act=False))
     
    def forward(self, x):    
        x = self.pre_norm(x) 
        q, k, v, u = self.in_proj(x).split(self.split_index, dim=1) 
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)  
        
        attn = (q.transpose(-2, -1) @ k) * self.scale     
        attn = attn.softmax(dim = -1)
    
        B, _, H, W = u.shape
        attn = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)     
        out  = self.out_proj(torch.cat((attn, u), dim=1)) 
     
        return out
    
if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for SDTA half precision validation.')
    device = torch.device('cuda')
    batch_size, channel, height, width = 1, 16, 20, 20
    inputs = torch.randn((batch_size, channel, height, width), device=device).half()    

    module = SDTA(channel).to(device).half()

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(GREEN + 'test reparameterization.' + RESET)  
    module = model_fuse_test(module)    
    outputs = module(inputs)   
    print(GREEN + 'test reparameterization done.' + RESET)

    module = module.float()   

    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),     
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)     
    print(RESET) 
