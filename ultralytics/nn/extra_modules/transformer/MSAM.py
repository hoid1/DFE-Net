'''    
本文件由BiliBili：魔傀面具整理
ultralytics/nn/module_images/TGRS2026-MSAM.png
ultralytics/nn/module_images/TGRS2026-MSAM.md  
论文链接：https://ieeexplore.ieee.org/document/11474595
'''    

import warnings
warnings.filterwarnings('ignore')
try:
    from calflops import calculate_flops
except Exception:      # calflops 只在文件底部的 __main__ 自测里用，缺了不影响建图
    calculate_flops = None
   
import numbers
import torch    
import torch.nn as nn
import torch.nn.functional as F
     
   
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):     
        super().__init__()    
        if isinstance(normalized_shape, numbers.Integral):     
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))    
    
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
     
     
class LayerNorm(nn.Module):
    def __init__(self, channels):    
        super().__init__()  
        self.body = WithBias_LayerNorm(channels)   

    def forward(self, x): 
        # Preserves the original module's flatten-and-restore behavior.  
        b, c, h, w = x.shape     
        return x.permute(0, 2, 3, 1).reshape(b, h * w, c).reshape(b, h, w, c).permute(0, 3, 1, 2)

  
class FeedForward(nn.Module):
    def __init__(self, channels, ffn_expansion_factor, bias):     
        super().__init__()
        hidden = int(channels * ffn_expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=bias)  
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=bias) 
     
    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)   
        return self.project_out(F.gelu(x1) * x2)    


class Attention(nn.Module):   
    def __init__(self, channels, num_heads, bias):  
        super().__init__()
        self.num_heads = num_heads    
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))     
        self.qkv_0 = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)    
        self.qkv_1 = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        self.qkv1conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=bias)  
        self.qkv2conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=bias)     
        self.qkv3conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=bias)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)  

    def forward(self, x, mask=None):     
        b, c, h, w = x.shape
        q = self.qkv1conv(self.qkv_0(x))   
        k = self.qkv2conv(self.qkv_1(x))   
        v = self.qkv3conv(self.qkv_2(x))     
        if mask is not None:     
            q, k = q * mask, k * mask
  
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)    
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)   
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)   
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) * self.temperature     
        out = (attn.softmax(dim=-1) @ v).reshape(b, c, h, w)  
        return self.project_out(out)     
 

class MSA_Head(nn.Module):    
    def __init__(self, channels=64, num_heads=4, ffn_expansion_factor=4, bias=False):
        super().__init__()
        self.norm1 = LayerNorm(channels)
        self.attn = Attention(channels, num_heads, bias)  
        self.norm2 = LayerNorm(channels)    
        self.ffn = FeedForward(channels, ffn_expansion_factor, bias)  

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask) 
        return x + self.ffn(self.norm2(x))   


class MSAM(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.background = MSA_Head(channels)
        self.foreground = MSA_Head(channels)  
        self.fuse = nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1)    
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x): 
        mask = torch.sigmoid(self.conv(x).detach())
        xf = self.foreground(x, mask)
        xb = self.background(x, 1 - mask)  
        return self.out(x * self.fuse(torch.cat([xb, xf], dim=1)))

    
if __name__ == '__main__':    
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 128, 20, 20  
    inputs = torch.randn((batch_size, channel, height, width)).to(device)

    module = MSAM(channel).to(device)

    outputs = module(inputs)  
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for MSAM half precision validation.')
    half_inputs = torch.randn((batch_size, channel, height, width), device='cuda').half() 
    half_outputs = MSAM(channel).cuda().half()(half_inputs)  
    print(GREEN + f'half inputs.size:{half_inputs.size()} outputs.size:{half_outputs.size()} dtype:{half_outputs.dtype}' + RESET)

    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,     
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True,     
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)
