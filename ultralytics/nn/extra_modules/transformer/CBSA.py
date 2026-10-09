'''
本文件由BiliBili：魔傀面具整理    
ultralytics/nn/module_images/NeurIPS2025-CBSA.png
ultralytics/nn/module_images/NeurIPS2025-CBSA.md
论文链接：https://arxiv.org/abs/2509.16875
'''   
  
import warnings   
warnings.filterwarnings('ignore')
from calflops import calculate_flops  
 
import torch
import torch.nn as nn     
import torch.nn.functional as F 
from einops import rearrange

class CBSA(nn.Module):
    def __init__(self, dim, num_heads):   
        super().__init__() 
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5
  
        self.attend = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim, bias=False)  
        
        self.step_x = nn.Parameter(torch.randn(num_heads, 1, 1))
        self.step_rep = nn.Parameter(torch.randn(num_heads, 1, 1)) 
        
        self.to_out = nn.Linear(dim, dim)
        
        self.pool = nn.AdaptiveAvgPool2d(output_size=(8, 8))  
    
    def attention(self, query, key, value):  
        dots = (query @ key.transpose(-1, -2)) * self.scale 
        attn = self.attend(dots) 
        out = attn @ value
        return out, attn 
     
    def forward(self, x, return_attn=False):
        restore_bchw = x.ndim == 4 
        if x.ndim == 4:
            b, c, h, width = x.shape
            x = x.flatten(2).transpose(1, 2).contiguous()
        elif x.ndim == 3:    
            b, n, c = x.shape
            h = width = int(n ** 0.5)     
            if h * width != n:     
                raise ValueError(f"CBSA token input requires square token count, got N={n}.")
        else:
            raise ValueError(f"CBSA expects input as [B, N, C] or [B, C, H, W], got {tuple(x.shape)}")    

        b, n, c = x.shape    
   
        w = self.proj(x)
        rep = self.pool(w[:, :, :].reshape(b, h, width, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
     
        w = w.reshape(b, n, self.num_heads, self.dim_head).permute(0, 2, 1, 3)   
        rep = rep.reshape(b, 64, self.num_heads, self.dim_head).permute(0, 2, 1, 3)     

        rep_delta, attn = self.attention(rep, w, w)
   
        if return_attn:  
            return attn.transpose(-1, -2) @ attn     
  
        rep = rep + self.step_rep * rep_delta   
        
        x_delta, _ = self.attention(rep, rep, rep)  
        x_delta = attn.transpose(-1, -2) @ x_delta
        x_delta = self.step_x * x_delta
        
        x_delta = rearrange(x_delta, 'b h n k -> b n (h k)')
        out = self.to_out(x_delta)
        if restore_bchw:
            return out.transpose(1, 2).reshape(b, c, h, width).contiguous()
        return out     
  
if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"     
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    batch_size, channel, height, width = 1, 128, 20, 20
    inputs = torch.randn((batch_size, channel, height, width)).to(device)

    module = CBSA(channel, num_heads=8).to(device)
    
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)    
     
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, channel, height, width),  
                                     output_as_string=True,
                                     output_precision=4,  
                                     print_detailed=True)
    print(RESET)     
