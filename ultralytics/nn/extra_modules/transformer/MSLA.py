'''
本文件由BiliBili：魔傀面具整理   
ultralytics/nn/module_images/MSLA.png
ultralytics/nn/module_images/MSLA.md  
论文链接：https://arxiv.org/pdf/2505.18823
'''   
 
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops
 
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttention(nn.Module):

    def __init__(self, dim, num_heads):  
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)    
     
    def forward(self, x):
        b, c, h, w = x.shape
     
        x = x.view(b, c, h * w).permute(0, 2, 1)  # (b, h*w, c)

        qkv = self.qkv(x).reshape(b, h * w, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)    
        q, k, v = qkv[0], qkv[1], qkv[2]

        key = F.softmax(k, dim=-1) 
        query = F.softmax(q, dim=-2)   
        context = key.transpose(-2, -1) @ v     
        x = (query @ context).reshape(b, h * w, c)

        x = self.proj(x) 
   
        x = x.permute(0, 2, 1).view(b, c, h, w)
  
        return x

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, padding=kernel_size // 2)     
        self.relu = nn.ReLU()
 
    def forward(self, x):
        residual = x     
        x = self.depthwise(x)  
        x = x + residual     
        x = self.relu(x)
        return x 
    
 

class MSLA(nn.Module):

    def __init__(self, dim, num_heads): 
        super().__init__()     
        self.dim = dim     
        self.num_heads = num_heads  
  
        self.dw_conv_3x3 = DepthwiseConv(dim // 4, kernel_size=3)
        self.dw_conv_5x5 = DepthwiseConv(dim // 4, kernel_size=5)    
        self.dw_conv_7x7 = DepthwiseConv(dim // 4, kernel_size=7)
        self.dw_conv_9x9 = DepthwiseConv(dim // 4, kernel_size=9)
    
        self.linear_attention = LinearAttention(dim = dim // 4, num_heads = num_heads)
 
        self.final_conv = nn.Conv2d(dim, dim, 1)

        self.scale_weights = nn.Parameter(torch.ones(4), requires_grad=True)  

    def forward(self, input_):    
        restore_bchw = input_.ndim == 4 
        if input_.ndim == 4: 
            b, c, h, w = input_.shape
            input_ = input_.flatten(2).transpose(1, 2).contiguous() 
        elif input_.ndim == 3:
            b, n, c = input_.shape    
            h = int(n ** 0.5)   
            w = int(n ** 0.5)   
            if h * w != n:
                raise ValueError(f"MSLA token input requires square token count, got N={n}.") 
        else:
            raise ValueError(f"MSLA expects input as [B, N, C] or [B, C, H, W], got {tuple(input_.shape)}")
 
        b, n, c = input_.shape
     
        input_reshaped = input_.view(b, c, h, w)

        split_size = c // 4     
        x_3x3 = input_reshaped[:, :split_size, :, :]
        x_5x5 = input_reshaped[:, split_size:2 * split_size, :, :] 
        x_7x7 = input_reshaped[:, 2 * split_size:3 * split_size:, :, :]
        x_9x9 = input_reshaped[:, 3 * split_size:, :, :]
    
        x_3x3 = self.dw_conv_3x3(x_3x3)
        x_5x5 = self.dw_conv_5x5(x_5x5)
        x_7x7 = self.dw_conv_7x7(x_7x7)
        x_9x9 = self.dw_conv_9x9(x_9x9)  

    
        att_3x3 = self.linear_attention(x_3x3)     
        att_5x5 = self.linear_attention(x_5x5)  
        att_7x7 = self.linear_attention(x_7x7)
        att_9x9 = self.linear_attention(x_9x9)     

     
        processed_input = torch.cat([     
            att_3x3 * self.scale_weights[0],
            att_5x5 * self.scale_weights[1], 
            att_7x7 * self.scale_weights[2],   
            att_9x9 * self.scale_weights[3]    
        ], dim=1)  

        final_output = self.final_conv(processed_input)   
   
        output_reshaped = final_output.reshape(b, n, self.dim)
 
   
        if restore_bchw: 
            return output_reshaped.transpose(1, 2).reshape(b, self.dim, h, w).contiguous()     
        return output_reshaped  

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 64, 20, 20
    inputs = torch.randn((batch_size, channel, height, width)).to(device)   
  
    module = MSLA(channel, num_heads=8).to(device)     
   
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
   
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,  
                                     input_shape=(batch_size, channel, height, width),
                                     output_as_string=True,    
                                     output_precision=4, 
                                     print_detailed=True)
    print(RESET)    
