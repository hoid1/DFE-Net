'''
本文件由BiliBili：魔傀面具整理
ultralytics/nn/module_images/TGRS2026-WFAM.png  
ultralytics/nn/module_images/TGRS2026-WFAM.md   
论文链接：https://ieeexplore.ieee.org/document/11474595     
''' 
     
import warnings  
warnings.filterwarnings('ignore')
try:
    from calflops import calculate_flops
except Exception:      # calflops 只在文件底部的 __main__ 自测里用，缺了不影响建图
    calculate_flops = None
  
import torch 
from torch import nn
from torch.nn import functional as F   
from pytorch_wavelets import DWTForward, DWTInverse
 
class ConvModule(nn.Module):
    def __init__(self, channels):  
        super().__init__()
        self.conv = nn.Sequential(    
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),    
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

    
class LLAttention(nn.Module):   
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):   
        super().__init__()     
        if dim % num_heads:
            num_heads = 1  
        self.num_heads = num_heads 
        self.head_dim = dim // num_heads   
        self.scale = self.head_dim**-0.5   
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias) 
        self.proj = nn.Linear(dim, dim)     
        self.attn_drop = nn.Dropout(attn_drop) 
        self.proj_drop = nn.Dropout(proj_drop)     
   
    def forward(self, x):
        batch, channels, height, width = x.shape    
        tokens = x.reshape(batch, channels, height * width).permute(0, 2, 1)    
        q = self.q(tokens).reshape(batch, height * width, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 
        kv = self.kv(tokens).reshape(batch, height * width, 2, self.num_heads, self.head_dim)   
        key, value = kv.permute(2, 0, 3, 1, 4)
        attention = self.attn_drop((q @ key.transpose(-2, -1) * self.scale).softmax(dim=-1))
        output = (attention @ value).transpose(1, 2).reshape(batch, height * width, channels)   
        return self.proj_drop(self.proj(output)).transpose(1, 2).reshape(batch, channels, height, width)  


class WaveletAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()  
        self.dwt = DWTForward(J=1, mode="zero", wave="haar")
        self.idwt = DWTInverse(wave="haar")   
        self.high_hl = ConvModule(channels)
        self.high_lh = ConvModule(channels)
        self.high_hh = ConvModule(channels)
        self.low_attention = LLAttention(channels)

    def forward(self, x):
        low, high = self.dwt(x)   
        high = high[0]
        high = torch.stack(     
            [self.high_hl(high[:, :, 0]), self.high_lh(high[:, :, 1]), self.high_hh(high[:, :, 2])], dim=2
        )
        return self.idwt((self.low_attention(low), [high])) + x  

  
class WFAM(nn.Module): 
    """Wavelet feature enhancement and alignment block.
    
    ``forward`` returns the enhanced first feature map.  The internally   
    estimated warp is intentionally not exposed.
    """    
    
    def __init__(self, in_channels, out_channels):
        super().__init__()  
        x_channels, y_channels = in_channels
        self.x_proj = nn.Conv2d(x_channels, out_channels, 1)
        self.y_proj = nn.Conv2d(y_channels, out_channels, 1)   
        self.wavelet = WaveletAttention(out_channels)
        self.offset_conv = nn.Conv2d(out_channels * 2, 2, kernel_size=3, padding=1)    
   
    @staticmethod
    def flow_warp(x, flow):
        batch, _, height, width = x.shape
        norm = x.new_tensor([width, height]).view(1, 1, 1, 2)   
        rows = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype).repeat(height, 1)
        cols = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype).view(-1, 1).repeat(1, width)  
        grid = torch.stack((rows, cols), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)    
        return F.grid_sample(x, grid + flow.permute(0, 2, 3, 1) / norm, align_corners=True)

    def forward(self, inputs):
        x, y = inputs 
        x, y = self.x_proj(x), self.y_proj(y)
        x, y = self.wavelet(x), self.wavelet(y)  
        flow = self.offset_conv(torch.cat((x, y), dim=1))   
        return self.flow_warp(y, flow)   
   
   
if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"  
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32
    ouc_channel = 32   
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device)    
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device)
  
    module = WFAM([channel_1, channel_2], ouc_channel).to(device)

    outputs = module([inputs_1, inputs_2])   
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET) 

    if not torch.cuda.is_available():     
        raise RuntimeError('CUDA is required for WFAM half precision validation.')   
    half_inputs = [torch.randn((batch_size, c, height, width), device='cuda').half() for c in (channel_1, channel_2)]
    half_outputs = WFAM([channel_1, channel_2], ouc_channel).cuda().half()(half_inputs)
    print(GREEN + f'half outputs.size:{half_outputs.size()} dtype:{half_outputs.dtype}' + RESET)   
  
    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,
                                     args=[[inputs_1, inputs_2]],
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)
