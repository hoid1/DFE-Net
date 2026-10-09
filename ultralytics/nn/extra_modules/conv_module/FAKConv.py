'''
本文件由BiliBili：魔傀面具整理   
ultralytics/nn/module_images/自研模块-FAKConv.drawio
ultralytics/nn/module_images/自研模块-FAKConv.md   
'''
    
import warnings
warnings.filterwarnings('ignore')
try:
    from calflops import calculate_flops
except Exception:      # calflops 只在文件底部的 __main__ 自测里用，缺了不影响建图
    calculate_flops = None
  
import torch
import torch.nn as nn 
import torch.nn.functional as F     
from ultralytics.nn.modules.conv import Conv, DSConv
     

class _HaarFrequencyDecomposition(nn.Module):
    """Depthwise 2x2 Haar transform, split into a low-frequency map and an   
    aggregated high-frequency (LH/HL/HH) response map."""
    def __init__(self, channels):    
        super().__init__()   
        self.channels = channels
 
        weights = torch.ones(4, 1, 2, 2)
        weights[1, 0, 0, 1] = -1     
        weights[1, 0, 1, 1] = -1  
        weights[2, 0, 1, 0] = -1    
        weights[2, 0, 1, 1] = -1    
        weights[3, 0, 1, 0] = -1
        weights[3, 0, 0, 1] = -1 
        self.register_buffer('weights', weights.repeat(channels, 1, 1, 1), persistent=False)
  
    def forward(self, x): 
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:    
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate') 
   
        out = F.conv2d(x, self.weights.to(x.dtype), bias=None, stride=2, groups=self.channels) / 4.0
        b, _, h, w = out.shape
        out = out.view(b, self.channels, 4, h, w) 
        low = out[:, :, 0]
        high = out[:, :, 1:].abs().sum(dim=2)
        return low, high   

 
class FAKConv(nn.Module):
    """Frequency-Aware Kernel Convolution (FAKConv).  
  
    Upgrades CKConv's purely spatial multi-kernel branches (square DW body +
    H/V strip head, one branch per kernel size in `kk`) by adding a parallel
    lightweight Haar frequency branch that explicitly separates low-frequency
    structure from high-frequency detail. The spatial multi-kernel feature and
    the frequency feature are fused through a content-adaptive dual-domain
    gate rather than a plain sum/concat, and the projected input is kept as a 
    residual for stable optimization.     
    """    
    def __init__(self, c1, c2, kk=[3, 5, 7], s=1):  
        super().__init__()
   
        if not isinstance(kk, list) or not all(ki in [3, 5, 7, 9] for ki in kk):  
            raise ValueError("k must be a list containing 3, 5, and/or 7")

        self.kk = kk     
        self.c1 = c1
        self.c2 = c2
        self.s = s    
  
        self.conv_1x1 = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()     

        self.branches = nn.ModuleDict()    
        for ki in kk:
            self.branches[f'k{ki}_body'] = Conv(c2, c2 // 2, (3, 3), s=1, g=c2 // 2)    
            self.branches[f'k{ki}_head_h'] = Conv(c2, c2 // 2, (1, ki), s=s, p=(0, (ki - 1) // 2), g=c2 // 2)  
            self.branches[f'k{ki}_head_v'] = Conv(c2 // 2, c2 // 2, (ki, 1), s=s, p=((ki - 1) // 2, 0), g=c2 // 2)
            self.branches[f'k{ki}_conv2'] = nn.Conv2d(c2 // 2, c2, 1, groups=c2 // 2)

        self.spatial_fuse = nn.Conv2d(len(kk) * c2, c2, 1, groups=16)  # note 1   

        self.frequency = _HaarFrequencyDecomposition(c2)    
        self.low_proj = Conv(c2, c2, 1)
        self.high_proj = Conv(c2, c2, 1)
        self.freq_fuse = DSConv(c2 * 2, c2, 3)     
     
        gate_hidden = max(c2 // 4, 8)
        self.branch_gate = nn.Sequential( 
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2 * 2, gate_hidden, 1, bias=False),
            nn.SiLU(inplace=True),     
            nn.Conv2d(gate_hidden, 2, 1, bias=True),
        )     
        self.spatial_scale = nn.Parameter(torch.tensor(1.0))
        self.frequency_scale = nn.Parameter(torch.tensor(0.1))    
  
    def forward(self, x):  
        x = self.conv_1x1(x)

        spatial_outputs = []    
        for ki in self.kk:
            y = self.branches[f'k{ki}_head_h'](x) 
            y = self.branches[f'k{ki}_head_v'](y)
            ys = self.branches[f'k{ki}_body'](x)
            out = ys + y     
            out = self.branches[f'k{ki}_conv2'](out)  
            spatial_outputs.append(out)    
        spatial_feature = self.spatial_fuse(torch.cat(spatial_outputs, dim=1)) 
 
        low, high = self.frequency(x)
        freq_feature = self.freq_fuse(torch.cat([self.low_proj(low), self.high_proj(high)], dim=1))
        freq_feature = F.interpolate(freq_feature, size=spatial_feature.shape[-2:], mode='bilinear', align_corners=False)

        branch_logits = self.branch_gate(torch.cat([spatial_feature, freq_feature], dim=1)) 
        spatial_weight, frequency_weight = torch.chunk(torch.softmax(branch_logits, dim=1), 2, dim=1)     

        residual = x if self.s == 1 else F.avg_pool2d(x, self.s)  
        out = residual + self.spatial_scale * spatial_weight * spatial_feature + self.frequency_scale * frequency_weight * freq_feature     
     
        return out     


if __name__ == '__main__':  
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32 
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
 
    module = FAKConv(in_channel, out_channel).to(device)   

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for FAKConv half precision validation.') 
    half_inputs = torch.randn((batch_size, in_channel, height, width), device='cuda').half()
    half_outputs = FAKConv(in_channel, out_channel).cuda().half()(half_inputs)   
    print(GREEN + f'half inputs.size:{half_inputs.size()} outputs.size:{half_outputs.size()} dtype:{half_outputs.dtype}' + RESET) 
    
    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width),     
                                     output_as_string=True,
                                     output_precision=4,   
                                     print_detailed=True)
    print(RESET)
