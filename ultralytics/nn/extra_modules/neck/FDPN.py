'''
本文件由BiliBili：魔傀面具整理
ultralytics/cfg/models/improve/neck/*/*-(FDPN/DFFPN/AGFDPN).yaml 的架构图：ultralytics/nn/module_images/自研模块-FDPN.png

ultralytics/nn/module_images/自研模块FocusFeature.png
ultralytics/nn/module_images/自研模块-AlignmentGuidedFocusFeature.png
ultralytics/nn/module_images/自研模块-AlignmentGuidedFocusFeature.md  
ultralytics/nn/module_images/自研模块-DynamicFrequencyFocusFeature.png
ultralytics/nn/module_images/自研模块-DynamicFrequencyFocusFeature.md
''' 

import torch     
import torch.nn as nn 
import torch.nn.functional as F 

from ultralytics.nn.modules import Conv, ADown   
from ultralytics.nn.modules.conv import autopad    

class FocusFeature(nn.Module):
    def __init__(self, inc, kernel_sizes=(5, 7, 9, 11), e=0.5) -> None:    
        super().__init__()    
        hidc = int(inc[1] * e)
        
        self.conv1 = nn.Sequential(    
            nn.Upsample(scale_factor=2),     
            Conv(inc[0], hidc, 1)
        )
        self.conv2 = Conv(inc[1], hidc, 1) if e != 1 else nn.Identity()    
        self.conv3 = ADown(inc[2], hidc)  
        
        self.dw_conv = nn.ModuleList(nn.Conv2d(hidc * 3, hidc * 3, kernel_size=k, padding=autopad(k), groups=hidc * 3) for k in kernel_sizes)
        self.pw_conv = Conv(hidc * 3, hidc * 3)
        self.conv_1x1 = Conv(hidc * 3, int(hidc / e))   
   
    def forward(self, x):
        x1, x2, x3 = x
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)     
     
        x = torch.cat([x1, x2, x3], dim=1)
        feature = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv], dim=0), dim=0) 
        feature = self.pw_conv(feature)   
    
        x = x + feature 
        return self.conv_1x1(x)  

class _AlignedFocusInputs(nn.Module):
    def __init__(self, inc, hidc, guided=False):    
        super().__init__()    
        self.low_to_mid = nn.Sequential( 
            nn.Upsample(scale_factor=2),
            Conv(inc[0], hidc, 1)
        )  
        self.mid_proj = Conv(inc[1], hidc, 1) if inc[1] != hidc else nn.Identity()
        self.high_to_mid = ADown(inc[2], hidc)
        self.guided = guided    
  
        if guided:
            self.low_align = _CrossScaleGuidedAlign(hidc)    
            self.high_align = _CrossScaleGuidedAlign(hidc) 

    def forward(self, x):    
        x_low, x_mid, x_high = x
        x_low = self.low_to_mid(x_low)
        x_mid = self.mid_proj(x_mid)
        x_high = self.high_to_mid(x_high)
    
        if self.guided:
            x_low = self.low_align(x_low, x_mid)
            x_high = self.high_align(x_high, x_mid)
  
        return x_low, x_mid, x_high 


class _MultiKernelSpatialFocus(nn.Module):
    def __init__(self, channels, kernel_sizes, dynamic=False):     
        super().__init__()
        self.dynamic = dynamic
        self.dw_conv = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=k, padding=autopad(k), groups=channels)
            for k in kernel_sizes
        ) 
        self.pw_conv = Conv(channels, channels, 1)   
 
        if dynamic:   
            hidden = max(channels // 4, len(kernel_sizes) + 1) 
            self.kernel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, kernel_size=1, bias=False),    
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, len(kernel_sizes) + 1, kernel_size=1, bias=True),     
            )
 
    def forward(self, x):
        branches = [x] + [layer(x) for layer in self.dw_conv]
        if self.dynamic:
            branch_weights = torch.softmax(self.kernel_gate(x), dim=1).unsqueeze(2)
            stacked = torch.stack(branches, dim=1)    
            feature = torch.sum(branch_weights * stacked, dim=1)
        else:   
            feature = torch.sum(torch.stack(branches, dim=0), dim=0)  
        return self.pw_conv(feature)  

     
class _HaarFrequencyDecomposition(nn.Module):   
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
        self.register_buffer('weights', torch.cat([weights] * channels, dim=0), persistent=False) 
  
    def forward(self, x):
        pad_h = x.shape[-2] % 2   
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:   
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
   
        out = F.conv2d(x, self.weights, bias=None, stride=2, groups=self.channels) / 4.0  
        batch_size, _, height, width = out.shape 
        out = out.view(batch_size, self.channels, 4, height, width)
        low = out[:, :, 0] 
        high = out[:, :, 1:].abs().sum(dim=2)  
        return low, high


class _CrossScaleGuidedAlign(nn.Module):   
    def __init__(self, channels):  
        super().__init__()     
        self.context = Conv(channels * 4, channels, 1)    
        self.refine = Conv(channels, channels, 3, g=channels)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),     
            nn.Sigmoid(),   
        )
    
    def forward(self, source, target):
        discrepancy = torch.abs(source - target)   
        consistency = source * target
        context = self.context(torch.cat([source, target, discrepancy, consistency], dim=1)) 
        return source + self.gate(context) * self.refine(context)
     
class DynamicFrequencyFocusFeature(nn.Module):    
    def __init__(self, inc, kernel_sizes=(5, 7, 9, 11), e=0.5):    
        super().__init__()    
        hidc = int(inc[1] * e)
        channels = hidc * 3
     
        self.align = _AlignedFocusInputs(inc, hidc, guided=False)
        self.spatial_focus = _MultiKernelSpatialFocus(channels, kernel_sizes, dynamic=True)  
        self.frequency = _HaarFrequencyDecomposition(channels)    
        self.low_proj = Conv(channels, channels, 1)
        self.high_proj = Conv(channels, channels, 1)    
        self.freq_proj = Conv(channels * 2, channels, 3)
        self.branch_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, max(channels // 4, 8), kernel_size=1, bias=False),
            nn.SiLU(inplace=True),    
            nn.Conv2d(max(channels // 4, 8), 2, kernel_size=1, bias=True),
        )     
        self.spatial_scale = nn.Parameter(torch.tensor(1.0))
        self.frequency_scale = nn.Parameter(torch.tensor(0.1))   
        self.output = Conv(channels, int(hidc / e), 1)     

    def forward(self, x):
        x_low, x_mid, x_high = self.align(x)
        fused = torch.cat([x_low, x_mid, x_high], dim=1)
   
        spatial_feature = self.spatial_focus(fused)
  
        low, high = self.frequency(fused)
        frequency_feature = self.freq_proj(torch.cat([self.low_proj(low), self.high_proj(high)], dim=1)) 
        frequency_feature = F.interpolate(     
            frequency_feature,  
            size=fused.shape[-2:],
            mode='bilinear',     
            align_corners=False,   
        )
   
        branch_logits = self.branch_gate(torch.cat([spatial_feature, frequency_feature], dim=1))     
        branch_weights = torch.softmax(branch_logits, dim=1)
        spatial_weight, frequency_weight = torch.chunk(branch_weights, 2, dim=1)

        refined = (  
            fused   
            + self.spatial_scale * spatial_weight * spatial_feature   
            + self.frequency_scale * frequency_weight * frequency_feature
        )
        return self.output(refined)    
   
   
class AlignmentGuidedFocusFeature(nn.Module):
    def __init__(self, inc, kernel_sizes=(5, 7, 9, 11), e=0.5):
        super().__init__()
        hidc = int(inc[1] * e) 
    
        self.align = _AlignedFocusInputs(inc, hidc, guided=True)
        self.discrepancy_proj = Conv(hidc * 3, hidc, 3)
        self.consistency_proj = Conv(hidc * 3, hidc, 3) 
        self.branch_gate = nn.Sequential( 
            Conv(hidc * 5, hidc, 1),   
            nn.Conv2d(hidc, 3, kernel_size=1, bias=True),
        )     
        self.guidance_residual = nn.Sequential(   
            Conv(hidc * 2, hidc, 1),
            Conv(hidc, hidc, 3, g=hidc),
        )     
        self.refine_focus = _MultiKernelSpatialFocus(hidc, kernel_sizes, dynamic=True)     
        self.guidance_scale = nn.Parameter(torch.tensor(0.1)) 
        self.output = Conv(hidc, int(hidc / e), 1)     
   
    def forward(self, x):    
        x_low, x_mid, x_high = self.align(x)   

        discrepancy = self.discrepancy_proj(
            torch.cat(     
                [    
                    torch.abs(x_low - x_mid),
                    torch.abs(x_mid - x_high), 
                    torch.abs(x_low - x_high),
                ],  
                dim=1,
            )
        )   
        consistency = self.consistency_proj(
            torch.cat(  
                [  
                    x_low * x_mid,
                    x_mid * x_high,  
                    x_low * x_high,
                ],     
                dim=1,   
            )
        )  
     
        branch_logits = self.branch_gate(torch.cat([x_low, x_mid, x_high, discrepancy, consistency], dim=1))     
        branch_weights = torch.softmax(branch_logits, dim=1)   
        low_weight, mid_weight, high_weight = torch.chunk(branch_weights, 3, dim=1)     
   
        fused = low_weight * x_low + mid_weight * x_mid + high_weight * x_high
        guidance = self.guidance_residual(torch.cat([discrepancy, consistency], dim=1)) 
        refined = fused + self.refine_focus(fused) + self.guidance_scale * guidance   
        return self.output(refined)