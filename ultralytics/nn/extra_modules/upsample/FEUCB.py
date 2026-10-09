'''  
本文件由BiliBili：魔傀面具整理
ultralytics/nn/module_images/自研模块-FEUCB.drawio
ultralytics/nn/module_images/自研模块-FEUCB.md  
'''

import warnings  
warnings.filterwarnings('ignore')     
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
from calflops import calculate_flops    
import numpy as np
import torch
import torch.nn as nn     
import torch.nn.functional as F  

from ultralytics.nn.modules.conv import Conv


class HaarWaveletSynthesis(nn.Module):
    '''
    Haar 小波逆变换（合成）算子。输入按通道分组的四个子带 [LL, LH, HL, HH]
    （每 4 个通道对应 1 个输出通道），通过固定的 Haar 逆变换滤波器重构出
    2 倍分辨率的特征图，用作高频补偿分支的空间重建器。
    '''     
    def __init__(self):
        super().__init__()    
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])    
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]]) 
        filts = np.stack([     
            ll[None, ::-1, ::-1], 
            lh[None, ::-1, ::-1],
            hl[None, ::-1, ::-1],
            hh[None, ::-1, ::-1],   
        ], axis=0)
        self.register_buffer('weight_base', torch.tensor(filts.copy()).to(torch.get_default_dtype()))

    def forward(self, subbands):   
        # subbands: (B, C, 4, H, W), 每个通道内顺序为 [LL, LH, HL, HH]   
        b, c, _, h, w = subbands.shape    
        x = subbands.reshape(b, c * 4, h, w)     
        weight = self.weight_base.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
        y = F.conv_transpose2d(x, weight, groups=c, stride=2)
        return y 
 
 
class FEUCB(nn.Module):   
    '''     
    Frequency-Compensated Efficient Up-Convolution Block (FEUCB)
     
    在原始 EUCB 主分支（nearest 上采样 + 深度可分离卷积 + 通道打乱 + 逐点卷积）之外，
    并联一个可学习的 Haar 小波高频补偿分支：将输入特征本身视为近似（LL）子带，
    通过深度卷积从输入直接预测 LH/HL/HH 高频子带，经 Haar 小波逆变换重构出与主分支
    同分辨率的高频细节图，再通过通道门控与主分支自适应融合，专门补偿 nearest 上采样
    过程中丢失的边缘与纹理信息。门控退化为 0 时，模块严格等价于原始 EUCB。     
    '''
    def __init__(self, in_channels, kernel_size=3):     
        super(FEUCB, self).__init__()   
     
        self.in_channels = in_channels
        self.out_channels = in_channels  

        # ---- 主分支：与原始 EUCB 一致 ----
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(self.in_channels, self.in_channels, kernel_size, g=self.in_channels) 
        ) 
        self.pwc = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
 
        # ---- 高频补偿分支：LL 取原始输入，LH/HL/HH 由深度卷积预测 ----  
        self.freq_pred = nn.Conv2d(self.in_channels, self.in_channels * 3, kernel_size=3,     
                                    padding=1, groups=self.in_channels, bias=True) 
        self.haar_synth = HaarWaveletSynthesis()  

        # ---- 门控融合：主分支与高频补偿分支自适应加权 ----
        self.gate = nn.Sequential(
            nn.Conv2d(self.in_channels * 2, self.in_channels, kernel_size=1, bias=True), 
            nn.Sigmoid()  
        )  

    def forward(self, x):   
        # 主分支
        y_main = self.up_dwc(x)
        y_main = self.channel_shuffle(y_main, self.in_channels) 
        y_main = self.pwc(y_main)
 
        # 高频补偿分支
        lh, hl, hh = self.freq_pred(x).chunk(3, dim=1)
        subbands = torch.stack([x, lh, hl, hh], dim=2)  # (B, C, 4, H, W)
        y_freq = self.haar_synth(subbands)               # (B, C, 2H, 2W)   

        # 门控融合：g -> 0 时退化为原始 EUCB
        g = self.gate(torch.cat([y_main, y_freq], dim=1))
        out = y_main + g * y_freq   
        return out

    def channel_shuffle(self, x, groups): 
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()   
        x = x.view(batchsize, -1, height, width) 
        return x
   
  
if __name__ == '__main__':     
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m" 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 16, 32, 32
    inputs = torch.randn((batch_size, channel, height, width)).to(device)     

    module = FEUCB(channel).to(device)    

    outputs = module(inputs)   
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    
    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module, 
                                     input_shape=(batch_size, channel, height, width),     
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)
