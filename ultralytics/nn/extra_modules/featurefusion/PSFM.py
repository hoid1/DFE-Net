'''
本文件由BiliBili：魔傀面具整理
ultralytics/nn/module_images/IF2023-PSFM.png
ultralytics/nn/module_images/IF2023-PSFM.md
论文链接：https://www.sciencedirect.com/science/article/abs/pii/S1566253523001860
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
   
import warnings    
warnings.filterwarnings('ignore')  
from calflops import calculate_flops
 
import torch
import torch.nn as nn 

from ultralytics.nn.modules import DSConv   
     
class GEFM(nn.Module):    
    """Global enhancement fusion module."""
   
    def __init__(self, in_channels, out_channels):
        super().__init__()  
        self.rgb_k = DSConv(out_channels, out_channels, 3)
        self.rgb_v = DSConv(out_channels, out_channels, 3)
        self.q = DSConv(in_channels, out_channels, 3)
        self.inf_k = DSConv(out_channels, out_channels, 3)  
        self.inf_v = DSConv(out_channels, out_channels, 3)
        self.second_reduce = DSConv(in_channels, out_channels, 3) 
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)    

    def forward(self, x, y): 
        q = self.q(torch.cat([x, y], dim=1))
        rgb_k = self.rgb_k(x)
        rgb_v = self.rgb_v(x)
        b, _, h, w = rgb_v.size()

        rgb_v = rgb_v.view(b, -1, w * h)
        rgb_k = rgb_k.view(b, -1, w * h).permute(0, 2, 1) 
        rgb_q = q.view(b, -1, w * h)
        rgb_mask = self.softmax(torch.bmm(rgb_k, rgb_q))
        rgb_refine = torch.bmm(rgb_v, rgb_mask.permute(0, 2, 1)).view(b, -1, h, w)
        rgb_refine = self.gamma1 * rgb_refine + y
    
        inf_k = self.inf_k(y)
        inf_v = self.inf_v(y)
        inf_v = inf_v.view(b, -1, w * h)
        inf_k = inf_k.view(b, -1, w * h).permute(0, 2, 1)
        inf_q = q.view(b, -1, w * h)  
        inf_mask = self.softmax(torch.bmm(inf_k, inf_q))   
        inf_refine = torch.bmm(inf_v, inf_mask.permute(0, 2, 1)).view(b, -1, h, w)     
        inf_refine = self.gamma2 * inf_refine + x 

        return self.second_reduce(torch.cat([rgb_refine, inf_refine], dim=1))
 

class DenseLayer(nn.Module):     
    def __init__(self, in_channels, out_channels, down_factor=4, k=2): 
        super().__init__()  
        mid_channels = out_channels // down_factor    
        self.down = nn.Conv2d(in_channels, mid_channels, 1)  
        self.denseblock = nn.ModuleList(DSConv(mid_channels * i, mid_channels, 3) for i in range(1, k + 1))     
        self.fuse = DSConv(in_channels + mid_channels, out_channels, 3)

    def forward(self, x):
        down_feats = self.down(x)     
        out_feats = []     
        for block in self.denseblock:
            feats = block(torch.cat([*out_feats, down_feats], dim=1))
            out_feats.append(feats)  
        feats = torch.cat((x, feats), dim=1)
        return self.fuse(feats)
    
  
class PSFM(nn.Module):     
    """Profound semantic fusion module.""" 

    def __init__(self, in_channels, out_channels):    
        super().__init__()
        self.rgb_obj = DenseLayer(in_channels[0], out_channels)
        self.inf_obj = DenseLayer(in_channels[1], out_channels)
        self.obj_fuse = GEFM(out_channels * 2, out_channels)     

    def forward(self, data):
        x, y = data
        rgb_sum = self.rgb_obj(x)   
        inf_sum = self.inf_obj(y)    
        return self.obj_fuse(rgb_sum, inf_sum)   
    
if __name__ == '__main__':   
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32
    ouc_channel = 32     
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device)
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device) 

    module = PSFM([channel_1, channel_2], ouc_channel).to(device)
   
    outputs = module([inputs_1, inputs_2])
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)   
     
    print(ORANGE) 
    flops, macs, _ = calculate_flops(model=module,   
                                     args=[[inputs_1, inputs_2]],
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=True)
    print(RESET)     
