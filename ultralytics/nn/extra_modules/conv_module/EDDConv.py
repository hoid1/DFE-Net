'''
本文件由BiliBili：魔傀面具整理 
ultralytics/nn/module_images/TGRS2026-EDDConv.md
ultralytics/nn/module_images/TGRS2026-EDDConv.png     
论文链接：https://ieeexplore.ieee.org/document/11500495 
'''

import warnings   
warnings.filterwarnings('ignore')  
try:
    from calflops import calculate_flops
except Exception:      # calflops 只在文件底部的 __main__ 自测里用，缺了不影响建图
    calculate_flops = None
     
import torch  
import torch.nn as nn
from torchvision.ops import deform_conv2d    
 
class DeformableConv2d(nn.Module):   
    # 带mask调制,即每个核的点都被0-1调制
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, dilation=1, groups=1, bias=True, *,    
                 offset_groups=1, with_mask=False     
                 ):    
        super().__init__()   
        assert in_dim % groups == 0, "输入通道数必须能被groups整除"    
        assert out_dim % groups == 0, "输出通道数必须能被groups整除"
        self.stride = stride   
        self.padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim // groups, kernel_size, kernel_size))   
        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))  
        else:
            self.bias = None  
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None

        self.with_mask = with_mask   
        self.param_generator = nn.Conv2d(in_dim, (3 if with_mask else 2) * offset_groups * kernel_size ** 2, 3, 1, 1)
        nn.init.constant_(self.param_generator.weight, 0.0)  # 关键初始化
        nn.init.constant_(self.param_generator.bias, 0.0)  # 初始无偏移   

    def forward(self, x):
        if self.with_mask:
            oh, ow, mask = self.param_generator(x).chunk(3, dim=1)    
            offset = torch.cat([oh, ow], dim=1)
            mask = mask.sigmoid()
        else:   
            offset = self.param_generator(x)   
            mask = None

        x = deform_conv2d( 
            x,
            offset=offset,
            weight=self.weight, 
            bias=self.bias, 
            stride=self.stride,  
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,   
        )   
        return x  


class MCCA(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__() 
        self.reduction = reduction_ratio   
        mid_channels = max(8, in_channels // reduction_ratio)  # 保证最小通道数
    
        # 水平分支（Width方向）    
        self.width_pool = nn.AdaptiveAvgPool2d((1, None))  # [N,C,H,W] -> [N,C,1,W]
        self.width_conv = nn.Sequential(     
            nn.Conv2d(in_channels, mid_channels, 1),   
            nn.BatchNorm2d(mid_channels),  
            nn.ReLU(),   
            nn.Conv2d(mid_channels, in_channels, 1),
            nn.Sigmoid()  
        )
    
        # 垂直分支（Height方向）
        self.height_pool = nn.AdaptiveAvgPool2d((None, 1))  # [N,C,H,W] -> [N,C,H,1]     
        self.height_conv = nn.Sequential(    
            nn.Conv2d(in_channels, mid_channels, 1),   
            nn.BatchNorm2d(mid_channels),   
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_channels, 1),
            nn.Sigmoid()
        )
     
        # 全局分支（Global Average Pooling）  
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_channels, 1),
            nn.Sigmoid()
        )
   
    def forward(self, x):
        # 水平注意力
        w_pool = self.width_pool(x)  # [N,C,1,W]
        w_attn = self.width_conv(w_pool)  # [N,C,1,W]     
 
        # 垂直注意力 
        h_pool = self.height_pool(x)  # [N,C,H,1]  
        h_attn = self.height_conv(h_pool)  # [N,C,H,1]
    
        # 全局注意力 
        g_pool = self.global_pool(x)  # [N,C,1,1]
        g_attn = self.global_conv(g_pool)  # [N,C,1,1]
   
        # 合成三维注意力
        combined_attn = w_attn * h_attn * g_attn

        # 应用注意力权重
        return x * combined_attn


class MCDC(nn.Module):     
    def __init__(self, in_channels, out_channels, kernel_size=3): 
        super().__init__()
        self.kernel_size = kernel_size
   
        # Main convolution parameters 
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))

        # Offset and modulation generators
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size ** 2, kernel_size=3, padding=1)
        self.modulation_conv = nn.Conv2d(in_channels, kernel_size ** 2, kernel_size=3, padding=1)
   
        # MCCA module
        self.mcca = MCCA(in_channels)   

        self.init_weights()  

    def init_weights(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.modulation_conv.weight)  

    def forward(self, x):   
        # Generate MCCA features     
        x_attn = self.mcca(x)
     
        # Generate offsets and modulation
        offsets = self.offset_conv(x_attn)  # [N, 2*k*k, H, W]
        modulation = torch.sigmoid(self.modulation_conv(x_attn))  # [N, k*k, H, W]
   
        # Apply deformable convolution    
        return deform_conv2d(
            input=x,
            offset=offsets,
            weight=self.weight,
            mask=modulation,
            padding=(self.kernel_size // 2, self.kernel_size // 2)   
        )

class EDDConv(nn.Module): 
    def __init__(self, in_ch, out_ch, dc_groups=3, offset_groups=3, kernel_list=[1, 3, 5], with_mask=True):
        super().__init__()
        self.num_convs = len(kernel_list)    
        self.dc_groups = dc_groups
        self.offset_groups = offset_groups
        # 路由网络保持不变
        self.router = nn.Sequential(    
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, self.num_convs, 1),
            nn.Softmax(dim=1)     
        )
        self.with_mask = with_mask 
        # 使用可变形卷积块替代普通卷积
        self.convs = nn.ModuleList([     
            # nn.Conv2d(in_ch, out_ch, k, padding=k // 2)
            # MCDC(in_ch, out_ch, k)     
            MCDC(in_ch, out_ch)  
            # DeformableConv2d(in_dim=in_ch, out_dim=out_ch, kernel_size=k, groups=self.dc_groups,
            #                  offset_groups=self.offset_groups, with_mask=self.with_mask)
            for k in kernel_list  
        ])  

    def forward(self, x):
        B, _, H, W = x.shape
        weights = self.router(x).squeeze(-1).squeeze(-1)  

        output = 0   
        for i, conv in enumerate(self.convs):    
            weight = weights[:, i].view(B, 1, 1, 1)  # 权重调整维度   
            output += weight * conv(x)  # 加权求和     

        return output   

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m" 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32  
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)
   
    module = EDDConv(in_channel, out_channel).to(device).eval()    
 
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
 
    if not torch.cuda.is_available(): 
        raise RuntimeError('CUDA is required for EDDConv half precision validation.')
    half_inputs = torch.randn((batch_size, in_channel, height, width), device='cuda').half()     
    half_outputs = EDDConv(in_channel, out_channel).cuda().eval().half()(half_inputs) 
    print(GREEN + f'half inputs.size:{half_inputs.size()} outputs.size:{half_outputs.size()} dtype:{half_outputs.dtype}' + RESET) 
  
    print(ORANGE)  
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width), 
                                     output_as_string=True,    
                                     output_precision=4,
                                     print_detailed=True)  
    print(RESET)    
