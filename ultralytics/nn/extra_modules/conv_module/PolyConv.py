'''    
本文件由BiliBili：魔傀面具整理 
ultralytics/nn/module_images/ICML2026-PolyConv.png  
ultralytics/nn/module_images/ICML2026-PolyConv.md
论文链接：https://arxiv.org/pdf/2605.20839   
'''

import warnings
warnings.filterwarnings('ignore')   
from calflops import calculate_flops   
  
import torch   
import torch.nn as nn
     
def poly_init(x: torch.Tensor, batch: bool = False):
    if batch:  
        return
    nn.init.kaiming_normal_(x, nonlinearity="relu")

 
class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, affine: bool = True, bias: bool = True):
        super().__init__(num_channels, elementwise_affine=affine, bias=bias) 
        self.bias_tf = bias  

    def forward(self, x):   
        u = x.mean(dim=1, keepdim=True)
        s = ((x * x).mean(dim=1, keepdim=True) - (u * u)).clamp(0)     
        x = (x - u) * torch.rsqrt(s + self.eps)
        if self.bias_tf:
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)     
        else:    
            x = x * self.weight.view(1, -1, 1, 1) 
        return x   
    

class ChannelBatchNorm(nn.Module):
    def __init__(
        self,  
        num_channels: int,
        eps: float = 1e-5, 
        momentum: float = 0.1,
        affine_hw: bool = True,
        affine_c: bool = True,
        bias: bool = True, 
        track_running_stats: bool = True,
    ):
        super().__init__()     
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine_hw = bool(affine_hw)   
        self.affine_c = bool(affine_c)   
        self.bias_tf = bool(bias)    
        self.track_running_stats = bool(track_running_stats)   

        self.gamma_hw = None
        self.beta_hw = None

        if self.affine_c:  
            self.gamma_c = nn.Parameter(torch.ones(self.num_channels))
            self.beta_c = None   
        else:
            self.register_parameter("gamma_c", None)
            self.register_parameter("beta_c", None) 
  
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.empty(0))
            self.register_buffer("running_var", torch.empty(0))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))    
        else:
            self.register_buffer("running_mean", None)    
            self.register_buffer("running_var", None) 
            self.register_buffer("num_batches_tracked", None)     
     
    def _check_input(self, x): 
        if x.dim() != 4:     
            raise ValueError(f"Expected (N,C,H,W), got {tuple(x.shape)}")
        if x.size(1) != self.num_channels:     
            raise ValueError(f"Expected C={self.num_channels}, got C={x.size(1)}")    

    def _maybe_init(self, H, W, device, dtype):    
        if self.track_running_stats:
            if (self.running_mean.numel() == 0) or (self.running_mean.shape[-2:] != (H, W)):
                self.running_mean = torch.zeros((1, 1, H, W), device=device, dtype=dtype)  
                self.running_var = torch.ones((1, 1, H, W), device=device, dtype=dtype)   
                self.num_batches_tracked = torch.tensor(0, device=device, dtype=torch.long)     
     
        if self.affine_hw:   
            need = (self.gamma_hw is None) or (tuple(self.gamma_hw.shape[-2:]) != (H, W))
            if need:
                self.gamma_hw = nn.Parameter(torch.ones((1, 1, H, W), device=device, dtype=dtype))
                self.beta_hw = ( 
                    nn.Parameter(torch.zeros((1, 1, H, W), device=device, dtype=dtype))
                    if self.bias_tf
                    else None
                )     

    def forward(self, x):
        self._check_input(x)
        _, C, H, W = x.shape    
        self._maybe_init(H, W, x.device, x.dtype)     

        if self.training:
            mean = x.mean(dim=(0, 1), keepdim=True)
            ex2 = (x * x).mean(dim=(0, 1), keepdim=True)
            var = (ex2 - mean * mean).clamp(min=0.0)     
 
            if self.track_running_stats:
                with torch.no_grad():     
                    self.num_batches_tracked += 1    
                    m = self.momentum    
                    self.running_mean.mul_(1 - m).add_(m * mean)
                    self.running_var.mul_(1 - m).add_(m * var)
        else:
            if self.track_running_stats and self.running_mean is not None and self.running_mean.numel() != 0:
                mean, var = self.running_mean, self.running_var    
            else:     
                mean = x.mean(dim=(0, 1), keepdim=True)   
                ex2 = (x * x).mean(dim=(0, 1), keepdim=True)
                var = (ex2 - mean * mean).clamp(min=0.0)    
 
        y = (x - mean) * torch.rsqrt(var + self.eps)     
     
        if self.affine_hw:   
            y = y * self.gamma_hw
            if self.bias_tf and (self.beta_hw is not None):  
                y = y + self.beta_hw

        if self.affine_c:
            y = y * self.gamma_c.view(1, C, 1, 1)
            if self.bias_tf and (self.beta_c is not None): 
                y = y + self.beta_c.view(1, C, 1, 1)   

        return y

     
def make_norm(norm_type, num_channels, bias=False): 
    if norm_type == "bn":
        return ChannelBatchNorm(num_channels, bias=bias)     
    if norm_type == "ln":
        return LayerNorm2d(num_channels, bias=bias)
    raise ValueError(f"Unsupported norm_type: {norm_type}")  
     

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):    
        super().__init__()
        self.drop_prob = float(drop_prob)  
     
    def forward(self, x): 
        if self.drop_prob == 0.0 or not self.training:    
            return x  
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob) 
        return x / keep_prob * random_tensor     
     
  
class PolyConv(nn.Module):
    def __init__(
        self,
        C: int,    
        Out_C: int,
        expansion: float = 1.0,
        stage: int = 1,
        norm: str = "ln",
        drop_path: float = 0.0,
        bias: bool = False,
        channel_flip: bool = True,
    ):    
        super().__init__()
        self.C = int(C)     
        self.out_C = int(Out_C)   
        self.C_inner = int(self.C * expansion)    
        self.channel_flip = bool(channel_flip)     
    
        self.proj_in = nn.Conv2d(self.C, self.C_inner, kernel_size=1, padding=0, bias=bias)
        if stage == 0:
            self.coarse = nn.Conv2d(
                self.C_inner,  
                self.C_inner, 
                kernel_size=3,     
                stride=1,   
                padding=2,    
                dilation=2,
                groups=self.C_inner,  
                bias=bias,     
            )   
        else:
            self.coarse = nn.Conv2d(
                self.C_inner,
                self.C_inner,
                kernel_size=5,
                stride=1,   
                padding=4,     
                dilation=2,  
                groups=self.C_inner,     
                bias=bias,    
            )
        self.fine = nn.Conv2d(
            self.C_inner,
            self.C_inner,
            kernel_size=3,    
            stride=1,
            padding=1,  
            groups=self.C_inner, 
            bias=bias,
        )
        self.out = nn.Sequential(
            nn.Conv2d(  
                self.C_inner,   
                self.C_inner,
                kernel_size=3,
                stride=1, 
                padding=1,
                groups=self.C_inner, 
                bias=bias,     
            ),
            nn.Conv2d(self.C_inner, self.out_C, kernel_size=1, padding=0, bias=bias),  
            make_norm(norm, self.out_C, bias=False),
            DropPath(drop_path),    
        )  
     
        poly_init(self.proj_in.weight)
        poly_init(self.coarse.weight)   
        poly_init(self.fine.weight)  
        poly_init(self.out[0].weight) 
        poly_init(self.out[1].weight)

    def forward(self, x):
        b = self.proj_in(x)    
        fine = self.fine(b)   
        if self.channel_flip: 
            fine = fine.flip(dims=[1])    
        return self.out(self.coarse(b) * fine)     

   
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)    

    module = PolyConv(in_channel, out_channel).to(device)

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,     
                                     input_shape=(batch_size, in_channel, height, width),
                                     output_as_string=True,     
                                     output_precision=4,    
                                     print_detailed=True)  
    print(RESET)
