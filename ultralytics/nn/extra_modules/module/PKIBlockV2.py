'''    
本文件由BiliBili：魔傀面具整理
ultralytics/nn/module_images/ECCV2026-PKIBlockV2.png     
ultralytics/nn/module_images/ECCV2026-PKIBlockV2.md   
论文链接：https://arxiv.org/pdf/2603.16341v1     
'''
     
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')  
    
import warnings
warnings.filterwarnings('ignore')
from calflops import calculate_flops
    
import torch    
import torch.nn as nn   
from typing import Any, Dict, Optional, Tuple, Type, Union  

from ultralytics.nn.modules.conv import Conv
  
from ultralytics.plugins.torch_utils import model_fuse_test    

def build_norm_layer(
    cfg: Dict[str, Any],
    num_features: int,
    postfix: Union[int, str] = "",
) -> Tuple[str, nn.Module]:
    """Build a normalization layer from a small config dictionary."""
    if not isinstance(cfg, dict):
        raise TypeError("norm_cfg must be a dict") 
   
    cfg = cfg.copy()
    layer_type = cfg.pop("type", "BN")
    requires_grad = cfg.pop("requires_grad", True)    
   
    if layer_type in {"BN", "BN2d", "BatchNorm2d"}:
        name = f"bn{postfix}"
        layer = nn.BatchNorm2d(num_features, **cfg)
    elif layer_type in {"SyncBN", "SyncBatchNorm"}:     
        name = f"bn{postfix}"    
        layer = nn.SyncBatchNorm(num_features, **cfg)
    elif layer_type in {"GN", "GroupNorm"}: 
        name = f"gn{postfix}"
        num_groups = cfg.pop("num_groups", 32)  
        layer = nn.GroupNorm(num_groups, num_features, **cfg)  
    elif layer_type in {"LN", "LayerNorm"}:   
        name = f"ln{postfix}"
        layer = nn.LayerNorm(num_features, **cfg)  
    else:
        raise KeyError(f"Unsupported normalization type: {layer_type}")
   
    for param in layer.parameters():
        param.requires_grad = requires_grad
    return name, layer   
  
   
def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,    
    training: bool = False,     
    scale_by_keep: bool = True,     
) -> torch.Tensor:
    """Drop residual paths per sample."""
    if drop_prob == 0.0 or not training:
        return x    

    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)  
    if scale_by_keep and keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor

     
class DropPath(nn.Module):  
    """Stochastic depth layer."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = scale_by_keep    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)  
   

def _fuse_bn_tensor(conv: Optional[nn.Conv2d], bn: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse a depthwise convolution and BatchNorm-like affine stats."""
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias    
    eps = bn.eps
    std = (running_var + eps).sqrt()
    scale = (gamma / std).reshape(-1, 1, 1, 1)   

    if conv is None:   
        kernel = torch.ones(
            bn.num_features,   
            1,    
            1,  
            1,
            device=gamma.device,    
            dtype=gamma.dtype,    
        )
    else:   
        kernel = conv.weight

    fused_bias = beta - running_mean * gamma / std 
    return kernel * scale, fused_bias     


class PKSModule(nn.Module):
    """ 
    Poly-Kernel Scope module with train-time branches and deploy-time fusion.   

    In eval mode, switch_to_deploy() fuses five depthwise branches into one
    19x19 depthwise convolution while preserving the surrounding conv0/conv1.  
    """

    def __init__(
        self,
        dim: int,
        deploy: bool = False,
        auto_reparam: bool = False,
        norm_cfg: Optional[Dict[str, Any]] = None,   
        branch_scale: float = 1.0,
    ) -> None: 
        super().__init__()    
        self.deploy = deploy
        self.auto_reparam = auto_reparam
        self.dim = dim   
        self.branch_scale = branch_scale
        self.max_k = 19 

        if norm_cfg is None:
            norm_cfg = {"type": "BN"}   
    
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)   
        self.conv1 = nn.Conv2d(dim, dim, 1)
     
        if deploy:
            self.fused_parallel_conv = nn.Conv2d(    
                dim,
                dim, 
                kernel_size=self.max_k,     
                padding=self.max_k // 2,  
                groups=dim,     
                bias=True, 
            )     
        else:
            k_axial = 19 
            self.branch1_axial = nn.Sequential(
                nn.Conv2d(
                    dim,
                    dim,
                    (1, k_axial),
                    stride=1, 
                    padding=(0, k_axial // 2), 
                    groups=dim,
                    bias=False,  
                ),  
                nn.Conv2d(
                    dim,   
                    dim,  
                    (k_axial, 1),
                    stride=1,
                    padding=(k_axial // 2, 0),
                    groups=dim,
                    bias=False,
                ), 
                build_norm_layer(norm_cfg, dim)[1],
            )
    
            k_b2, d_b2 = 7, 3     
            pad_b2 = (k_b2 - 1) * d_b2 // 2  
            self.branch2_sparse = nn.Sequential(    
                nn.Conv2d(
                    dim,    
                    dim,
                    k_b2,  
                    stride=1,
                    padding=pad_b2,
                    dilation=d_b2,  
                    groups=dim,   
                    bias=False, 
                ), 
                build_norm_layer(norm_cfg, dim)[1],
            )

            k_b3, d_b3 = 5, 3
            pad_b3 = (k_b3 - 1) * d_b3 // 2
            self.branch3_sparse = nn.Sequential(  
                nn.Conv2d(   
                    dim,
                    dim,  
                    k_b3,
                    stride=1,
                    padding=pad_b3,  
                    dilation=d_b3,    
                    groups=dim,    
                    bias=False,
                ),   
                build_norm_layer(norm_cfg, dim)[1], 
            )

            k_b4, d_b4 = 3, 3   
            pad_b4 = (k_b4 - 1) * d_b4 // 2 
            self.branch4_sparse = nn.Sequential(
                nn.Conv2d(
                    dim, 
                    dim,
                    k_b4,
                    stride=1,  
                    padding=pad_b4,   
                    dilation=d_b4,
                    groups=dim,    
                    bias=False,
                ), 
                build_norm_layer(norm_cfg, dim)[1],   
            )
  
            self.branch5_dense = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),    
                build_norm_layer(norm_cfg, dim)[1],
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   
        if self.auto_reparam and not self.training and not self.deploy:    
            self.switch_to_deploy()    
  
        if self.deploy: 
            attn = self.conv0(x)
            attn = self.fused_parallel_conv(attn)   
            attn = self.conv1(attn)
            return x * attn
 
        x_feat = self.conv0(x)
        attn = self.branch1_axial(x_feat)
        attn = attn + self.branch2_sparse(x_feat)  
        attn = attn + self.branch3_sparse(x_feat)   
        attn = attn + self.branch4_sparse(x_feat)
        attn = attn + self.branch5_dense(x_feat)    
        attn = attn * self.branch_scale
        attn = self.conv1(attn)
        return x * attn
  
    def switch_to_deploy(self) -> None:
        if self.deploy:     
            return
   
        with torch.no_grad():
            device = self.branch1_axial[0].weight.device
            dtype = self.branch1_axial[0].weight.dtype    
            center_k = self.max_k // 2
            fused_kernel = torch.zeros(
                self.dim,
                1,   
                self.max_k,
                self.max_k,     
                device=device,   
                dtype=dtype,    
            )
            fused_bias = torch.zeros(self.dim, device=device, dtype=dtype)    

            def fuse_dilated_branch(  
                branch: nn.Sequential,  
                k_size: int,
                dilation: int,    
            ) -> torch.Tensor: 
                branch_kernel, branch_bias = _fuse_bn_tensor(branch[0], branch[1])  
                center_small = k_size // 2 

                for i in range(k_size):
                    for j in range(k_size):
                        h_idx = center_k + (i - center_small) * dilation
                        w_idx = center_k + (j - center_small) * dilation
                        if 0 <= h_idx < self.max_k and 0 <= w_idx < self.max_k:
                            fused_kernel[:, :, h_idx, w_idx] += branch_kernel[:, :, i, j]  

                return branch_bias
    
            k1 = self.branch1_axial[0].weight
            k2, b2 = _fuse_bn_tensor(self.branch1_axial[1], self.branch1_axial[2]) 
            fused_kernel += torch.matmul(k2, k1)
            fused_bias += b2  
    
            fused_bias += fuse_dilated_branch(self.branch2_sparse, k_size=7, dilation=3)
            fused_bias += fuse_dilated_branch(self.branch3_sparse, k_size=5, dilation=3)
            fused_bias += fuse_dilated_branch(self.branch4_sparse, k_size=3, dilation=3)  
            fused_bias += fuse_dilated_branch(self.branch5_dense, k_size=3, dilation=1) 
     
            fused_kernel *= self.branch_scale
            fused_bias *= self.branch_scale 

            self.fused_parallel_conv = nn.Conv2d(
                self.dim,
                self.dim,
                self.max_k,
                padding=self.max_k // 2, 
                groups=self.dim,
                bias=True,
            ).to(device=device, dtype=dtype)
            self.fused_parallel_conv.weight.copy_(fused_kernel)     
            self.fused_parallel_conv.bias.copy_(fused_bias)

        del (
            self.branch1_axial,   
            self.branch2_sparse,
            self.branch3_sparse,    
            self.branch4_sparse, 
            self.branch5_dense,  
        )
        self.deploy = True


class PKSBlock(nn.Module):   
    def __init__(     
        self,
        dim: int,
        deploy: bool = False,  
        auto_reparam: bool = False, 
        norm_cfg: Optional[Dict[str, Any]] = None,
        branch_scale: float = 1.0,   
    ) -> None:   
        super().__init__()  
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()   
        self.spatial_gating_unit = PKSModule(
            dim,
            deploy=deploy,
            auto_reparam=auto_reparam,
            norm_cfg=norm_cfg,    
            branch_scale=branch_scale,
        )
        self.proj_2 = nn.Conv2d(dim, dim, 1)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x.clone()
        x = self.proj_1(x)    
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x) 
        return x + shortcut    
   
    def switch_to_deploy(self) -> None:
        self.spatial_gating_unit.switch_to_deploy() 


class Mlp(nn.Module):
    def __init__(
        self,     
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,     
        act_layer: Type[nn.Module] = nn.GELU,   
        drop: float = 0.0,     
    ) -> None: 
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)    
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,  
            1,
            bias=True,
            groups=hidden_features, 
        ) 
        self.act = act_layer()
        self.drop = nn.Dropout(drop)     
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)  
        self.drop2 = nn.Dropout(drop)
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:   
        x = self.fc1(x)
        x = self.dwconv(x)    
        x = self.act(x)    
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
     

class PKINetV2Block(nn.Module):  
    def __init__(
        self,
        in_dim: int,
        out_dim: int,   
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,    
        act_layer: Type[nn.Module] = nn.GELU,
        auto_reparam: bool = False,
        norm_cfg: Optional[Dict[str, Any]] = None,
        branch_scale: float = 1.0,
        deploy: bool = False, 
    ) -> None:
        super().__init__() 
    
        if norm_cfg is None:  
            norm_cfg = {"type": "BN"} 
 
        self.norm1 = build_norm_layer(norm_cfg, out_dim)[1]
        self.norm2 = build_norm_layer(norm_cfg, out_dim)[1]
        self.attn = PKSBlock(
            out_dim,     
            deploy=deploy,
            auto_reparam=auto_reparam, 
            norm_cfg=norm_cfg,    
            branch_scale=branch_scale, 
        )     

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()   
        mlp_hidden_dim = int(out_dim * mlp_ratio)     
        self.mlp = Mlp(    
            in_features=out_dim,    
            hidden_features=mlp_hidden_dim,   
            act_layer=act_layer,
            drop=drop,    
        )  
  
        self.layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            self.layer_scale_init_value * torch.ones(out_dim),
            requires_grad=True,  
        ) 
        self.layer_scale_2 = nn.Parameter(  
            self.layer_scale_init_value * torch.ones(out_dim),  
            requires_grad=True,  
        ) 

        self.conv = Conv(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()   
  
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.conv(x)   
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x))  
        ) 
        x = x + self.drop_path(  
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
        )    
        return x

    def convert_to_deploy(self) -> None:
        self.attn.switch_to_deploy() 
   

if __name__ == '__main__':    
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m" 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 16, 32, 32, 32    
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = PKINetV2Block(in_channel, out_channel).to(device)     

    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)  
     
    if not torch.cuda.is_available():   
        raise RuntimeError('CUDA is required for half validation.')
    half_inputs = torch.randn((batch_size, in_channel, height, width), device='cuda').half()    
    half_module = PKINetV2Block(in_channel, out_channel).cuda().half()
    half_outputs = half_module(half_inputs)
    print(GREEN + f'half inputs.size:{half_inputs.size()} outputs.size:{half_outputs.size()} dtype:{half_outputs.dtype}' + RESET)

    print(GREEN + 'test reparameterization.' + RESET)     
    module = model_fuse_test(module)     
    outputs = module(inputs)
    print(GREEN + 'test reparameterization done.' + RESET)

    print(ORANGE)
    flops, macs, _ = calculate_flops(model=module,
                                     input_shape=(batch_size, in_channel, height, width), 
                                     output_as_string=True,     
                                     output_precision=4,  
                                     print_detailed=True)
    print(RESET) 
