import os, sys  
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
  
import warnings
warnings.filterwarnings('ignore')

import torch  
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple   
from functools import partial

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.extra_modules.mlp.SEFN import SEFN 
 
class Scale(nn.Module):     
    """
    Scale vector by element multiplications.
    """   
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()    
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)
    
    def forward(self, x):
        return x * self.scale    
  
class StarReLU(nn.Module):    
    """     
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,     
        scale_learnable=True, bias_learnable=True,  
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)  
        self.scale = nn.Parameter(scale_value * torch.ones(1),   
            requires_grad=scale_learnable)     
        self.bias = nn.Parameter(bias_value * torch.ones(1),   
            requires_grad=bias_learnable) 
    def forward(self, x):    
        return self.scale * self.relu(x)**2 + self.bias
 
class Mlp(nn.Module):    
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.    
    """  
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=StarReLU, drop=0., mlp_ratio=4, bias=False, **kwargs): 
        super().__init__() 
        out_features = out_features or in_features     
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop) 

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0]) 
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=bias)  
        self.drop2 = nn.Dropout(drop_probs[1])
     
    def forward(self, x):  
        x = self.fc1(x)
        x = self.act(x)     
        x = self.drop1(x)
        x = self.fc2(x)    
        x = self.drop2(x)
        return x   

class LayerNorm2d(nn.Module):  
    """2D Layer Normalization module inspired by Detectron2 and ConvNeXt implementations.  

    This class implements layer normalization for 2D feature maps, normalizing across the channel dimension while
    preserving spatial dimensions.
  
    Attributes:     
        weight (nn.Parameter): Learnable scale parameter. 
        bias (nn.Parameter): Learnable bias parameter.
        eps (float): Small constant for numerical stability.  

    References: 
        https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
        https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py  
    """  
    
    def __init__(self, dim: int, eps: float = 1e-6):     
        """Initialize LayerNorm2d with the given parameters.  

        Args:
            dim (int): Number of channels in the input.
            eps (float): Small constant for numerical stability.
        """   
        super().__init__()    
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps     

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        """Perform forward pass for 2D layer normalization.  
  
        Args: 
            x (torch.Tensor): Input tensor.    

        Returns:     
            (torch.Tensor): Normalized output tensor.    
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True) 
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]   

class MetaFormer_Block(nn.Module):
    """
    Implementation of one MetaFormer block.
    """   
    def __init__(self, in_dim, dim,
                 token_mixer=nn.Identity, mlp=Mlp,   
                 norm_layer=LayerNorm2d,     
                 drop_path=0., mlp_ratio=2, 
                 layer_scale_init_value=None, res_scale_init_value=None, selfatt=False  
                 ):

        super().__init__()  

        self.norm1 = norm_layer(dim) 
        if selfatt:
            self.token_mixer = token_mixer(dim)
        else:
            self.token_mixer = token_mixer(dim, dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()    
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()     
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)  
        self.mlp = mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim) 
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()  
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity() 
 
        self.conv1x1 = Conv(in_dim, dim, 1) if in_dim != dim else nn.Identity()
        
    def forward(self, x):
        x = self.conv1x1(x)
        # x size: [B, C, H, W]
        x = self.res_scale1(x) + \
            self.layer_scale1(   
                self.drop_path1(
                    self.token_mixer(self.norm1(x))   
                )
            )
        x = self.res_scale2(x) + \
            self.layer_scale2( 
                self.drop_path2( 
                    self.mlp(self.norm2(x))   
                )   
            )
        return x
  
class MonaOp(nn.Module):
    def __init__(self, in_features): 
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)  
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )    

    def forward(self, x):   
        identity = x 
        conv1_x = self.conv1(x) 
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)
    
        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity
  
        identity = x
    
        x = self.projector(x)

        return identity + x
   
# CVPR2025
# https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_5100_Breaking_Performance_Shackles_of_Full_Fine-Tuning_on_Visual_Recognition_CVPR_2025_paper.pdf     
class Mona(nn.Module):  
    def __init__(self,   
                 in_dim):  
        super().__init__()   
    
        self.project1 = nn.Conv2d(in_dim, 64, 1)    
        self.nonlinear = F.gelu  
        self.project2 = nn.Conv2d(64, in_dim, 1)

        self.dropout = nn.Dropout(p=0.1)

        self.adapter_conv = MonaOp(64) 
     
        self.norm = LayerNorm2d(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim, 1, 1) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim, 1, 1))    

    def forward(self, x, hw_shapes=None):   
        identity = x  
   
        x = self.norm(x) * self.gamma + x * self.gammax
   
        project1 = self.project1(x)  
  
        project1 = self.adapter_conv(project1)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)   
    
        return identity + project2 

class MetaFormer_Mona(nn.Module):     
    """   
    Implementation of one MetaFormer block.
    """
    def __init__(self, in_dim, dim,    
                 token_mixer=nn.Identity, mlp=Mlp,
                 norm_layer=LayerNorm2d,   
                 drop_path=0., mlp_ratio=2,  
                 layer_scale_init_value=None, res_scale_init_value=None, selfatt=False     
                 ):

        super().__init__()
  
        self.norm1 = norm_layer(dim)
        if selfatt:
            self.token_mixer = token_mixer(dim)
        else:
            self.token_mixer = token_mixer(dim, dim) 
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()  
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity() 
        self.mona1 = Mona(dim)

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()     
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()   
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
        self.mona2 = Mona(dim)  
        
        self.conv1x1 = Conv(in_dim, dim, 1) if in_dim != dim else nn.Identity() 
   
    def forward(self, x):  
        x = self.conv1x1(x)    
        # x size: [B, C, H, W]     
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
        x = self.mona1(x)   

        x = self.res_scale2(x) + \
            self.layer_scale2(    
                self.drop_path2(  
                    self.mlp(self.norm2(x))   
                )
            )
        x = self.mona2(x)     

        return x
     
class NCHW2NLC2NCHW(nn.Module):  
    def __init__(self, dim, module): 
        super().__init__() 

        self.module = module(dim)
    
    def forward(self, x):     
        B, C, H, W = x.size()
        x_nlc = x.flatten(2).permute(0, 2, 1) # B C H W -> N L C
        x_nlc = self.module(x_nlc) # 经过对应的需要NLC输入的模块  
        x_nchw = x_nlc.permute(0, 2, 1).view([B, C, H, W]).contiguous() # N L C -> B C H W     
        return x_nchw
 
# WACV2025|ultralytics/nn/extra_modules/mlp/SEFN.py
class MetaFormer_SEFN(nn.Module):
    """
    Implementation of one MetaFormer block.
    """
    def __init__(self, in_dim, dim,
                 token_mixer=nn.Identity, 
                 norm_layer=LayerNorm2d,
                 drop_path=0., mlp_ratio=2,
                 layer_scale_init_value=None, res_scale_init_value=None, selfatt=False
                 ):  
  
        super().__init__()
    
        self.norm1 = norm_layer(dim) 
        if selfatt:  
            self.token_mixer = token_mixer(dim)   
        else:  
            self.token_mixer = token_mixer(dim, dim)     
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()    
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()  
     
        self.norm2 = norm_layer(dim)   
        self.mlp = SEFN(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
        
        self.conv1x1 = Conv(in_dim, dim, 1) if in_dim != dim else nn.Identity() 
        
    def forward(self, x):  
        x = self.conv1x1(x)  
        x_spatial = x  
        # x size: [B, C, H, W]     
        x = self.res_scale1(x) + \
            self.layer_scale1(  
                self.drop_path1(  
                    self.token_mixer(self.norm1(x))   
                )
            )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(     
                    self.mlp(self.norm2(x), x_spatial)
                )
            )
        return x

class MetaFormer_Mona_SEFN(nn.Module):
    """  
    Implementation of one MetaFormer block.  
    """   
    def __init__(self, in_dim, dim,
                 token_mixer=nn.Identity,
                 norm_layer=LayerNorm2d,
                 drop_path=0., mlp_ratio=2, 
                 layer_scale_init_value=None, res_scale_init_value=None, selfatt=False  
                 ):  
    
        super().__init__() 
     
        self.norm1 = norm_layer(dim)   
        if selfatt: 
            self.token_mixer = token_mixer(dim)
        else:
            self.token_mixer = token_mixer(dim, dim)    
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity() 
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
        self.mona1 = Mona(dim)    

        self.norm2 = norm_layer(dim)
        self.mlp = SEFN(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim) 
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()  
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()  
        self.mona2 = Mona(dim)    
 
        self.conv1x1 = Conv(in_dim, dim, 1) if in_dim != dim else nn.Identity()     
        
    def forward(self, x):
        x = self.conv1x1(x)
        x_spatial = x
        # x size: [B, C, H, W]    
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(  
                    self.token_mixer(self.norm1(x))  
                )  
            )
        x = self.mona1(x)     
    
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2( 
                    self.mlp(self.norm2(x), x_spatial)
                )
            ) 
        x = self.mona2(x)

        return x     
