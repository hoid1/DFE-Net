"""  
GoldYOLO neck operators.
   
This file migrates the minimal runtime blocks required by GoldYOLO neck YAMLs.
"""

import numpy as np   
import torch   
import torch.nn as nn     
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv   

from .EfficientRepBiPAN import RepVGGBlock  
    

def onnx_adaptive_avg_pool2d(x, output_size):
    stride_size = np.floor(np.array(x.shape[-2:]) / output_size).astype(np.int32)
    kernel_size = np.array(x.shape[-2:]) - (output_size - 1) * stride_size  
    avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
    return avg(x)  
   
 
def get_avg_pool():
    return onnx_adaptive_avg_pool2d if torch.onnx.is_in_onnx_export() else nn.functional.adaptive_avg_pool2d
  
    
def get_shape(tensor):   
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [v.cpu().numpy() for v in shape]
    return shape


class SimFusion_3in(nn.Module):
    """Three-branch scale-aligned fusion used by GoldYOLO."""    

    def __init__(self, in_channel_list, out_channels):
        super().__init__()     
        self.cv1 = Conv(in_channel_list[0], out_channels, act=nn.ReLU()) if in_channel_list[0] != out_channels else nn.Identity()     
        self.cv2 = Conv(in_channel_list[1], out_channels, act=nn.ReLU()) if in_channel_list[1] != out_channels else nn.Identity()
        self.cv3 = Conv(in_channel_list[2], out_channels, act=nn.ReLU()) if in_channel_list[2] != out_channels else nn.Identity()     
        self.cv_fuse = Conv(out_channels * 3, out_channels, act=nn.ReLU()) 
        self.downsample = nn.functional.adaptive_avg_pool2d

    def forward(self, x):
        _, _, h, w = x[1].shape
        output_size = (h, w)
        if torch.onnx.is_in_onnx_export():  
            self.downsample = onnx_adaptive_avg_pool2d  
            output_size = np.array([h, w])    

        x0 = self.cv1(self.downsample(x[0], output_size))
        x1 = self.cv2(x[1])   
        x2 = self.cv3(F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=False))
        return self.cv_fuse(torch.cat((x0, x1, x2), dim=1))

  
class SimFusion_4in(nn.Module):
    """Four-branch scale-aligned fusion used by GoldYOLO."""

    def __init__(self):
        super().__init__()    
        self.avg_pool = nn.functional.adaptive_avg_pool2d

    def forward(self, x):
        x_l, x_m, x_s, x_n = x
        _, _, h, w = x_s.shape 
        output_size = np.array([h, w])
  
        if torch.onnx.is_in_onnx_export(): 
            self.avg_pool = onnx_adaptive_avg_pool2d

        x_l = self.avg_pool(x_l, output_size)   
        x_m = self.avg_pool(x_m, output_size)
        x_n = F.interpolate(x_n, size=(h, w), mode="bilinear", align_corners=False)   
        return torch.cat([x_l, x_m, x_s, x_n], dim=1) 
  

class IFM(nn.Module):   
    """Information fusion module used by GoldYOLO."""     

    def __init__(self, inc, ouc, embed_dim_p=96, fuse_block_num=3):    
        super().__init__() 
        self.conv = nn.Sequential(
            Conv(inc, embed_dim_p),
            *[RepVGGBlock(embed_dim_p, embed_dim_p) for _ in range(fuse_block_num)],
            Conv(embed_dim_p, sum(ouc)),
        )   

    def forward(self, x):    
        return self.conv(x)   
     
     
class h_sigmoid(nn.Module):  
    def __init__(self, inplace=True):     
        super().__init__()    
        self.relu = nn.ReLU6(inplace=inplace)     

    def forward(self, x): 
        return self.relu(x + 3) / 6     


class InjectionMultiSum_Auto_pool(nn.Module): 
    """Global-to-local feature injection used in GoldYOLO neck."""    
     
    def __init__(self, inp, oup, global_inp, flag):
        super().__init__()
        self.global_inp = global_inp    
        self.flag = flag 
        self.local_embedding = Conv(inp, oup, 1, act=False)
        self.global_embedding = Conv(global_inp[self.flag], oup, 1, act=False)
        self.global_act = Conv(global_inp[self.flag], oup, 1, act=False)
        self.act = h_sigmoid()  

    def forward(self, x):
        x_l, x_g = x
        _, _, h, w = x_l.shape
        _, _, g_h, _ = x_g.shape 
        use_pool = h < g_h
     
        global_info = x_g.split(self.global_inp, dim=1)[self.flag]
        local_feat = self.local_embedding(x_l)    
        global_act = self.global_act(global_info)  
        global_feat = self.global_embedding(global_info)   

        if use_pool:
            avg_pool = get_avg_pool()   
            output_size = np.array([h, w]) 
            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)
        else:
            sig_act = F.interpolate(self.act(global_act), size=(h, w), mode="bilinear", align_corners=False)  
            global_feat = F.interpolate(global_feat, size=(h, w), mode="bilinear", align_corners=False)

        return local_feat * sig_act + global_feat  


class PyramidPoolAgg(nn.Module):     
    """GoldYOLO pyramid pooling aggregation."""
  
    def __init__(self, inc, ouc, stride, pool_mode="torch"):
        super().__init__() 
        self.stride = stride
        self.pool = nn.functional.adaptive_avg_pool2d if pool_mode == "torch" else onnx_adaptive_avg_pool2d   
        self.conv = Conv(inc, ouc)     
 
    def forward(self, inputs):    
        _, _, h, w = get_shape(inputs[-1])   
        h = (h - 1) // self.stride + 1
        w = (w - 1) // self.stride + 1
        output_size = np.array([h, w])   
     
        if not hasattr(self, "pool"):  
            self.pool = nn.functional.adaptive_avg_pool2d 
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_adaptive_avg_pool2d

        out = [self.pool(inp, output_size) for inp in inputs]
        return self.conv(torch.cat(out, dim=1))


def drop_path(x, drop_prob=0.0, training=False):     
    if drop_prob == 0.0 or not training:  
        return x   
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)   
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)   
    random_tensor.floor_()     
    return x.div(keep_prob) * random_tensor
  

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):     
        super().__init__()
        self.drop_prob = drop_prob  
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)    
 
   
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0): 
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv(in_features, hidden_features, act=False)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = nn.ReLU6()  
        self.fc2 = Conv(hidden_features, out_features, act=False)   
        self.drop = nn.Dropout(drop)
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)    
        x = self.act(x)   
        x = self.drop(x)     
        x = self.fc2(x)   
        x = self.drop(x)
        return x


class GOLDYOLO_Attention(nn.Module): 
    def __init__(self, dim, key_dim, num_heads, attn_ratio=4):
        super().__init__()    
        self.num_heads = num_heads    
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads 
        self.d = int(attn_ratio * key_dim)
        self.dh = self.d * num_heads    
  
        self.to_q = Conv(dim, self.nh_kd, 1, act=False)   
        self.to_k = Conv(dim, self.nh_kd, 1, act=False) 
        self.to_v = Conv(dim, self.dh, 1, act=False)     
        self.proj = nn.Sequential(nn.ReLU6(), Conv(self.dh, dim, act=False))     
   
    def forward(self, x): 
        b, _, h, w = get_shape(x)    
        qq = self.to_q(x).reshape(b, self.num_heads, self.key_dim, h * w).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(b, self.num_heads, self.key_dim, h * w)
        vv = self.to_v(x).reshape(b, self.num_heads, self.d, h * w).permute(0, 1, 3, 2)    

        attn = torch.matmul(qq, kk).softmax(dim=-1)  
        out = torch.matmul(attn, vv).permute(0, 1, 3, 2).reshape(b, self.dh, h, w)   
        return self.proj(out)
  

class top_Block(nn.Module):   
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4.0, attn_ratio=2.0, drop=0.0, drop_path=0.0):     
        super().__init__()  
        self.attn = GOLDYOLO_Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity() 
        mlp_hidden_dim = int(dim * mlp_ratio)     
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)    

    def forward(self, x):   
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x
 

class TopBasicLayer(nn.Module):
    """GoldYOLO transformer aggregation layer."""  

    def __init__(  
        self,    
        embedding_dim,
        ouc_list,
        block_num=2,  
        key_dim=8, 
        num_heads=4,
        mlp_ratio=4.0,
        attn_ratio=2.0,    
        drop=0.0, 
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.block_num = block_num
        _ = attn_drop
        self.transformer_blocks = nn.ModuleList(  
            [
                top_Block(
                    embedding_dim, 
                    key_dim=key_dim, 
                    num_heads=num_heads,   
                    mlp_ratio=mlp_ratio,
                    attn_ratio=attn_ratio,
                    drop=drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
                for i in range(self.block_num)  
            ]
        )
        self.conv = nn.Conv2d(embedding_dim, sum(ouc_list), 1)
    
    def forward(self, x):
        for i in range(self.block_num): 
            x = self.transformer_blocks[i](x)
        return self.conv(x)     


class AdvPoolFusion(nn.Module):
    """Two-branch pooling fusion used in GoldYOLO."""
  
    def forward(self, x):  
        x1, x2 = x
        pool = onnx_adaptive_avg_pool2d if torch.onnx.is_in_onnx_export() else nn.functional.adaptive_avg_pool2d   
        _, _, h, w = x2.shape
        x1 = pool(x1, np.array([h, w]))    
        return torch.cat([x1, x2], dim=1)    

    
__all__ = (
    "AdvPoolFusion",
    "IFM",
    "InjectionMultiSum_Auto_pool",     
    "PyramidPoolAgg",
    "SimFusion_3in",
    "SimFusion_4in",
    "TopBasicLayer", 
)  
