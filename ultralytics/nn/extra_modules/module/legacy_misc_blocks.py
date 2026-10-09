'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/attention.py:1067-1115 , nn/extra_modules/block.py:1450-1470 , nn/extra_modules/block.py:1472-1492 , nn/extra_modules/block.py:2754-2770 , nn/extra_modules/block.py:4896-4900 , nn/extra_modules/block.py:5073-5101 , nn/extra_modules/block.py:7078-7108 , nn/extra_modules/block.py:7110-7147 , nn/extra_modules/head.py:246-321 , nn/extra_modules/kan_convs/kagn_conv.py:137-144 , nn/extra_modules/kan_convs/kagn_conv.py:8-124 , nn/extra_modules/moe/experts.py:59-70 , nn/extra_modules/moe/experts.py:72-93 , nn/extra_modules/moe/experts.py:95-116 , nn/extra_modules/moe/loss.py:5-45 , nn/extra_modules/moe/moe_module.py:7-119 , nn/extra_modules/moe/routers.py:131-157 , nn/extra_modules/moe/routers.py:159-177 , nn/extra_modules/moe/routers.py:179-204 , nn/extra_modules/moe/routers.py:97-129 , nn/extra_modules/yolov13.py:543-568
C 组：0526 版独立模块，注册方式见 tasks.py 中 legacy0526 分支。
包含：DGCST, SPPF_LSKA, RCSOSA, ESMoE, RCM, KAGNConv2DLayer, DetectAux, Silence, FullPAD_Tunnel
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

from typing import Dict
from timm.layers import DropPath
import torch.nn.functional as F
from typing import Tuple
from torch.nn.functional import conv2d
from functools import lru_cache
import math
import torch.nn as nn
import torch
from ultralytics.nn.extra_modules.attention.ACA import make_divisible
from ultralytics.nn.extra_modules.module.IDWB import ConvMlp
from ultralytics.nn.extra_modules.module.iRMB import SEAttention
from ultralytics.nn.modules.block import DFL
from ultralytics.nn.modules.conv import Conv, DWConv, RepConv
from ultralytics.utils.tal import dist2bbox, make_anchors


# ---- 原样迁移自 nn/extra_modules/moe/routers.py:97-129 ----
class BaseRouter(nn.Module):
    def __init__(self, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.softmax = nn.Softmax(dim=1)

    def _process_logits(self, logits: torch.Tensor, noise_std: float, training: bool) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Unified logic to process logits into Top-K selection."""
        B = logits.shape[0]
        
        # 1) Add noise during training (simplified Gumbel-Softmax trick)
        if training and noise_std > 0:
            logits = logits + torch.randn_like(logits) * noise_std
        
        # 2) Compute probabilities
        probs = self.softmax(logits)
        
        # 3) Select Top-K
        topk_vals, topk_indices = torch.topk(probs, self.top_k, dim=1)
        
        # 4) Normalize weights
        sum_vals = topk_vals.sum(dim=1, keepdim=True) + 1e-6
        topk_vals = topk_vals / sum_vals
        
        # 5) Collect loss-related info (train only)
        loss_dict = {}
        if training:
            loss_dict['router_logits'] = logits
            loss_dict['router_probs'] = probs
            loss_dict['topk_indices'] = topk_indices
            
        return topk_vals, topk_indices, loss_dict

# ---- 原样迁移自 nn/extra_modules/moe/routers.py:159-177 ----
class AdaptiveRoutingLayer(BaseRouter):
    def __init__(self, in_channels, num_experts, reduction=8, top_k=2, noise_std=1.0):
        super().__init__(num_experts, top_k)
        self.noise_std = noise_std
        reduced_channels = max(in_channels // reduction, 8)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, num_experts, 1, bias=False),
            nn.BatchNorm2d(num_experts)
        )

    def forward(self, x):
        pooled = self.avg_pool(x)
        logits = self.router(pooled).squeeze(-1).squeeze(-1) # [B, E]
        return self._process_logits(logits, self.noise_std, self.training)

# ---- 原样迁移自 nn/extra_modules/block.py:5073-5101 ----
class DGCST(nn.Module):
    # Dynamic Group Convolution Shuffle Transformer
    def __init__(self, c1, c2) -> None:
        super().__init__()
        
        self.c = c2 // 4
        self.gconv = Conv(self.c, self.c, g=self.c)
        self.conv1 = Conv(c1, c2, 1)
        self.conv2 = nn.Sequential(
            Conv(c2, c2, 1),
            Conv(c2, c2, 1)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = torch.split(x, [self.c, x.size(1) - self.c], 1)
        
        x1 = self.gconv(x1)
        
        # shuffle
        b, n, h, w = x1.size()
        b_n = b * n // 2
        y = x1.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        y = torch.cat((y[0], y[1]), 1)
        
        x = torch.cat([y, x2], 1)
        return x + self.conv2(x)

# ---- 原样迁移自 nn/extra_modules/head.py:246-321 ----
class DetectAux(nn.Module):
    """YOLOv8 Detect head with Aux Head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch) // 2  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch[:self.nl])
        self.cv3 = nn.ModuleList(nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            ) for x in ch[:self.nl])
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch[self.nl:])
        self.cv5 = nn.ModuleList(nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            ) for x in ch[self.nl:])
        self.dfl_aux = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            for i in range(self.nl, 2 * self.nl):
                x[i] = torch.cat((self.cv4[i - self.nl](x[i]), self.cv5[i - self.nl](x[i])), 1)
            return x
        elif self.dynamic or self.shape != shape:
            if hasattr(self, 'dfl_aux'):
                for i in range(self.nl, 2 * self.nl):
                    x[i] = torch.cat((self.cv4[i - self.nl](x[i]), self.cv5[i - self.nl](x[i])), 1)
            
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x[:self.nl], self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x[:self.nl]], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x[:self.nl])

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        
        for a, b, s in zip(m.cv4, m.cv5, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
    
    def switch_to_deploy(self):
        del self.cv4, self.cv5, self.dfl_aux

# ---- 原样迁移自 nn/extra_modules/moe/routers.py:131-157 ----
class EfficientSpatialRouter(BaseRouter):
    def __init__(self, in_channels, num_experts, reduction=8, top_k=2, noise_std=1.0, pool_scale=4):
        super().__init__(num_experts, top_k)
        self.noise_std = noise_std
        self.pool_scale = pool_scale
        reduced_channels = max(in_channels // reduction, 8)
        
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, num_experts, 1, bias=False),
            nn.BatchNorm2d(num_experts)  # numerical stability
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Pre-pooling optimization
        if H > self.pool_scale and W > self.pool_scale:
            x_in = F.avg_pool2d(x, kernel_size=self.pool_scale, stride=self.pool_scale)
        else:
            x_in = x
            
        out = self.router(x_in) # [B, E, H', W']
        global_logits = torch.mean(out, dim=[2, 3]) # [B, E]
        
        return self._process_logits(global_logits, self.noise_std, self.training)

# ---- 原样迁移自 nn/extra_modules/moe/experts.py:72-93 ----
class GhostExpert(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)[:, :self.out_channels, :, :]

# ---- 原样迁移自 nn/extra_modules/moe/experts.py:95-116 ----
class InvertedResidualExpert(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, expand_ratio=2):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_expand = expand_ratio != 1
        layers = []
        if self.use_expand:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x): return self.conv(x)

# ---- 原样迁移自 nn/extra_modules/moe/routers.py:179-204 ----
class LocalRoutingLayer(BaseRouter):
    def __init__(self, in_channels, num_experts, reduction=8, top_k=2, noise_std=1.0):
        super().__init__(num_experts, top_k)
        self.noise_std = noise_std
        # Even for local routing, default to 2x downsampling to save FLOPs with minimal texture loss
        self.pool_scale = 2 
        
        reduced_channels = max(in_channels // reduction, 8)
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 3, padding=1, bias=False), 
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, num_experts, 1, bias=False),
            nn.BatchNorm2d(num_experts)
        )
        
    def forward(self, x):
        # Moderate downsampling to accelerate
        if x.shape[2] > self.pool_scale:
            x_in = F.avg_pool2d(x, kernel_size=self.pool_scale, stride=self.pool_scale)
        else:
            x_in = x
            
        out = self.router(x_in) 
        global_logits = torch.mean(out, dim=[2, 3])
        return self._process_logits(global_logits, self.noise_std, self.training)

# ---- 原样迁移自 nn/extra_modules/moe/loss.py:5-45 ----
class MoELoss(nn.Module):
    """
    Computes auxiliary losses for Mixture of Experts (MoE) models, 
    specifically Load Balancing Loss and Z-Loss.
    """
    def __init__(self, balance_loss_coeff=0.01, z_loss_coeff=1e-3, num_experts=4, top_k=2):
        super().__init__()
        self.balance_loss_coeff = balance_loss_coeff
        self.z_loss_coeff = z_loss_coeff
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, router_probs, router_logits, expert_indices):
        """
        Args:
            router_probs (torch.Tensor): Probability distribution predicted by router [B, E]
            router_logits (torch.Tensor): Logits from router [B, E]
            expert_indices (torch.Tensor): Selected expert indices [B, k]
        
        Returns:
            torch.Tensor: Combined auxiliary loss
        """
        # 1) Load Balancing Loss
        # importance: probability distribution predicted by router (differentiable)
        importance = router_probs.mean(dim=0) 
        
        # usage: which experts were actually selected (non-differentiable, detached)
        usage_mask = torch.zeros_like(router_probs)
        # B, E = router_probs.shape
        # usage_mask = torch.zeros(B, E, dtype=router_probs.dtype, device=router_probs.device)
        for k in range(self.top_k):
            usage_mask.scatter_(1, expert_indices[:, k].unsqueeze(1), 1.0)
        usage = usage_mask.mean(dim=0)
        
        balance_loss = self.num_experts * torch.sum(importance * usage.detach())
        
        # 2) Z-Loss (numerical stability)
        # Penalize square of log(sum(exp(logits))) to prevent logits from exploding.
        z_loss = torch.mean(torch.logsumexp(router_logits, dim=1)**2)
        
        return (self.balance_loss_coeff * balance_loss) + (self.z_loss_coeff * z_loss)

# ---- 原样迁移自 nn/extra_modules/moe/experts.py:59-70 ----
class SimpleExpert(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=2):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x): return self.conv(x)

# ---- 原样迁移自 nn/extra_modules/moe/moe_module.py:7-119 ----
class ESMoE(nn.Module):
    """Improved MoE with pluggable routers/experts and a shared expert for stability."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        expert_type: str = 'simple',   # ['simple', 'ghost', 'inverted']
        router_type: str = 'efficient',# ['efficient', 'local', 'adaptive']
        noise_std: float = 1.0,
        balance_loss_coeff: float = 0.01,
        router_z_loss_coeff: float = 1e-3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_loss_coeff = balance_loss_coeff
        self.router_z_loss_coeff = router_z_loss_coeff
        
        # 1) Instantiate Router
        if router_type == 'local':
            self.routing = LocalRoutingLayer(in_channels, num_experts, top_k=top_k, noise_std=noise_std)
        elif router_type == 'adaptive':
            self.routing = AdaptiveRoutingLayer(in_channels, num_experts, top_k=top_k, noise_std=noise_std)
        else:
            self.routing = EfficientSpatialRouter(in_channels, num_experts, top_k=top_k, noise_std=noise_std)

        # 2) Instantiate Experts
        self.experts = nn.ModuleList()
        if expert_type == 'ghost':
            expert_cls = GhostExpert
        elif expert_type == 'inverted':
            expert_cls = InvertedResidualExpert
        else:
            expert_cls = SimpleExpert
            
        for _ in range(num_experts):
            self.experts.append(expert_cls(in_channels, out_channels))
            
        # 3) Shared expert (Always active)
        self.shared_expert = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        self._init_weights()
        self.aux_loss = torch.tensor(0.0)
        self.moe_loss_fn = MoELoss(balance_loss_coeff, router_z_loss_coeff, num_experts, top_k)
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Robust router init: find the last Conv layer to initialize
        # Keep initial expert probabilities nearly uniform
        for m in self.routing.router.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv:
            nn.init.normal_(last_conv.weight, mean=0, std=0.01)
            if last_conv.bias is not None:
                nn.init.constant_(last_conv.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1) Routing (standardized interface)
        # loss_dict contains training loss inputs; empty during inference
        routing_weights, routing_indices, loss_dict = self.routing(x)
        
        # 2) Shared expert compute
        shared_out = self.shared_expert(x)
        
        # 3) Sparse expert compute
        # Initialize outputs with zeros
        expert_output = torch.zeros(B, self.out_channels, H, W, device=x.device, dtype=x.dtype)
        
        indices_flat = routing_indices.view(B, self.top_k)
        weights_flat = routing_weights.view(B, self.top_k)
        
        for i in range(self.num_experts):
            # Find all samples assigned to expert i
            mask = (indices_flat == i)
            if mask.any():
                batch_idx, k_idx = torch.where(mask)
                
                # Select input and compute
                inp = x[batch_idx]
                out = self.experts[i](inp)
                
                # Select weights and broadcast
                w = weights_flat[batch_idx, k_idx].view(-1, 1, 1, 1)
                
                # Accumulate results
                expert_output.index_add_(0, batch_idx, out.to(expert_output.dtype) * w.to(expert_output.dtype))
        
        final_output = shared_out + expert_output

        # 4) Compute and return Loss during training
        if self.training and loss_dict:
             self.aux_loss = self.moe_loss_fn(loss_dict['router_probs'], loss_dict['router_logits'], loss_dict['topk_indices'])
        else:
             pass

        return final_output

# ---- 原样迁移自 nn/extra_modules/yolov13.py:543-568 ----
class FullPAD_Tunnel(nn.Module):
    """
    A gated fusion module for the Full-Pipeline Aggregation-and-Distribution (FullPAD) paradigm.

    This module implements a gated residual connection used to fuse features. It takes two inputs: the original
    feature map and a correlation-enhanced feature map. It then computes `output = original + gate * enhanced`,
    where `gate` is a learnable scalar parameter that adaptively balances the contribution of the enhanced features.

    Methods:
        forward: Performs the gated fusion of two input feature maps.

    Examples:
        >>> import torch
        >>> model = FullPAD_Tunnel()
        >>> original_feature = torch.randn(2, 64, 32, 32)
        >>> enhanced_feature = torch.randn(2, 64, 32, 32)
        >>> output = model([original_feature, enhanced_feature])
        >>> print(output.shape)
        torch.Size([2, 64, 32, 32])
    """
    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.0))
    def forward(self, x):
        out = x[0] + self.gate * x[1]
        return out

# ---- 原样迁移自 nn/extra_modules/kan_convs/kagn_conv.py:8-124 ----
class KAGNConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, conv_w_fun, input_dim, output_dim, degree, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1, dropout: float = 0.0, ndim: int = 2):
        super(KAGNConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.base_activation = nn.SiLU()
        self.conv_w_fun = conv_w_fun
        self.ndim = ndim
        self.dropout = None
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        self.base_conv = nn.ModuleList([conv_class(input_dim // groups,
                                                   output_dim // groups,
                                                   kernel_size,
                                                   stride,
                                                   padding,
                                                   dilation,
                                                   groups=1,
                                                   bias=False) for _ in range(groups)])

        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups) for _ in range(groups)])

        poly_shape = (groups, output_dim // groups, (input_dim // groups) * (degree + 1)) + tuple(
            kernel_size for _ in range(ndim))

        self.poly_weights = nn.Parameter(torch.randn(*poly_shape))
        self.beta_weights = nn.Parameter(torch.zeros(degree + 1, dtype=torch.float32))

        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

        nn.init.kaiming_uniform_(self.poly_weights, nonlinearity='linear')
        nn.init.normal_(
            self.beta_weights,
            mean=0.0,
            std=1.0 / ((kernel_size ** ndim) * self.inputdim * (self.degree + 1.0)),
        )

    def beta(self, n, m):
        return (
                       ((m + n) * (m - n) * n ** 2) / (m ** 2 / (4.0 * n ** 2 - 1.0))
               ) * self.beta_weights[n]

    @lru_cache(maxsize=128)  # Cache to avoid recomputation of Legendre polynomials
    def gram_poly(self, x, degree):
        p0 = x.new_ones(x.size())

        if degree == 0:
            return p0.unsqueeze(-1)

        p1 = x
        grams_basis = [p0, p1]

        for i in range(2, degree + 1):
            p2 = x * p1 - self.beta(i - 1, i) * p0
            grams_basis.append(p2)
            p0, p1 = p1, p2

        return torch.concatenate(grams_basis, dim=1)

    def forward_kag(self, x, group_index):
        # Apply base activation to input and then linear transform with base weights
        basis = self.base_conv[group_index](self.base_activation(x))

        # Normalize x to the range [-1, 1] for stable Legendre polynomial computation
        x = torch.tanh(x).contiguous()

        if self.dropout is not None:
            x = self.dropout(x)

        grams_basis = self.base_activation(self.gram_poly(x, self.degree))
        # Check correct eralization
        # y = einsum(
        #     grams_basis,
        #     self.grams_basis_weights,
        #     "b l d, l o d -> b o",
        # )

        y = self.conv_w_fun(grams_basis, self.poly_weights[group_index],
                            stride=self.stride, dilation=self.dilation,
                            padding=self.padding, groups=1)

        y = self.base_activation(self.layer_norm[group_index](y + basis))

        return y

    def forward(self, x):

        # x = self.base_conv(x)
        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_kag(_x.clone(), group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y

# ---- 原样迁移自 nn/extra_modules/kan_convs/kagn_conv.py:137-144 ----
class KAGNConv2DLayer(KAGNConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm2d):
        super(KAGNConv2DLayer, self).__init__(nn.Conv2d, norm_layer, conv2d,
                                              input_dim, output_dim,
                                              degree, kernel_size,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=2, dropout=dropout)

# ---- 原样迁移自 nn/extra_modules/attention.py:1067-1115 ----
class LSKA(nn.Module):
    # Large-Separable-Kernel-Attention
    # https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention/tree/main
    def __init__(self, dim, k_size=7):
        super().__init__()

        self.k_size = k_size

        if k_size == 7:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,2), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=(2,0), groups=dim, dilation=2)
        elif k_size == 11:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,4), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=(4,0), groups=dim, dilation=2)
        elif k_size == 23:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1,1), padding=(0,9), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1,1), padding=(9,0), groups=dim, dilation=3)
        elif k_size == 35:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1,1), padding=(0,15), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1,1), padding=(15,0), groups=dim, dilation=3)
        elif k_size == 41:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1,1), padding=(0,18), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1,1), padding=(18,0), groups=dim, dilation=3)
        elif k_size == 53:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1,1), padding=(0,24), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1,1), padding=(24,0), groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn

# ---- 原样迁移自 nn/extra_modules/block.py:7078-7108 ----
class RCA(nn.Module):
    def __init__(self, inp, kernel_size=1, ratio=2, band_kernel_size=11, dw_size=(1,1), padding=(0,0), stride=1, square_kernel_size=3, relu=True):
        super(RCA, self).__init__()
        self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size, padding=square_kernel_size//2, groups=inp)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        gc=inp//ratio
        self.excite = nn.Sequential(
                nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc),
                nn.BatchNorm2d(gc),
                nn.ReLU(inplace=True),
                nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc),
                nn.Sigmoid()
            )
    
    def sge(self, x):
        #[N, D, C, 1]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_gather = x_h + x_w #.repeat(1,1,1,x_w.shape[-1])
        ge = self.excite(x_gather) # [N, 1, C, 1]
        
        return ge

    def forward(self, x):
        loc=self.dwconv_hw(x)
        att=self.sge(x)
        out = att*loc
        
        return out

# ---- 原样迁移自 nn/extra_modules/block.py:7110-7147 ----
class RCM(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            token_mixer=RCA,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=2,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,
            dw_size=11,
            square_kernel_size=3,
            ratio=1,
    ):
        super().__init__()
        self.token_mixer = token_mixer(dim, band_kernel_size=dw_size, square_kernel_size=square_kernel_size, ratio=ratio)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:1450-1470 ----
class SR(nn.Module):
    # Shuffle RepVGG
    def __init__(self, c1, c2):
        super().__init__()
        c1_ = int(c1 // 2)
        c2_ = int(c2 // 2)
        self.repconv = RepConv(c1_, c2_, bn=True)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat((x1, self.repconv(x2)), dim=1)
        out = self.channel_shuffle(out, 2)
        return out

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:1472-1492 ----
class RCSOSA(nn.Module):
    # VoVNet with Res Shuffle RepVGG
    def __init__(self, c1, c2, n=1, se=False, g=1, e=0.5):
        super().__init__()
        n_ = n // 2
        c_ = make_divisible(int(c1 * e), 8)
        self.conv1 = RepConv(c1, c_, bn=True)
        self.conv3 = RepConv(int(c_ * 3), c2, bn=True)
        self.sr1 = nn.Sequential(*[SR(c_, c_) for _ in range(n_)])
        self.sr2 = nn.Sequential(*[SR(c_, c_) for _ in range(n_)])

        self.se = None
        if se:
            self.se = SEAttention(c2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.sr1(x1)
        x3 = self.sr2(x2)
        x = torch.cat((x1, x2, x3), 1)
        return self.conv3(x) if self.se is None else self.se(self.conv3(x))

# ---- 原样迁移自 nn/extra_modules/block.py:2754-2770 ----
class SPPF_LSKA(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.lska = LSKA(c_ * 4, k_size=11)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(self.lska(torch.cat((x, y1, y2, self.m(y2)), 1)))

# ---- 原样迁移自 nn/extra_modules/block.py:4896-4900 ----
class Silence(nn.Module):
    def __init__(self):
        super(Silence, self).__init__()
    def forward(self, x):    
        return x

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- SR ----
    try:
        module = SR(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'SR  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'SR  自测跳过: {e}' + RESET)
    # ---- RCSOSA ----
    try:
        module = RCSOSA(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'RCSOSA  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'RCSOSA  自测跳过: {e}' + RESET)
    # ---- SPPF_LSKA ----
    try:
        module = SPPF_LSKA(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'SPPF_LSKA  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'SPPF_LSKA  自测跳过: {e}' + RESET)

