'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/backbone/inceptionnext.py:88-123 , nn/backbone/rmt.py:102-176 , nn/backbone/rmt.py:178-233 , nn/backbone/rmt.py:20-33 , nn/backbone/rmt.py:235-278 , nn/backbone/rmt.py:280-314 , nn/backbone/rmt.py:36-100 , nn/extra_modules/ConverseNet.py:178-217 , nn/extra_modules/EVSSM.py:270-361 , nn/extra_modules/LWGANet.py:273-330 , nn/extra_modules/MaIR.py:520-584 , nn/extra_modules/MobileUViT.py:112-156 , nn/extra_modules/MobileUViT.py:159-177 , nn/extra_modules/MobileUViT.py:180-202 , nn/extra_modules/MobileUViT.py:204-220 , nn/extra_modules/MobileUViT.py:93-109 , nn/extra_modules/UniConvNet.py:120-188 , nn/extra_modules/UniConvNet.py:191-229 , nn/extra_modules/UniConvNet.py:62-92 , nn/extra_modules/block.py:10496-10522 , nn/extra_modules/block.py:10524-10550 , nn/extra_modules/block.py:10552-10556 , nn/extra_modules/block.py:10558-10561 , nn/extra_modules/block.py:10563-10567 , nn/extra_modules/block.py:10569-10572 , nn/extra_modules/block.py:10596-10600 , nn/extra_modules/block.py:10602-10605 , nn/extra_modules/block.py:10879-10906 , nn/extra_modules/block.py:10908-10912 , nn/extra_modules/block.py:10914-10917 , nn/extra_modules/block.py:10924-10928 , nn/extra_modules/block.py:10930-10933 , nn/extra_modules/block.py:11146-11159 , nn/extra_modules/block.py:11161-11165 , nn/extra_modules/block.py:11167-11170 , nn/extra_modules/block.py:11270-11318 , nn/extra_modules/block.py:11320-11324 , nn/extra_modules/block.py:11326-11329 , nn/extra_modules/block.py:11635-11683 , nn/extra_modules/block.py:11699-11746 , nn/extra_modules/block.py:11748-11752 , nn/extra_modules/block.py:11754-11757 , nn/extra_modules/block.py:12501-12507 , nn/extra_modules/block.py:12509-12514 , nn/extra_modules/block.py:13237-13270 , nn/extra_modules/block.py:13272-13276 , nn/extra_modules/block.py:13278-13281 , nn/extra_modules/block.py:13371-13375 , nn/extra_modules/block.py:13377-13380 , nn/extra_modules/block.py:13566-13570 , nn/extra_modules/block.py:13572-13575 , nn/extra_modules/block.py:13811-13815 , nn/extra_modules/block.py:13817-13820 , nn/extra_modules/block.py:13903-13907 , nn/extra_modules/block.py:13909-13912 , nn/extra_modules/block.py:15269-15273 , nn/extra_modules/block.py:15275-15278 , nn/extra_modules/block.py:15284-15288 , nn/extra_modules/block.py:15290-15293 , nn/extra_modules/block.py:15299-15303 , nn/extra_modules/block.py:15305-15308 , nn/extra_modules/block.py:15374-15378 , nn/extra_modules/block.py:15380-15383 , nn/extra_modules/block.py:15389-15393 , nn/extra_modules/block.py:15395-15398 , nn/extra_modules/block.py:15938-15971 , nn/extra_modules/block.py:15973-15977 , nn/extra_modules/block.py:15979-15982 , nn/extra_modules/block.py:1753-1763 , nn/extra_modules/block.py:1765-1769 , nn/extra_modules/block.py:1771-1774 , nn/extra_modules/block.py:1780-1791 , nn/extra_modules/block.py:1793-1797 , nn/extra_modules/block.py:1799-1802 , nn/extra_modules/block.py:3727-3748 , nn/extra_modules/block.py:3750-3756 , nn/extra_modules/block.py:3758-3777 , nn/extra_modules/block.py:4659-4759 , nn/extra_modules/block.py:4761-4771 , nn/extra_modules/block.py:4791-4802 , nn/extra_modules/block.py:4956-4971 , nn/extra_modules/block.py:4990-4994 , nn/extra_modules/block.py:4996-4999 , nn/extra_modules/block.py:5027-5034 , nn/extra_modules/block.py:5058-5062 , nn/extra_modules/block.py:5064-5067 , nn/extra_modules/block.py:5107-5127 , nn/extra_modules/block.py:5129-5151 , nn/extra_modules/block.py:5167-5180 , nn/extra_modules/block.py:5182-5216 , nn/extra_modules/block.py:5218-5222 , nn/extra_modules/block.py:5224-5227 , nn/extra_modules/block.py:5229-5248 , nn/extra_modules/block.py:5310-5314 , nn/extra_modules/block.py:5316-5319 , nn/extra_modules/block.py:6103-6114 , nn/extra_modules/block.py:6116-6121 , nn/extra_modules/block.py:6123-6127 , nn/extra_modules/block.py:6129-6132 , nn/extra_modules/block.py:6296-6314 , nn/extra_modules/block.py:6428-6497 , nn/extra_modules/block.py:6499-6503 , nn/extra_modules/block.py:6505-6508 , nn/extra_modules/block.py:7460-7499 , nn/extra_modules/block.py:7501-7505 , nn/extra_modules/block.py:7507-7510 , nn/extra_modules/block.py:7516-7532 , nn/extra_modules/block.py:7534-7546 , nn/extra_modules/block.py:7548-7558 , nn/extra_modules/block.py:7560-7575 , nn/extra_modules/block.py:7577-7603 , nn/extra_modules/block.py:7606-7626 , nn/extra_modules/block.py:7633-7637 , nn/extra_modules/block.py:7639-7642 , nn/extra_modules/block.py:7950-8017 , nn/extra_modules/block.py:8020-8087 , nn/extra_modules/block.py:8090-8159 , nn/extra_modules/block.py:8162-8236 , nn/extra_modules/block.py:8238-8242 , nn/extra_modules/block.py:8244-8247 , nn/extra_modules/block.py:8261-8270 , nn/extra_modules/block.py:8348-8356 , nn/extra_modules/block.py:8368-8372 , nn/extra_modules/block.py:8374-8377 , nn/extra_modules/block.py:8436-8549 , nn/extra_modules/block.py:8551-8571 , nn/extra_modules/block.py:8573-8603 , nn/extra_modules/block.py:8605-8622 , nn/extra_modules/block.py:8644-8648 , nn/extra_modules/block.py:8650-8653 , nn/extra_modules/block.py:8707-8713 , nn/extra_modules/block.py:8715-8720 , nn/extra_modules/block.py:8722-8728 , nn/extra_modules/block.py:8730-8735 , nn/extra_modules/block.py:8737-8743 , nn/extra_modules/block.py:8745-8750 , nn/extra_modules/block.py:8752-8758 , nn/extra_modules/block.py:8760-8765 , nn/extra_modules/block.py:8767-8773 , nn/extra_modules/block.py:8775-8780 , nn/extra_modules/block.py:901-955 , nn/extra_modules/block.py:9103-9146 , nn/extra_modules/block.py:9149-9167 , nn/extra_modules/block.py:9170-9207 , nn/extra_modules/block.py:9210-9233 , nn/extra_modules/block.py:9236-9278 , nn/extra_modules/block.py:9281-9309 , nn/extra_modules/block.py:9311-9315 , nn/extra_modules/block.py:9317-9320 , nn/extra_modules/block.py:9424-9454 , nn/extra_modules/block.py:9456-9500 , nn/extra_modules/block.py:9503-9515 , nn/extra_modules/block.py:9518-9553 , nn/extra_modules/block.py:9556-9568 , nn/extra_modules/block.py:957-961 , nn/extra_modules/block.py:9570-9574 , nn/extra_modules/block.py:9577-9580 , nn/extra_modules/block.py:9602-9660 , nn/extra_modules/block.py:963-966 , nn/extra_modules/block.py:9662-9666 , nn/extra_modules/block.py:9668-9671 , nn/extra_modules/block.py:9680-9691 , nn/extra_modules/block.py:9693-9722 , nn/extra_modules/block.py:9724-9728 , nn/extra_modules/block.py:9730-9733 , nn/extra_modules/dsan.py:100-137 , nn/extra_modules/hcfnet.py:22-62 , nn/extra_modules/hcfnet.py:64-81 , nn/extra_modules/hcfnet.py:84-114 , nn/extra_modules/hcfnet.py:9-20 , nn/extra_modules/metaformer.py:157-171 , nn/extra_modules/metaformer.py:174-200 , nn/extra_modules/metaformer.py:203-217 , nn/extra_modules/metaformer.py:276-319 , nn/extra_modules/metaformer.py:49-86 , nn/extra_modules/metaformer.py:89-100 , nn/extra_modules/orepa.py:21-305 , nn/extra_modules/orepa.py:424-571 , nn/extra_modules/orepa.py:573-702 , nn/extra_modules/semnet.py:118-189 , nn/extra_modules/semnet.py:191-198 , nn/extra_modules/semnet.py:201-222 , nn/extra_modules/srconvnet.py:220-241 , nn/extra_modules/srconvnet.py:244-254 , nn/extra_modules/srconvnet.py:88-125 , nn/extra_modules/tsdn.py:108-139 , nn/extra_modules/tsdn.py:142-190 , nn/extra_modules/tsdn.py:209-230
A2 组：0526 版 CSP 包装壳 + 其内核（内核在 2107 版中不存在，一并迁入）。
壳签名 (c1, c2, n=1, c3k=False, ...) 与 base_modules + repeat_modules 契约一致，旧 yaml 可一字不改。
本组包含：C3k2_AdditiveBlock, C3k2_CaFormer, C3k2_ConvFormer, C3k2_ConverseB, C3k2_DSAN, C3k2_DTAB, C3k2_DynamicFilter, C3k2_ETB, C3k2_EVS, C3k2_FCA_CTA, C3k2_FCA_SFA, C3k2_FDT, C3k2_FMABlock, C3k2_Faster_EMA, C3k2_GhostDynamicConv, C3k2_HDRAB, C3k2_HFERB, C3k2_Heat, C3k2_IDWB, C3k2_IdentityFormer, C3k2_KAN, C3k2_Kat, C3k2_LFEM, C3k2_LGLB, C3k2_LWGA, C3k2_MFEB, C3k2_MSM, C3k2_MogaBlock, C3k2_OREPA, C3k2_PKIModule, C3k2_PPA, C3k2_PoolingFormer, C3k2_RAB, C3k2_REPVGGOREPA, C3k2_RMB, C3k2_RVB_EMA, C3k2_RandomMixing, C3k2_RetBlock, C3k2_SBSM, C3k2_SFHF, C3k2_SHSA, C3k2_SMAFB, C3k2_UniConvBlock, C3k2_gConv, CSP_EDLAN, RepNCSPELAN4_CAA
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

from typing import Callable
from timm.layers import DropPath
import torch.nn.functional as F
from ultralytics.nn.extra_modules.kan_convs import FastKANConv2DLayer
from torch.jit import Final
from ultralytics.nn.extra_modules.kan_convs import KACNConv2DLayer
from ultralytics.nn.extra_modules.kan_convs import KAGNConv2DLayer
from ultralytics.nn.extra_modules.kan_convs import KALNConv2DLayer
from ultralytics.nn.extra_modules.kan_convs import KANConv2DLayer
from typing import List
from typing import Optional
from torch import Tensor
from typing import Tuple
import torch.utils.checkpoint as checkpoint
import torch.nn.init as init
import math
import torch.nn as nn
import numpy as np
try:
    from ultralytics.nn.extra_modules.ops_dcnv3 import modules as opsm
except Exception:
    # 未编译 compile_module/ops_dcnv3 时先占位为 None。
    # 只有真的构建 UniConvBlock(C3k2_UniConvBlock)才会用到,不影响其余模块导入。
    opsm = None
from functools import partial
from einops import rearrange
import torch
from timm.layers import trunc_normal_
from timm.layers import use_fused_attn
from ultralytics.nn.extra_modules.attention.ACA import make_divisible
from ultralytics.nn.extra_modules.attention.ema import EMA
from ultralytics.nn.extra_modules.block.MetaFormer import StarReLU
from ultralytics.nn.extra_modules.conv_module.Converse2D import Converse2D
from ultralytics.nn.extra_modules.conv_module.DSA import DSA
from ultralytics.nn.extra_modules.conv_module.DynamicConv import DynamicConv
from ultralytics.nn.extra_modules.conv_module.wdbb import transI_fusebn, transVI_multiscale
from ultralytics.nn.extra_modules.featurefusion.LCA import CAB
from ultralytics.nn.extra_modules.head.LSPCD import Scale
from ultralytics.nn.extra_modules.mamba.GLSS2D import ConvBN
from ultralytics.nn.extra_modules.mamba.GLVSS import Mamba
from ultralytics.nn.extra_modules.mamba.MaIR import VMM, mair_ids_generate, mair_shift_ids_generate
from ultralytics.nn.extra_modules.mamba.PatchMamba import FeedForward
from ultralytics.nn.extra_modules.mamba.TinyViM import Conv2d_BN, SS2D
from ultralytics.nn.extra_modules.mamba.TransMixer import Mlp
from ultralytics.nn.extra_modules.module.elgca import LayerNorm
from ultralytics.nn.extra_modules.mlp.DFFN import DFFN
from ultralytics.nn.extra_modules.mlp.DIFF import ElementScale
from ultralytics.nn.extra_modules.mlp.DML import DyConv
from ultralytics.nn.extra_modules.mlp.EDFFN import EDFFN
from ultralytics.nn.extra_modules.mlp.KAN import KAN
from ultralytics.nn.extra_modules.mlp.SEFN import SEFN
from ultralytics.nn.extra_modules.module.CSSC import ChannelPool
from ultralytics.nn.extra_modules.module.DynamicFilter import DynamicFilter
from ultralytics.nn.extra_modules.module.HFRB import FourierUnit
from ultralytics.nn.extra_modules.module.IDWB import ConvMlp, InceptionDWConv2d
from ultralytics.nn.extra_modules.module.LEGBlock import LFEA, Scharr
from ultralytics.nn.extra_modules.module.LWGA import D_GA, GA, GA12, LA, MRA, PA
from ultralytics.nn.extra_modules.module.PKIBlock import CAA
from ultralytics.nn.extra_modules.module.PKIBlockV2 import build_norm_layer
from ultralytics.nn.extra_modules.module.RCB import LayerScale
from ultralytics.nn.extra_modules.module.RepViTBlock import RepViTBlock
from ultralytics.nn.extra_modules.module.fasterblock import Partial_conv3
from ultralytics.nn.extra_modules.module.iRMB import SEAttention
from ultralytics.nn.extra_modules.module.mambaout import LayerNormGeneral
from ultralytics.nn.extra_modules.module.vHeat import Heat2D
from ultralytics.nn.extra_modules.stem.LoG import Gaussian
from ultralytics.nn.extra_modules.transformer.CTA import Channel_Transposed_Attention
from ultralytics.nn.extra_modules.transformer.DilatedGCSA import FixedPosEmb, RelPosEmb
from ultralytics.nn.extra_modules.transformer.FSSA import Attention_F, Attention_S
from ultralytics.nn.extra_modules.transformer.GSA import GSA
from ultralytics.nn.extra_modules.transformer.RSA import RSA
from ultralytics.nn.extra_modules.transformer.SFA import Spatial_Frequency_Attention
from ultralytics.nn.extra_modules.transformer.SHSA import SHSA
from ultralytics.nn.modules.block import Bottleneck, C3k, C3k2, RepVGGDW

class Residual(nn.Module):
    """通用残差包装（迁移自旧版 nn/extra_modules/block.py，勿与 ultralytics.nn.modules.block.Residual 混淆）。"""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

from ultralytics.nn.modules.conv import Conv, autopad
from ultralytics.nn.modules.transformer import LayerNorm2d
from ultralytics.utils.torch_utils import autocast


# ---- 原样迁移自 nn/extra_modules/block.py:7548-7558 ----
class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

# ---- 原样迁移自 nn/extra_modules/block.py:7534-7546 ----
class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

# ---- 原样迁移自 nn/extra_modules/block.py:7577-7603 ----
class AdditiveTokenMixer(nn.Module):
    """
    改变了proj函数的输入，不对q+k卷积，而是对融合之后的结果proj
    """
    def __init__(self, dim=512, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out

# ---- 原样迁移自 nn/extra_modules/block.py:7560-7575 ----
class LocalIntegration(nn.Module):
    """
    """
    def __init__(self, dim, ratio=1, act_layer=nn.ReLU, norm_layer=nn.GELU):
        super().__init__()
        mid_dim = round(ratio * dim)
        self.network = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1, 1, 0),
            norm_layer(mid_dim),
            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1, groups=mid_dim),
            act_layer(),
            nn.Conv2d(mid_dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.network(x)

# ---- 原样迁移自 nn/extra_modules/block.py:7516-7532 ----
class Mlp_CASVIT(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:7606-7626 ----
class AdditiveBlock(nn.Module):
    """
    """
    def __init__(self, dim, mlp_ratio=4., attn_bias=False, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.local_perception = LocalIntegration(dim, ratio=1, act_layer=act_layer, norm_layer=norm_layer)
        self.norm1 = norm_layer(dim)
        self.attn = AdditiveTokenMixer(dim, attn_bias=attn_bias, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_CASVIT(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.local_perception(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:6103-6114 ----
def choose_kan(name, c1, c2, k):
    if name == 'FastKANConv2DLayer':
        kan = FastKANConv2DLayer(c1, c2, kernel_size=k, padding=k // 2)
    elif name == 'KANConv2DLayer':
        kan = KANConv2DLayer(c1, c2, kernel_size=k, padding=k // 2)
    elif name == 'KALNConv2DLayer':
        kan = KALNConv2DLayer(c1, c2, kernel_size=k, padding=k // 2)
    elif name == 'KACNConv2DLayer':
        kan = KACNConv2DLayer(c1, c2, kernel_size=k, padding=k // 2)
    elif name == 'KAGNConv2DLayer':
        kan = KAGNConv2DLayer(c1, c2, kernel_size=k, padding=k // 2)
    return kan

# ---- 原样迁移自 nn/extra_modules/block.py:6116-6121 ----
class Bottleneck_KAN(Bottleneck):
    def __init__(self, c1, c2, kan_mothed, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = choose_kan(kan_mothed, c1, c_, k[0])
        self.cv2 = choose_kan(kan_mothed, c_, c2, k[1])

# ---- 原样迁移自 nn/extra_modules/orepa.py:21-305 ----
class OREPA(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=None,
                 groups=1,
                 dilation=1,
                 act=True,
                 internal_channels_1x1_3x3=None,
                 deploy=False,
                 single_init=False, 
                 weight_only=False,
                 init_hyper_para=1.0, init_hyper_gamma=1.0):
        super(OREPA, self).__init__()
        self.deploy = deploy

        self.nonlinear = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.weight_only = weight_only
        
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.stride = stride
        padding = autopad(kernel_size, padding, dilation)
        self.padding = padding
        self.dilation = dilation

        if deploy:
            self.orepa_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)

        else:

            self.branch_counter = 0

            self.weight_orepa_origin = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), kernel_size, kernel_size))
            init.kaiming_uniform_(self.weight_orepa_origin, a=math.sqrt(0.0))
            self.branch_counter += 1

            self.weight_orepa_avg_conv = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), 1,
                            1))
            self.weight_orepa_pfir_conv = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), 1,
                            1))
            init.kaiming_uniform_(self.weight_orepa_avg_conv, a=0.0)
            init.kaiming_uniform_(self.weight_orepa_pfir_conv, a=0.0)
            self.register_buffer(
                'weight_orepa_avg_avg',
                torch.ones(kernel_size,
                        kernel_size).mul(1.0 / kernel_size / kernel_size))
            self.branch_counter += 1
            self.branch_counter += 1

            self.weight_orepa_1x1 = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), 1,
                            1))
            init.kaiming_uniform_(self.weight_orepa_1x1, a=0.0)
            self.branch_counter += 1

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups <= 4 else 2 * in_channels

            if internal_channels_1x1_3x3 == in_channels:
                self.weight_orepa_1x1_kxk_idconv1 = nn.Parameter(
                    torch.zeros(in_channels, int(in_channels / self.groups), 1, 1))
                id_value = np.zeros(
                    (in_channels, int(in_channels / self.groups), 1, 1))
                for i in range(in_channels):
                    id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
                id_tensor = torch.from_numpy(id_value).type_as(
                    self.weight_orepa_1x1_kxk_idconv1)
                self.register_buffer('id_tensor', id_tensor)

            else:
                self.weight_orepa_1x1_kxk_idconv1 = nn.Parameter(
                    torch.zeros(internal_channels_1x1_3x3,
                                int(in_channels / self.groups), 1, 1))
                id_value = np.zeros(
                    (internal_channels_1x1_3x3, int(in_channels / self.groups), 1, 1))
                for i in range(internal_channels_1x1_3x3):
                    id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
                id_tensor = torch.from_numpy(id_value).type_as(
                    self.weight_orepa_1x1_kxk_idconv1)
                self.register_buffer('id_tensor', id_tensor)
                #init.kaiming_uniform_(
                    #self.weight_orepa_1x1_kxk_conv1, a=math.sqrt(0.0))
            self.weight_orepa_1x1_kxk_conv2 = nn.Parameter(
                torch.Tensor(out_channels,
                            int(internal_channels_1x1_3x3 / self.groups),
                            kernel_size, kernel_size))
            init.kaiming_uniform_(self.weight_orepa_1x1_kxk_conv2, a=math.sqrt(0.0))
            self.branch_counter += 1

            expand_ratio = 8
            self.weight_orepa_gconv_dw = nn.Parameter(
                torch.Tensor(in_channels * expand_ratio, 1, kernel_size,
                            kernel_size))
            self.weight_orepa_gconv_pw = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels * expand_ratio / self.groups), 1, 1))
            init.kaiming_uniform_(self.weight_orepa_gconv_dw, a=math.sqrt(0.0))
            init.kaiming_uniform_(self.weight_orepa_gconv_pw, a=math.sqrt(0.0))
            self.branch_counter += 1

            self.vector = nn.Parameter(torch.Tensor(self.branch_counter, self.out_channels))
            if weight_only is False:
                self.bn = nn.BatchNorm2d(self.out_channels)

            self.fre_init()

            init.constant_(self.vector[0, :], 0.25 * math.sqrt(init_hyper_gamma))  #origin
            init.constant_(self.vector[1, :], 0.25 * math.sqrt(init_hyper_gamma))  #avg
            init.constant_(self.vector[2, :], 0.0 * math.sqrt(init_hyper_gamma))  #prior
            init.constant_(self.vector[3, :], 0.5 * math.sqrt(init_hyper_gamma))  #1x1_kxk
            init.constant_(self.vector[4, :], 1.0 * math.sqrt(init_hyper_gamma))  #1x1
            init.constant_(self.vector[5, :], 0.5 * math.sqrt(init_hyper_gamma))  #dws_conv

            self.weight_orepa_1x1.data = self.weight_orepa_1x1.mul(init_hyper_para)
            self.weight_orepa_origin.data = self.weight_orepa_origin.mul(init_hyper_para)
            self.weight_orepa_1x1_kxk_conv2.data = self.weight_orepa_1x1_kxk_conv2.mul(init_hyper_para)
            self.weight_orepa_avg_conv.data = self.weight_orepa_avg_conv.mul(init_hyper_para)
            self.weight_orepa_pfir_conv.data = self.weight_orepa_pfir_conv.mul(init_hyper_para)

            self.weight_orepa_gconv_dw.data = self.weight_orepa_gconv_dw.mul(math.sqrt(init_hyper_para))
            self.weight_orepa_gconv_pw.data = self.weight_orepa_gconv_pw.mul(math.sqrt(init_hyper_para))

            if single_init:
                #   Initialize the vector.weight of origin as 1 and others as 0. This is not the default setting.
                self.single_init()  

    def fre_init(self):
        prior_tensor = torch.Tensor(self.out_channels, self.kernel_size,
                                    self.kernel_size)
        half_fg = self.out_channels / 2
        for i in range(self.out_channels):
            for h in range(3):
                for w in range(3):
                    if i < half_fg:
                        prior_tensor[i, h, w] = math.cos(math.pi * (h + 0.5) *
                                                         (i + 1) / 3)
                    else:
                        prior_tensor[i, h, w] = math.cos(math.pi * (w + 0.5) *
                                                         (i + 1 - half_fg) / 3)

        self.register_buffer('weight_orepa_prior', prior_tensor)

    def weight_gen(self):
        weight_orepa_origin = torch.einsum('oihw,o->oihw',
                                          self.weight_orepa_origin,
                                          self.vector[0, :])

        weight_orepa_avg = torch.einsum('oihw,hw->oihw', self.weight_orepa_avg_conv, self.weight_orepa_avg_avg)
        weight_orepa_avg = torch.einsum(
             'oihw,o->oihw',
             torch.einsum('oi,hw->oihw', self.weight_orepa_avg_conv.squeeze(3).squeeze(2),
                          self.weight_orepa_avg_avg), self.vector[1, :])


        weight_orepa_pfir = torch.einsum(
            'oihw,o->oihw',
            torch.einsum('oi,ohw->oihw', self.weight_orepa_pfir_conv.squeeze(3).squeeze(2),
                          self.weight_orepa_prior), self.vector[2, :])

        weight_orepa_1x1_kxk_conv1 = None
        if hasattr(self, 'weight_orepa_1x1_kxk_idconv1'):
            weight_orepa_1x1_kxk_conv1 = (self.weight_orepa_1x1_kxk_idconv1 +
                                        self.id_tensor).squeeze(3).squeeze(2)
        elif hasattr(self, 'weight_orepa_1x1_kxk_conv1'):
            weight_orepa_1x1_kxk_conv1 = self.weight_orepa_1x1_kxk_conv1.squeeze(3).squeeze(2)
        else:
            raise NotImplementedError
        weight_orepa_1x1_kxk_conv2 = self.weight_orepa_1x1_kxk_conv2

        if self.groups > 1:
            g = self.groups
            t, ig = weight_orepa_1x1_kxk_conv1.size()
            o, tg, h, w = weight_orepa_1x1_kxk_conv2.size()
            weight_orepa_1x1_kxk_conv1 = weight_orepa_1x1_kxk_conv1.view(
                g, int(t / g), ig)
            weight_orepa_1x1_kxk_conv2 = weight_orepa_1x1_kxk_conv2.view(
                g, int(o / g), tg, h, w)
            weight_orepa_1x1_kxk = torch.einsum('gti,gothw->goihw',
                                              weight_orepa_1x1_kxk_conv1,
                                              weight_orepa_1x1_kxk_conv2).reshape(
                                                  o, ig, h, w)
        else:
            weight_orepa_1x1_kxk = torch.einsum('ti,othw->oihw',
                                              weight_orepa_1x1_kxk_conv1,
                                              weight_orepa_1x1_kxk_conv2)
        weight_orepa_1x1_kxk = torch.einsum('oihw,o->oihw', weight_orepa_1x1_kxk, self.vector[3, :])

        weight_orepa_1x1 = 0
        if hasattr(self, 'weight_orepa_1x1'):
            weight_orepa_1x1 = transVI_multiscale(self.weight_orepa_1x1,
                                                self.kernel_size)
            weight_orepa_1x1 = torch.einsum('oihw,o->oihw', weight_orepa_1x1,
                                           self.vector[4, :])

        weight_orepa_gconv = self.dwsc2full(self.weight_orepa_gconv_dw,
                                          self.weight_orepa_gconv_pw,
                                          self.in_channels, self.groups)
        weight_orepa_gconv = torch.einsum('oihw,o->oihw', weight_orepa_gconv,
                                        self.vector[5, :])

        weight = weight_orepa_origin + weight_orepa_avg + weight_orepa_1x1 + weight_orepa_1x1_kxk + weight_orepa_pfir + weight_orepa_gconv

        return weight

    def dwsc2full(self, weight_dw, weight_pw, groups, groups_conv=1):

        t, ig, h, w = weight_dw.size()
        o, _, _, _ = weight_pw.size()
        tg = int(t / groups)
        i = int(ig * groups)
        ogc = int(o / groups_conv)
        groups_gc = int(groups / groups_conv)
        weight_dw = weight_dw.view(groups_conv, groups_gc, tg, ig, h, w)
        weight_pw = weight_pw.squeeze().view(ogc, groups_conv, groups_gc, tg)

        weight_dsc = torch.einsum('cgtihw,ocgt->cogihw', weight_dw, weight_pw)
        return weight_dsc.reshape(o, int(i/groups_conv), h, w)

    def forward(self, inputs=None):
        if hasattr(self, 'orepa_reparam'):
            return self.nonlinear(self.orepa_reparam(inputs))
        
        weight = self.weight_gen()

        if self.weight_only is True:
            return weight

        out = F.conv2d(
            inputs,
            weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)
        return self.nonlinear(self.bn(out))

    def get_equivalent_kernel_bias(self):
        return transI_fusebn(self.weight_gen(), self.bn)

    def switch_to_deploy(self):
        if hasattr(self, 'or1x1_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.orepa_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                     kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding, dilation=self.dilation, groups=self.groups, bias=True)
        self.orepa_reparam.weight.data = kernel
        self.orepa_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('weight_orepa_origin')
        self.__delattr__('weight_orepa_1x1')
        self.__delattr__('weight_orepa_1x1_kxk_conv2')
        if hasattr(self, 'weight_orepa_1x1_kxk_idconv1'):
            self.__delattr__('id_tensor')
            self.__delattr__('weight_orepa_1x1_kxk_idconv1')
        elif hasattr(self, 'weight_orepa_1x1_kxk_conv1'):
            self.__delattr__('weight_orepa_1x1_kxk_conv1')
        else:
            raise NotImplementedError
        self.__delattr__('weight_orepa_avg_avg')  
        self.__delattr__('weight_orepa_avg_conv')
        self.__delattr__('weight_orepa_pfir_conv')
        self.__delattr__('weight_orepa_prior')
        self.__delattr__('weight_orepa_gconv_dw')
        self.__delattr__('weight_orepa_gconv_pw')

        self.__delattr__('bn')
        self.__delattr__('vector')

    def init_gamma(self, gamma_value):
        init.constant_(self.vector, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        init.constant_(self.vector[0, :], 1.0)

# ---- 原样迁移自 nn/extra_modules/block.py:1753-1763 ----
class Bottleneck_OREPA(Bottleneck):
    """Standard bottleneck with OREPA."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        if k[0] == 1:
            self.cv1 = Conv(c1, c_)
        else:
            self.cv1 = OREPA(c1, c_, k[0])
        self.cv2 = OREPA(c_, c2, k[1], groups=g)

# ---- 原样迁移自 nn/extra_modules/orepa.py:424-571 ----
class OREPA_3x3_RepVGG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=None, groups=1, dilation=1, act=True,
                 internal_channels_1x1_3x3=None,
                 deploy=False):
        super(OREPA_3x3_RepVGG, self).__init__()
        self.deploy = deploy

        self.nonlinear = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        padding = autopad(kernel_size, padding, dilation)
        assert padding == kernel_size // 2

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.branch_counter = 0

        self.weight_rbr_origin = nn.Parameter(torch.Tensor(out_channels, int(in_channels/self.groups), kernel_size, kernel_size))
        init.kaiming_uniform_(self.weight_rbr_origin, a=math.sqrt(1.0))
        self.branch_counter += 1


        if groups < out_channels:
            self.weight_rbr_avg_conv = nn.Parameter(torch.Tensor(out_channels, int(in_channels/self.groups), 1, 1))
            self.weight_rbr_pfir_conv = nn.Parameter(torch.Tensor(out_channels, int(in_channels/self.groups), 1, 1))
            init.kaiming_uniform_(self.weight_rbr_avg_conv, a=1.0)
            init.kaiming_uniform_(self.weight_rbr_pfir_conv, a=1.0)
            self.weight_rbr_avg_conv.data
            self.weight_rbr_pfir_conv.data
            self.register_buffer('weight_rbr_avg_avg', torch.ones(kernel_size, kernel_size).mul(1.0/kernel_size/kernel_size))
            self.branch_counter += 1

        else:
            raise NotImplementedError
        self.branch_counter += 1

        if internal_channels_1x1_3x3 is None:
            internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels   # For mobilenet, it is better to have 2X internal channels

        if internal_channels_1x1_3x3 == in_channels:
            self.weight_rbr_1x1_kxk_idconv1 = nn.Parameter(torch.zeros(in_channels, int(in_channels/self.groups), 1, 1))
            id_value = np.zeros((in_channels, int(in_channels/self.groups), 1, 1))
            for i in range(in_channels):
                id_value[i, i % int(in_channels/self.groups), 0, 0] = 1
            id_tensor = torch.from_numpy(id_value).type_as(self.weight_rbr_1x1_kxk_idconv1)
            self.register_buffer('id_tensor', id_tensor)

        else:
            self.weight_rbr_1x1_kxk_conv1 = nn.Parameter(torch.Tensor(internal_channels_1x1_3x3, int(in_channels/self.groups), 1, 1))
            init.kaiming_uniform_(self.weight_rbr_1x1_kxk_conv1, a=math.sqrt(1.0))
        self.weight_rbr_1x1_kxk_conv2 = nn.Parameter(torch.Tensor(out_channels, int(internal_channels_1x1_3x3/self.groups), kernel_size, kernel_size))
        init.kaiming_uniform_(self.weight_rbr_1x1_kxk_conv2, a=math.sqrt(1.0))
        self.branch_counter += 1

        expand_ratio = 8
        self.weight_rbr_gconv_dw = nn.Parameter(torch.Tensor(in_channels*expand_ratio, 1, kernel_size, kernel_size))
        self.weight_rbr_gconv_pw = nn.Parameter(torch.Tensor(out_channels, in_channels*expand_ratio, 1, 1))
        init.kaiming_uniform_(self.weight_rbr_gconv_dw, a=math.sqrt(1.0))
        init.kaiming_uniform_(self.weight_rbr_gconv_pw, a=math.sqrt(1.0))
        self.branch_counter += 1

        if out_channels == in_channels and stride == 1:
            self.branch_counter += 1

        self.vector = nn.Parameter(torch.Tensor(self.branch_counter, self.out_channels))
        self.bn = nn.BatchNorm2d(out_channels)

        self.fre_init()

        init.constant_(self.vector[0, :], 0.25)    #origin
        init.constant_(self.vector[1, :], 0.25)      #avg
        init.constant_(self.vector[2, :], 0.0)      #prior
        init.constant_(self.vector[3, :], 0.5)    #1x1_kxk
        init.constant_(self.vector[4, :], 0.5)     #dws_conv


    def fre_init(self):
        prior_tensor = torch.Tensor(self.out_channels, self.kernel_size, self.kernel_size)
        half_fg = self.out_channels/2
        for i in range(self.out_channels):
            for h in range(3):
                for w in range(3):
                    if i < half_fg:
                        prior_tensor[i, h, w] = math.cos(math.pi*(h+0.5)*(i+1)/3)
                    else:
                        prior_tensor[i, h, w] = math.cos(math.pi*(w+0.5)*(i+1-half_fg)/3)

        self.register_buffer('weight_rbr_prior', prior_tensor)

    def weight_gen(self):

        weight_rbr_origin = torch.einsum('oihw,o->oihw', self.weight_rbr_origin, self.vector[0, :])

        weight_rbr_avg = torch.einsum('oihw,o->oihw', torch.einsum('oihw,hw->oihw', self.weight_rbr_avg_conv, self.weight_rbr_avg_avg), self.vector[1, :])
        
        weight_rbr_pfir = torch.einsum('oihw,o->oihw', torch.einsum('oihw,ohw->oihw', self.weight_rbr_pfir_conv, self.weight_rbr_prior), self.vector[2, :])

        weight_rbr_1x1_kxk_conv1 = None
        if hasattr(self, 'weight_rbr_1x1_kxk_idconv1'):
            weight_rbr_1x1_kxk_conv1 = (self.weight_rbr_1x1_kxk_idconv1 + self.id_tensor).squeeze()
        elif hasattr(self, 'weight_rbr_1x1_kxk_conv1'):
            weight_rbr_1x1_kxk_conv1 = self.weight_rbr_1x1_kxk_conv1.squeeze()
        else:
            raise NotImplementedError
        weight_rbr_1x1_kxk_conv2 = self.weight_rbr_1x1_kxk_conv2

        if self.groups > 1:
            g = self.groups
            t, ig = weight_rbr_1x1_kxk_conv1.size()
            o, tg, h, w = weight_rbr_1x1_kxk_conv2.size()
            weight_rbr_1x1_kxk_conv1 = weight_rbr_1x1_kxk_conv1.view(g, int(t/g), ig)
            weight_rbr_1x1_kxk_conv2 = weight_rbr_1x1_kxk_conv2.view(g, int(o/g), tg, h, w)
            weight_rbr_1x1_kxk = torch.einsum('gti,gothw->goihw', weight_rbr_1x1_kxk_conv1, weight_rbr_1x1_kxk_conv2).view(o, ig, h, w)
        else:
            weight_rbr_1x1_kxk = torch.einsum('ti,othw->oihw', weight_rbr_1x1_kxk_conv1, weight_rbr_1x1_kxk_conv2)

        weight_rbr_1x1_kxk = torch.einsum('oihw,o->oihw', weight_rbr_1x1_kxk, self.vector[3, :])

        weight_rbr_gconv = self.dwsc2full(self.weight_rbr_gconv_dw, self.weight_rbr_gconv_pw, self.in_channels)
        weight_rbr_gconv = torch.einsum('oihw,o->oihw', weight_rbr_gconv, self.vector[4, :])    

        weight = weight_rbr_origin + weight_rbr_avg + weight_rbr_1x1_kxk + weight_rbr_pfir + weight_rbr_gconv

        return weight

    def dwsc2full(self, weight_dw, weight_pw, groups):
        
        t, ig, h, w = weight_dw.size()
        o, _, _, _ = weight_pw.size()
        tg = int(t/groups)
        i = int(ig*groups)
        weight_dw = weight_dw.view(groups, tg, ig, h, w)
        weight_pw = weight_pw.squeeze().view(o, groups, tg)
        
        weight_dsc = torch.einsum('gtihw,ogt->ogihw', weight_dw, weight_pw)
        return weight_dsc.view(o, i, h, w)

    def forward(self, inputs):
        weight = self.weight_gen()
        out = F.conv2d(inputs, weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        return self.nonlinear(self.bn(out))

# ---- 原样迁移自 nn/extra_modules/orepa.py:573-702 ----
class RepVGGBlock_OREPA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=None, groups=1, dilation=1, act=True, deploy=False, use_se=False):
        super(RepVGGBlock_OREPA, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        padding = autopad(kernel_size, padding, dilation)
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        assert kernel_size == 3
        assert padding == 1

        self.nonlinearity = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        if use_se:
            self.se = SEAttention(out_channels, reduction=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = OREPA_3x3_RepVGG(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, dilation=1)
            self.rbr_1x1 = ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, groups=groups, dilation=1)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        out1 = self.rbr_dense(inputs)
        out2 = self.rbr_1x1(inputs)
        out3 = id_out
        out = out1 + out2 + out3

        return self.nonlinearity(self.se(out))


    #   Optional. This improves the accuracy and facilitates quantization.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()

    # Not used for OREPA
    def get_custom_L2(self):
        K3 = self.rbr_dense.weight_gen()
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if not isinstance(branch, nn.BatchNorm2d):
            if isinstance(branch, OREPA_3x3_RepVGG):
                kernel = branch.weight_gen()
            elif isinstance(branch, ConvBN):
                kernel = branch.conv.weight
            else:
                raise NotImplementedError
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels, out_channels=self.rbr_dense.out_channels,
                                     kernel_size=self.rbr_dense.kernel_size, stride=self.rbr_dense.stride,
                                     padding=self.rbr_dense.padding, dilation=self.rbr_dense.dilation, groups=self.rbr_dense.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')

# ---- 原样迁移自 nn/extra_modules/block.py:1780-1791 ----
class Bottleneck_REPVGGOREPA(Bottleneck):
    """Standard bottleneck with DCNV2."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        if k[0] == 1:
            self.cv1 = Conv(c1, c_, 1)
        else:
            self.cv1 = RepVGGBlock_OREPA(c1, c_, 3)
        
        self.cv2 = RepVGGBlock_OREPA(c_, c2, 3, groups=g)

# ---- 原样迁移自 nn/extra_modules/block.py:7633-7637 ----
class C3k_AdditiveBlock(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(AdditiveBlock(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:7639-7642 ----
class C3k2_AdditiveBlock(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_AdditiveBlock(self.c, self.c, 2, shortcut, g) if c3k else AdditiveBlock(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/metaformer.py:49-86 ----
class MF_Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1
        
        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        
    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# ---- 原样迁移自 nn/extra_modules/metaformer.py:157-171 ----
class LayerNormWithoutBias(nn.Module):
    """
    Equal to partial(LayerNormGeneral, bias=False) but faster, 
    because it directly utilizes otpimized F.layer_norm
    """
    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.bias = None
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)

# ---- 原样迁移自 nn/extra_modules/metaformer.py:276-319 ----
class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """
    def __init__(self, dim,
                 token_mixer=nn.Identity, mlp=Mlp,
                 norm_layer=partial(LayerNormWithoutBias, eps=1e-6),
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None
                 ):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
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
        return x.permute(0, 3, 1, 2)

# ---- 原样迁移自 nn/extra_modules/block.py:8767-8773 ----
class C3k_CaFormer(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MetaFormerBlock(
                dim=c_, token_mixer=MF_Attention
            ) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:8775-8780 ----
class C3k2_CaFormer(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_CaFormer(self.c, self.c, n, shortcut, g) if c3k else MetaFormerBlock(
                dim=self.c, token_mixer=MF_Attention
            ) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/metaformer.py:174-200 ----
class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, dim, expansion_ratio=2,
        act1_layer=StarReLU, act2_layer=nn.Identity, 
        bias=False, kernel_size=7, padding=3,
        **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias) # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:8752-8758 ----
class C3k_ConvFormer(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MetaFormerBlock(
                dim=c_, token_mixer=SepConv
            ) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:8760-8765 ----
class C3k2_ConvFormer(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_ConvFormer(self.c, self.c, n, shortcut, g) if c3k else MetaFormerBlock(
                dim=self.c, token_mixer=SepConv
            ) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/ConverseNet.py:178-217 ----
class ConverseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale=1, padding_mode='replicate', eps=1e-5):
        super(ConverseBlock, self).__init__()
        """
        ConverseBlock: A Convolutional Block for Image Restoration using Converse2D Operations.

        This block consists of two main sub-blocks, each incorporating normalization, pointwise convolution,
        non-linearity, and (optionally) a custom reverse convolution (`Converse2D`) for learnable upsampling.
        It also includes residual connections to preserve information and improve gradient flow.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels to be produced by the block.
            kernel_size (int, optional): Kernel size used in the `Converse2D` operation. Default: 3.
            scale (int, optional): Upsampling scale factor. Default: 1 (no upsampling).
            padding_mode (str, optional): Padding mode to use in `Converse2D`. One of {'reflect', 'replicate', 'circular', 'constant'}. Default: 'circular'.
            eps (float, optional): A small epsilon value for numerical stability in normalization layers. Default: 1e-6.

        Forward:
            x (Tensor): Input tensor of shape (N, in_channels, H, W)
            Returns:
                Tensor: Output tensor of shape (N, out_channels, H * scale, W * scale)
        """

        self.conv1 = nn.Sequential(LayerNorm(in_channels, eps=1e-5, data_format="channels_first"),
                                   nn.Conv2d(in_channels, 2*out_channels, 1, 1, 0),
                                   nn.GELU(),
                                   Converse2D(2*out_channels, 2*out_channels, kernel_size, scale=scale, padding_mode=padding_mode, eps=eps), 
                                   nn.GELU(),
                                   nn.Conv2d(2*out_channels, out_channels, 1, 1, 0))
                                  
        self.conv2 = nn.Sequential(LayerNorm(in_channels, eps=1e-5, data_format="channels_first"),
                                   nn.Conv2d(out_channels, 2*out_channels, 1, 1, 0),
                                   nn.GELU(),
                                   nn.Conv2d(2*out_channels, out_channels, 1, 1, 0))
                                  
    def forward(self, x):
        x = self.conv1(x) + x
        x = self.conv2(x) + x
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:15299-15303 ----
class C3k_ConverseB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(ConverseBlock(c_, c_, 3) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:15305-15308 ----
class C3k2_ConverseB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_ConverseB(self.c, self.c, 2, shortcut, g) if c3k else ConverseBlock(self.c, self.c, 3) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/dsan.py:100-137 ----
class DSAN(nn.Module):
    def __init__(self, dim, kernel_size=7, dw_kernel_size=5, stride=1, dilation=1, group=1, 
                 mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = DSA(dim, kernel_size, dw_kernel_size, stride, dilation, group)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:13811-13815 ----
class C3k_DSAN(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DSAN(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:13817-13820 ----
class C3k2_DSAN(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DSAN(self.c, self.c, 2, shortcut, g) if c3k else DSAN(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/tsdn.py:108-139 ----
class DilatedMDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(DilatedMDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

# ---- 原样迁移自 nn/extra_modules/tsdn.py:142-190 ----
class DilatedOCA(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_heads, dim_head, bias):
        super(DilatedOCA, self).__init__()
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_spatial_heads
        self.scale = self.dim_head**-0.5

        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)
        self.qkv = nn.Conv2d(self.dim, self.inner_dim*3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        self.rel_pos_emb = RelPosEmb(
            block_size = window_size,
            rel_size = window_size + (self.overlap_win_size - window_size),
            dim_head = self.dim_head
        )
        self.fixed_pos_emb = FixedPosEmb(window_size, self.overlap_win_size)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qs, ks, vs = qkv.chunk(3, dim=1)

        # spatial attention
        qs = rearrange(qs, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = self.window_size, p2 = self.window_size)
        ks, vs = map(lambda t: self.unfold(t), (ks, vs))
        ks, vs = map(lambda t: rearrange(t, 'b (c j) i -> (b i) j c', c = self.inner_dim), (ks, vs))

        # print(f'qs.shape:{qs.shape}, ks.shape:{ks.shape}, vs.shape:{vs.shape}')
        #split heads
        qs, ks, vs = map(lambda t: rearrange(t, 'b n (head c) -> (b head) n c', head = self.num_spatial_heads), (qs, ks, vs))

        # attention
        qs = qs * self.scale
        spatial_attn = (qs @ ks.transpose(-2, -1))
        spatial_attn += self.rel_pos_emb(qs)
        spatial_attn += self.fixed_pos_emb()
        spatial_attn = spatial_attn.softmax(dim=-1)

        out = (spatial_attn @ vs)

        out = rearrange(out, '(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)', head = self.num_spatial_heads, h = h // self.window_size, w = w // self.window_size, p1 = self.window_size, p2 = self.window_size)

        # merge spatial and channel
        out = self.project_out(out)

        return out

# ---- 原样迁移自 nn/extra_modules/tsdn.py:209-230 ----
class DTAB(nn.Module):
    def __init__(self, dim, window_size=4, overlap_ratio=0.5, num_channel_heads=4, num_spatial_heads=2, spatial_dim_head=16, ffn_expansion_factor=1, bias=False, LayerNorm_type='BiasFree'):
        super(DTAB, self).__init__()

        self.spatial_attn = DilatedOCA(dim, window_size, overlap_ratio, num_spatial_heads, spatial_dim_head, bias)
        self.channel_attn = DilatedMDTA(dim, num_channel_heads, bias)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.norm4 = LayerNorm(dim, LayerNorm_type)

        self.channel_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.spatial_ffn = FeedForward(dim, ffn_expansion_factor, bias)


    def forward(self, x):
        x = x + self.channel_attn(self.norm1(x))
        x = x + self.channel_ffn(self.norm2(x))
        x = x + self.spatial_attn(self.norm3(x))
        x = x + self.spatial_ffn(self.norm4(x))
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:10924-10928 ----
class C3k_DTAB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DTAB(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:10930-10933 ----
class C3k2_DTAB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DTAB(self.c, self.c, 2, shortcut, g) if c3k else DTAB(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:12501-12507 ----
class C3k_DynamicFilter(C3k):
    def __init__(self, c1, c2, n=1, size=None, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MetaFormerBlock(
                dim=c_, token_mixer=partial(DynamicFilter, size=size),
            ) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:12509-12514 ----
class C3k2_DynamicFilter(C3k2):
    def __init__(self, c1, c2, n=1, size=None, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DynamicFilter(self.c, self.c, n, size, shortcut, g) if c3k else MetaFormerBlock(
                dim=self.c, token_mixer=partial(DynamicFilter, size=size),
            ) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:11146-11159 ----
class ETB(nn.Module):
    def __init__(self, dim=128, num_heads=4, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super(ETB, self).__init__()
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn_S = Attention_S(dim, num_heads, bias)
        self.attn_F = Attention_F(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + torch.add(self.attn_F(self.norm1(x)),self.attn_S(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:11161-11165 ----
class C3k_ETB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(ETB(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:11167-11170 ----
class C3k2_ETB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_ETB(self.c, self.c, 2, shortcut, g) if c3k else ETB(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/EVSSM.py:270-361 ----
class EVS(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3, bias=False, LayerNorm_type='WithBias', att=True, idx=3, patch=80):
        super(EVS, self).__init__()

        self.att = att
        self.idx = idx
        if self.att:
            self.norm1 = LayerNorm(dim)
            self.attn = SS2D(d_model=dim, patch=patch)

        self.norm2 = LayerNorm(dim)
        self.ffn = EDFFN(dim, ffn_expansion_factor, bias)

        self.kernel_size = (patch, patch)

    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c, h, w)
        # assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

    def forward(self, x):
        if self.att:

            # if self.idx % 2 == 1:
            #     x = torch.flip(x, dims=(-2, -1)).contiguous()
            # if self.idx % 2 == 0:
            #     x = torch.transpose(x, dim0=-2, dim1=-1).contiguous()
#            if self.idx % 4 == 3:
#                x = torch.flip(x, dims=(-2, -1)).contiguous()
#            if self.idx % 4 == 0:
#                x = torch.transpose(x, dim0=-2, dim1=-1).contiguous()

            x = self.grids(x)
            x = x + self.attn(self.norm1(x))
            x = self.grids_inverse(x)

        x = x + self.ffn(self.norm2(x))

        return x

# ---- 原样迁移自 nn/extra_modules/block.py:13566-13570 ----
class C3k_EVS(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(EVS(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:13572-13575 ----
class C3k2_EVS(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_EVS(self.c, self.c, n, shortcut, g) if c3k else EVS(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:10524-10550 ----
class FCA_CTA(nn.Module):
    def __init__(self, dim, num_heads=4, expansion_factor=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, b_idx=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        # CTA
        self.attn = Channel_Transposed_Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        ffn_hidden_dim = int(dim * expansion_factor)
        # DFFN
        self.ffn = DFFN(in_features=dim, hidden_features=ffn_hidden_dim, out_features=dim, act_layer=act_layer)

    def forward(self, x):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """        
        b, n, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x.transpose(1, 2).reshape((b, n, H, W))

# ---- 原样迁移自 nn/extra_modules/block.py:10563-10567 ----
class C3k_FCA_CTA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(FCA_CTA(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:10569-10572 ----
class C3k2_FCA_CTA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_FCA_CTA(self.c, self.c, 2, shortcut, g) if c3k else FCA_CTA(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:10496-10522 ----
class FCA_SFA(nn.Module):
    def __init__(self, dim, num_heads=4, reso=64, split_size=[2,4],shift_size=[1,2], expansion_factor=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, b_idx=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        # SFA
        self.attn = Spatial_Frequency_Attention(
            dim, num_heads=num_heads, reso=reso, split_size=split_size, shift_size=shift_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop, b_idx=b_idx
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        ffn_hidden_dim = int(dim * expansion_factor)
        # DFFN
        self.ffn = DFFN(in_features=dim, hidden_features=ffn_hidden_dim, out_features=dim, act_layer=act_layer)

    def forward(self, x):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """        
        b, n, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x.transpose(1, 2).reshape((b, n, H, W))

# ---- 原样迁移自 nn/extra_modules/block.py:10552-10556 ----
class C3k_FCA_SFA(C3k):
    def __init__(self, c1, c2, n=1, reso=None, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(FCA_SFA(c_, reso=reso) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:10558-10561 ----
class C3k2_FCA_SFA(C3k2):
    def __init__(self, c1, c2, n=1, reso=None, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_FCA_SFA(self.c, self.c, 2, reso, shortcut, g) if c3k else FCA_SFA(self.c, reso=reso) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:11270-11318 ----
class FDT(nn.Module):
    def __init__(self, inp_channels, num_heads=4, window_sizes=4, shifts=0, shared_depth=1, ffn_expansion_factor=2.66):
        super(FDT, self).__init__()
        self.shared_depth = shared_depth
        
        modules_ffd = {}
        modules_att = {} 
        modules_norm = {} 
        for i in range(shared_depth):
            modules_ffd['ffd{}'.format(i)] = FeedForward(inp_channels, ffn_expansion_factor, bias=False)
            modules_att['att_{}'.format(i)] = RSA(channels=inp_channels, num_heads=num_heads, shifts=shifts, window_sizes=window_sizes)
            modules_norm['norm_{}'.format(i)] = LayerNorm(inp_channels, 'WithBias')
            modules_norm['norm_{}'.format(i+2)] = LayerNorm(inp_channels, 'WithBias')
        self.modules_ffd = nn.ModuleDict(modules_ffd)
        self.modules_att = nn.ModuleDict(modules_att)
        self.modules_norm = nn.ModuleDict(modules_norm)

        modulec_ffd = {}
        modulec_att = {} 
        modulec_norm = {} 
        for i in range(shared_depth):
            modulec_ffd['ffd{}'.format(i)] = FeedForward(inp_channels, ffn_expansion_factor, bias=False)
            modulec_att['att_{}'.format(i)] = GSA(channels=inp_channels, num_heads=num_heads)
            modulec_norm['norm_{}'.format(i)] = LayerNorm(inp_channels, 'WithBias')
            modulec_norm['norm_{}'.format(i+2)] = LayerNorm(inp_channels, 'WithBias')
        self.modulec_ffd = nn.ModuleDict(modulec_ffd)
        self.modulec_att = nn.ModuleDict(modulec_att)
        self.modulec_norm = nn.ModuleDict(modulec_norm)

    def forward(self, x):
        atn = None
        B, C, H, W = x.size()
        for i in range(self.shared_depth):
            if i == 0: ## only calculate attention for the 1-st module
                x_, atn = self.modules_att['att_{}'.format(i)](self.modules_norm['norm_{}'.format(i)](x), None)
                x = self.modules_ffd['ffd{}'.format(i)](self.modules_norm['norm_{}'.format(i+2)](x_ + x)) + x_
            else:
                x_ = self.modules_att['att_{}'.format(i)](self.modules_norm['norm_{}'.format(i)](x), atn)
                x = self.modules_ffd['ffd{}'.format(i)](self.modules_norm['norm_{}'.format(i+2)](x_ + x)) + x_

        for i in range(self.shared_depth):
            if i == 0: ## only calculate attention for the 1-st module
                x_, atn = self.modulec_att['att_{}'.format(i)](self.modulec_norm['norm_{}'.format(i)](x), None)
                x = self.modulec_ffd['ffd{}'.format(i)](self.modulec_norm['norm_{}'.format(i+2)](x_ + x)) + x_
            else:
                x = self.modulec_att['att_{}'.format(i)](self.modulec_norm['norm_{}'.format(i)](x), atn)
                x = self.modulec_ffd['ffd{}'.format(i)](self.modulec_norm['norm_{}'.format(i+2)](x_ + x)) + x_
        
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:11320-11324 ----
class C3k_FDT(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(FDT(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:11326-11329 ----
class C3k2_FDT(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_FDT(self.c, self.c, 2, shortcut, g) if c3k else FDT(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/srconvnet.py:88-125 ----
class FConvMod(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        layer_scale_init_value = 1e-6
        self.num_heads = num_heads
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = FourierUnit(dim)
        self.v = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(num_heads), requires_grad=True)
        self.CPE = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        shortcut = x
        pos_embed = self.CPE(x)
        x = self.norm(x)
        a = self.a(x)
        v = self.v(x)
        a = rearrange(a, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        a_all = torch.split(a, math.ceil(N // 4), dim=-1)
        v_all = torch.split(v, math.ceil(N // 4), dim=-1)
        attns = []
        for a, v in zip(a_all, v_all):
            attn = a * v
            attn = self.layer_scale.unsqueeze(-1).unsqueeze(-1) * attn
            attns.append(attn)
        x = torch.cat(attns, dim=-1)
        x = F.softmax(x, dim=-1)
        x = rearrange(x, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        x = x + pos_embed
        x = self.proj(x)
        out = x + shortcut
        
        return out

# ---- 原样迁移自 nn/extra_modules/srconvnet.py:220-241 ----
class MixFFN(nn.Module):
    def __init__(self, dim, num_kernels=16):
        super().__init__()
        self.proj_in = nn.Conv2d(dim, dim * 2, 1)
        self.conv1 = DyConv(dim, kernel_size=5, groups=dim, num_kernels=num_kernels)
        self.conv2 = DyConv(dim, kernel_size=7, groups=dim, num_kernels=num_kernels)
        self.proj_out = nn.Conv2d(dim * 2, dim, 1)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.act(self.proj_in(x))
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.act(self.conv1(x1)).unsqueeze(dim=2)
        x2 = self.act(self.conv2(x2)).unsqueeze(dim=2)
        x = torch.cat([x1, x2], dim=2)
        x = rearrange(x, 'b c g h w -> b (c g) h w')
        x = self.proj_out(x) 
        x = x + shortcut
        return x

# ---- 原样迁移自 nn/extra_modules/srconvnet.py:244-254 ----
class FMABlock(nn.Module):
    def __init__(self, dim, num_heads=8, num_kernels=16):
        super().__init__()
        self.attention = FConvMod(dim, num_heads)
        self.ffn = MixFFN(dim, num_kernels)

    def forward(self, x):
        x = self.attention(x)
        x = self.ffn(x)
        
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:15374-15378 ----
class C3k_FMABlock(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(FMABlock(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:15380-15383 ----
class C3k2_FMABlock(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_FMABlock(self.c, self.c, 2, shortcut, g) if c3k else FMABlock(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:901-955 ----
class Faster_Block_EMA(nn.Module):
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,
                 mlp_ratio=2,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )
        self.attention = EMA(dim)
        
        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.attention(self.drop_path(self.mlp(x)))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:957-961 ----
class C3k_Faster_EMA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Faster_Block_EMA(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:963-966 ----
class C3k2_Faster_EMA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_Faster_EMA(self.c, self.c, 2, shortcut, g) if c3k else Faster_Block_EMA(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:4956-4971 ----
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, act_layer=nn.SiLU, num_experts=4):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = DynamicConv(inp, init_channels, kernel_size, stride, num_experts=num_experts)

        self.cheap_operation = DynamicConv(init_channels, new_channels, dw_size, 1, g=init_channels, num_experts=num_experts)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

# ---- 原样迁移自 nn/extra_modules/block.py:4990-4994 ----
class C3k_GhostDynamicConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostModule(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:4996-4999 ----
class C3k2_GhostDynamicConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_GhostDynamicConv(self.c, self.c, 2, shortcut, g) if c3k else GhostModule(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:9602-9660 ----
class HDRAB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(HDRAB, self).__init__()
        kernel_size = 3
        reduction = 8
        reduction_2 = 2

        self.cab = CAB(in_channels, reduction, bias)
        
        self.conv1x1_1 = nn.Conv2d(in_channels, in_channels // reduction_2, 1)

        self.conv1 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size, padding=2, dilation=2, bias=bias)

        self.conv3 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size, padding=3, dilation=3, bias=bias)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size, padding=4, dilation=4, bias=bias)

        self.conv3_1 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size, padding=3, dilation=3, bias=bias)
        self.relu3_1 = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size, padding=2, dilation=2, bias=bias)

        self.conv1_1 = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)
        self.relu1_1 = nn.ReLU(inplace=True)

        self.conv_tail = nn.Conv2d(in_channels // reduction_2, out_channels // reduction_2, kernel_size=kernel_size, padding=1, dilation=1, bias=bias)
        
        self.conv1x1_2 = nn.Conv2d(in_channels // reduction_2, in_channels, 1)

    def forward(self, y):
        y_d = self.conv1x1_1(y)
        y1 = self.conv1(y_d)
        y1_1 = self.relu1(y1)
        y2 = self.conv2(y1_1)
        y2_1 = y2 + y_d

        y3 = self.conv3(y2_1)
        y3_1 = self.relu3(y3)
        y4 = self.conv4(y3_1)
        y4_1 = y4 + y2_1

        y5 = self.conv3_1(y4_1)
        y5_1 = self.relu3_1(y5)
        y6 = self.conv2_1(y5_1+y3)
        y6_1 = y6 + y4_1

        y7 = self.conv1_1(y6_1+y2_1)
        y7_1 = self.relu1_1(y7)
        y8 = self.conv_tail(y7_1+y1)
        y8_1 = y8 + y6_1

        y9 = self.cab(self.conv1x1_2(y8_1))
        y9_1 = y + y9

        return y9_1

# ---- 原样迁移自 nn/extra_modules/block.py:9662-9666 ----
class C3k_HDRAB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(HDRAB(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:9668-9671 ----
class C3k2_HDRAB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_HDRAB(self.c, self.c, 2, shortcut, g) if c3k else HDRAB(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:10879-10906 ----
class HFERB(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.mid_dim = dim//2
        self.dim = dim
        self.act = nn.GELU()
        self.last_fc = nn.Conv2d(self.dim, self.dim, 1)

        # High-frequency enhancement branch
        self.fc = nn.Conv2d(self.mid_dim, self.mid_dim, 1)
        self.max_pool = nn.MaxPool2d(3, 1, 1)

        # Local feature extraction branch
        self.conv = nn.Conv2d(self.mid_dim, self.mid_dim, 3, 1, 1)

    def forward(self, x):
        self.h, self.w = x.shape[2:]
        short = x

        # Local feature extraction branch
        lfe = self.act(self.conv(x[:,:self.mid_dim,:,:]))

        # High-frequency enhancement branch
        hfe = self.act(self.fc(self.max_pool(x[:,self.mid_dim:,:,:])))

        x = torch.cat([lfe, hfe], dim=1)
        x = short + self.last_fc(x)
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:10908-10912 ----
class C3k_HFERB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(HFERB(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:10914-10917 ----
class C3k2_HFERB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_HFERB(self.c, self.c, 2, shortcut, g) if c3k else HFERB(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:6296-6314 ----
class Mlp_Heat(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:6428-6497 ----
class HeatBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        res: int = 14,
        infer_mode = False,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
        use_checkpoint: bool = False,
        drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        mlp_ratio: float = 4.0,
        post_norm = True,
        layer_scale = None,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(hidden_dim)
        self.op = Heat2D(res=res, dim=hidden_dim, hidden_dim=hidden_dim, infer_mode=infer_mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp_Heat(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=True)
        self.post_norm = post_norm
        self.layer_scale = layer_scale is not None
        
        self.infer_mode = infer_mode
        
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_dim),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(hidden_dim),
                                       requires_grad=True)
        
        self.freq_embed = nn.Parameter(torch.zeros(res, res, hidden_dim), requires_grad=True)
        trunc_normal_(self.freq_embed, std=0.02)
        self.op.infer_init_heat2d(self.freq_embed)

    def _forward(self, x: torch.Tensor):
        if not self.layer_scale:
            if self.post_norm:
                x = x + self.drop_path(self.norm1(self.op(x, self.freq_embed)))
                if self.mlp_branch:
                    x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.op(self.norm1(x), self.freq_embed))
                if self.mlp_branch:
                    x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
            return x
        if self.post_norm:
            x = x + self.drop_path(self.gamma1[:, None, None] * self.norm1(self.op(x, self.freq_embed)))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None] * self.norm2(self.mlp(x))) # FFN
        else:
            x = x + self.drop_path(self.gamma1[:, None, None] * self.op(self.norm1(x), self.freq_embed))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None] * self.mlp(self.norm2(x))) # FFN
        return x
    
    def forward(self, input: torch.Tensor):
        if not self.training:
            self.op.infer_init_heat2d(self.freq_embed)
        
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

# ---- 原样迁移自 nn/extra_modules/block.py:6499-6503 ----
class C3k_Heat(C3k):
    def __init__(self, c1, c2, n=1, feat_size=None, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(HeatBlock(c_, feat_size) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:6505-6508 ----
class C3k2_Heat(C3k2):
    def __init__(self, c1, c2, n=1, feat_size=None, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_Heat(self.c, self.c, 2, feat_size, shortcut, g) if c3k else HeatBlock(self.c, feat_size) for _ in range(n))

# ---- 原样迁移自 nn/backbone/inceptionnext.py:88-123 ----
class MetaNeXtBlock(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            token_mixer=InceptionDWConv2d,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=4,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,
            
    ):
        super().__init__()
        self.token_mixer = token_mixer(dim)
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

# ---- 原样迁移自 nn/extra_modules/block.py:10596-10600 ----
class C3k_IDWB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MetaNeXtBlock(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:10602-10605 ----
class C3k2_IDWB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_IDWB(self.c, self.c, 2, shortcut, g) if c3k else MetaNeXtBlock(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:8707-8713 ----
class C3k_IdentityFormer(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MetaFormerBlock(
                dim=c_, token_mixer=nn.Identity, norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False)
            ) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:8715-8720 ----
class C3k2_IdentityFormer(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_IdentityFormer(self.c, self.c, n, shortcut, g) if c3k else MetaFormerBlock(
                dim=self.c, token_mixer=nn.Identity, norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False)
            ) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:6123-6127 ----
class C3k_KAN(C3k):
    def __init__(self, c1, c2, n=1, kan_mothed=None, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_KAN(c_, c_, kan_mothed, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:6129-6132 ----
class C3k2_KAN(C3k2):
    def __init__(self, c1, c2, n=1, kan_mothed=None, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_KAN(self.c, self.c, 2, kan_mothed, shortcut, g) if c3k else Bottleneck_KAN(self.c, self.c, kan_mothed, shortcut, g) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:11635-11683 ----
class KatAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:11699-11746 ----
class Kat(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int=8,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = KAN,
            act_init: str = 'gelu',
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = KatAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            act_init=act_init,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.size()
        x = x.flatten(2).permute(0, 2, 1)
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x.permute(0, 2, 1).view([-1, C, H, W]).contiguous()

# ---- 原样迁移自 nn/extra_modules/block.py:11748-11752 ----
class C3k_Kat(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Kat(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:11754-11757 ----
class C3k2_Kat(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_Kat(self.c, self.c, 2, shortcut, g) if c3k else Kat(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:13237-13270 ----
class LFE_Module(nn.Module):
    def __init__(self,
                 dim,
                 stage=1,
                 mlp_ratio=2,
                 drop_path=0.1,
                 ):
        super().__init__()
        self.stage = stage
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_layer: List[nn.Module] = [
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)]

        self.mlp = nn.Sequential(*mlp_layer)
        self.LFEA = LFEA(dim)

        if stage == 0:
            self.Scharr_edge = Scharr(dim)
        else:
            self.gaussian = Gaussian(dim, 5, 1.0)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: Tensor) -> Tensor:
        # show_feature(x)
        if self.stage == 0:
            att = self.Scharr_edge(x)
        else:
            att = self.gaussian(x)
        x_att = self.LFEA(x, att)
        x = x + self.norm(self.drop_path(self.mlp(x_att)))
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:13272-13276 ----
class C3k_LFEM(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(LFE_Module(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:13278-13281 ----
class C3k2_LFEM(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_LFEM(self.c, self.c, n, shortcut, g) if c3k else LFE_Module(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/MobileUViT.py:93-109 ----
class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 3, padding=1, groups=in_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 3, padding=1, groups=in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ---- 原样迁移自 nn/extra_modules/MobileUViT.py:159-177 ----
class LocalAgg(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 9, padding=4, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 9, padding=4, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.sg = nn.Sigmoid()

    def forward(self, x):
        x = x + x * (self.sg(self.pos_embed(x)) - 0.5)
        x = x + x * (self.sg(self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))) - 0.5)
        x = x + x * (self.sg(self.drop_path(self.mlp(self.norm2(x)))) - 0.5)
        return x

# ---- 原样迁移自 nn/extra_modules/MobileUViT.py:112-156 ----
class GlobalSparseAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr = sr_ratio
        if self.sr > 1:
            self.sampler = nn.AvgPool2d(1, sr_ratio)
            kernel_size = sr_ratio
            self.LocalProp = nn.ConvTranspose2d(dim, dim, kernel_size, stride=sr_ratio, groups=dim)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sampler = nn.Identity()
            self.upsample = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x, H: int, W: int):
        B, N, C = x.shape
        if self.sr > 1.:
            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.sampler(x)
            x = x.flatten(2).transpose(1, 2)

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        if self.sr > 1:
            x = x.permute(0, 2, 1).reshape(B, C, int(H / self.sr), int(W / self.sr))
            x = self.LocalProp(x)
            x = x.reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# ---- 原样迁移自 nn/extra_modules/MobileUViT.py:180-202 ----
class SelfAttn(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1.):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = GlobalSparseAttn(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x

# ---- 原样迁移自 nn/extra_modules/MobileUViT.py:204-220 ----
class LGLBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=2):
        super().__init__()

        if sr_ratio > 1:
            self.LocalAgg = LocalAgg(dim, mlp_ratio, drop, drop_path, act_layer)
        else:
            self.LocalAgg = nn.Identity()

        self.SelfAttn = SelfAttn(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer,
                                 norm_layer, sr_ratio)

    def forward(self, x):
        x = self.LocalAgg(x)
        x = self.SelfAttn(x)
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:15284-15288 ----
class C3k_LGLB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(LGLBlock(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:15290-15293 ----
class C3k2_LGLB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_LGLB(self.c, self.c, 2, shortcut, g) if c3k else LGLBlock(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/LWGANet.py:273-330 ----
class LWGA_Block(nn.Module):
    def __init__(self,
                 dim,
                 stage,
                 att_kernel=11,
                 mlp_ratio=2.0,
                 drop_path=0,
                 act_layer=nn.SiLU,
                 norm_layer=nn.BatchNorm2d
                 ):
        super().__init__()
        self.stage = stage
        self.dim_split = dim // 4
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.PA = PA(self.dim_split, norm_layer, act_layer)     # PA is point attention
        self.LA = LA(self.dim_split, norm_layer, act_layer)     # LA is local attention
        self.MRA = MRA(self.dim_split, att_kernel, norm_layer)  # MRA is medium-range attention
        if stage == 2:
            self.GA3 = D_GA(self.dim_split, norm_layer)         # GA3 is global attention (stage of 3)
        elif stage == 3:
            self.GA4 = GA(self.dim_split)                       # GA4 is global attention (stage of 4)
            self.norm = norm_layer(self.dim_split)
        else:
            self.GA12 = GA12(self.dim_split, act_layer)         # GA12 is global attention (stages of 1 and 2)
            self.norm = norm_layer(self.dim_split)
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        # for training/inference
        shortcut = x.clone()
        x1, x2, x3, x4 = torch.split(x, [self.dim_split, self.dim_split, self.dim_split, self.dim_split], dim=1)
        x1 = x1 + self.PA(x1)
        x2 = self.LA(x2)
        x3 = self.MRA(x3)
        if self.stage == 2:
            x4 = x4 + self.GA3(x4)
        elif self.stage == 3:
            x4 = self.norm(x4 + self.GA4(x4))
        else:
            x4 = self.norm(x4 + self.GA12(x4))
        x_att = torch.cat((x1, x2, x3, x4), 1)

        x = shortcut + self.norm1(self.drop_path(self.mlp(x_att)))

        return x

# ---- 原样迁移自 nn/extra_modules/block.py:15389-15393 ----
class C3k_LWGA(C3k):
    def __init__(self, c1, c2, n=1, stage=None, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(LWGA_Block(c_, stage) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:15395-15398 ----
class C3k2_LWGA(C3k2):
    def __init__(self, c1, c2, n=1, stage=None, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_LWGA(self.c, self.c, 2, stage, shortcut, g) if c3k else LWGA_Block(self.c, stage) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:15938-15971 ----
class MFEB(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[2, 4, 8]):
        super(MFEB, self).__init__()
        act = nn.ReLU
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.layer1 = Conv(in_channels, in_channels, 3, act=act)
        self.layer2 = Conv(in_channels, in_channels, 3, d=rate1, act=act)
        self.layer3 = Conv(in_channels, in_channels, 3, d=rate2, act=act)
        self.layer4 = Conv(in_channels, in_channels, 3, d=rate3, act=act)
        self.project = Conv(in_channels, out_channels, 1, act=act)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim = 2)
        self.softmax_1 = nn.Sigmoid()
        self.SE1 = nn.Conv2d(in_channels,in_channels,1)
        self.SE2 = nn.Conv2d(in_channels,in_channels,1)
        self.SE3 = nn.Conv2d(in_channels,in_channels,1)
        self.SE4 = nn.Conv2d(in_channels,in_channels,1)
    def forward(self, x):
        y0 = self.layer1(x)
        y1 = self.layer2(y0+x)
        y2 = self.layer3(y1+x)
        y3 = self.layer4(y2+x)
        y0_weight = self.SE1(self.gap(y0))
        y1_weight = self.SE2(self.gap(y1))
        y2_weight = self.SE3(self.gap(y2))
        y3_weight = self.SE4(self.gap(y3))
        weight = torch.cat([y0_weight,y1_weight,y2_weight,y3_weight],2)
        weight = self.softmax(self.softmax_1(weight))
        y0_weight = torch.unsqueeze(weight[:,:,0],2)
        y1_weight = torch.unsqueeze(weight[:,:,1],2)
        y2_weight = torch.unsqueeze(weight[:,:,2],2)
        y3_weight = torch.unsqueeze(weight[:,:,3],2)
        x_att = y0_weight*y0+y1_weight*y1+y2_weight*y2+y3_weight*y3
        return self.project(x_att+x) 

# ---- 原样迁移自 nn/extra_modules/block.py:15973-15977 ----
class C3k_MFEB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MFEB(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:15979-15982 ----
class C3k2_MFEB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MFEB(self.c, self.c, 2, shortcut, g) if c3k else MFEB(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:9518-9553 ----
class spatial_strip_att(nn.Module):
    def __init__(self, dim, kernel=3, dilation=1, group=2, H=True) -> None:
        super().__init__()

        self.k = kernel
        pad = dilation*(kernel-1) // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel//2, 1) if H else (1, kernel//2)
        self.dilation = dilation
        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group*kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Tanh()
        self.inside_all = nn.Parameter(torch.zeros(dim,1,1), requires_grad=True)
        self.lamb_l = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(dim), requires_grad=True)
        gap_kernel = (None,1) if H else (1, None) 
        self.gap = nn.AdaptiveAvgPool2d(gap_kernel)

    def forward(self, x):
        identity_input = x.clone()
        filter = self.ap(x)
        filter = self.conv(filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel, dilation=self.dilation).reshape(n, self.group, c//self.group, self.k, h*w)
        n, c1, p, q = filter.shape
        filter = filter.reshape(n, c1//self.k, self.k, p*q).unsqueeze(2)
        filter = self.filter_act(filter)
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)

        out_low = out * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)
        out_low = out_low * self.lamb_l[None,:,None,None]
        out_high = identity_input * (self.lamb_h[None,:,None,None]+1.)

        return out_low + out_high

# ---- 原样迁移自 nn/extra_modules/block.py:9503-9515 ----
class cubic_attention(nn.Module):
    def __init__(self, dim, group, dilation, kernel) -> None:
        super().__init__()

        self.H_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel)
        self.W_spatial_att = spatial_strip_att(dim, dilation=dilation, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim,1,1))
        self.beta = nn.Parameter(torch.ones(dim,1,1))

    def forward(self, x):
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta

# ---- 原样迁移自 nn/extra_modules/block.py:9456-9500 ----
class dynamic_filter(nn.Module):
    def __init__(self, inchannels, kernel_size=3, dilation=1, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        self.dilation = dilation

        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size**2)
        self.act = nn.Tanh()
    
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.pad = nn.ReflectionPad2d(self.dilation*(kernel_size-1)//2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.inside_all = nn.Parameter(torch.zeros(inchannels,1,1), requires_grad=True)

    def forward(self, x):
        identity_input = x
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)     

        n, c, h, w = x.shape  
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size, dilation=self.dilation).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        n,c1,p,q = low_filter.shape
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
       
        low_filter = self.act(low_filter)
    
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        out_low = low_part * (self.inside_all + 1.) - self.inside_all * self.gap(identity_input)

        out_low = out_low * self.lamb_l[None,:,None,None]

        out_high = (identity_input) * (self.lamb_h[None,:,None,None] + 1.) 

        return out_low + out_high

# ---- 原样迁移自 nn/extra_modules/block.py:9556-9568 ----
class MultiShapeKernel(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1, group=8):
        super().__init__()

        self.square_att = dynamic_filter(inchannels=dim, dilation=dilation, group=group, kernel_size=kernel_size)
        self.strip_att = cubic_attention(dim, group=group, dilation=dilation, kernel=kernel_size)

    def forward(self, x):

        x1 = self.strip_att(x)
        x2 = self.square_att(x)

        return x1+x2

# ---- 原样迁移自 nn/extra_modules/block.py:9424-9454 ----
class DeepPoolLayer(nn.Module):
    def __init__(self, k):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [8,4,2]
        dilation = [3,7,9]
        pools, convs, dynas = [],[],[]
        for j, i in enumerate(self.pools_sizes):
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
            dynas.append(MultiShapeKernel(dim=k, kernel_size=3, dilation=dilation[j]))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.dynas = nn.ModuleList(dynas)
        self.relu = nn.GELU()
        self.conv_sum = nn.Conv2d(k, k, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            if i == 0:
                y = self.dynas[i](self.convs[i](self.pools[i](x)))
            else:
                y = self.dynas[i](self.convs[i](self.pools[i](x)+y_up))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.pools_sizes)-1:
                y_up = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        resl = self.relu(resl)
        resl = self.conv_sum(resl)

        return resl

# ---- 原样迁移自 nn/extra_modules/block.py:9570-9574 ----
class C3k_MSM(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DeepPoolLayer(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:9577-9580 ----
class C3k2_MSM(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MSM(self.c, self.c, 2, shortcut, g) if c3k else DeepPoolLayer(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:7950-8017 ----
class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        act_type (str): The type of activation. Defaults to 'GELU'.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 kernel_size=3,
                 act_type='GELU',
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = nn.GELU()

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:8020-8087 ----
class MultiOrderDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel.

    Args:
        embed_dims (int): Number of input channels.
        dw_dilation (list): Dilations of three DWConv layers.
        channel_split (list): The raletive ratio of three splited channels.
    """

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3,],
                 channel_split=[1, 3, 4,],
                ):
        super(MultiOrderDWConv, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1, dilation=dw_dilation[0],
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1, dilation=dw_dilation[1],
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1, dilation=dw_dilation[2],
        )
        # a channel convolution
        self.PW_conv = nn.Conv2d(  # point-wise convolution
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims-self.embed_dims_2:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:8090-8159 ----
class MultiOrderGatedAggregation(nn.Module):
    """Spatial Block with Multi-order Gated Aggregation.

    Args:
        embed_dims (int): Number of input channels.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for Spatial Block.
            Defaults to 'SiLU'.
    """

    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False,
                ):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )
        self.proj_2 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = nn.SiLU()
        self.act_gate = nn.SiLU()

        # decompose
        self.sigma = ElementScale(
            embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        # x_d: [B, C, H, W] -> [B, C, 1, 1]
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    def forward_gating(self, g, v):
        with torch.autocast(device_type='cuda', enabled=False):
            g = g.to(torch.float32)
            v = v.to(torch.float32)
            return self.proj_2(self.act_gate(g) * self.act_gate(v))

    def forward(self, x):
        shortcut = x.clone()
        # proj 1x1
        x = self.feat_decompose(x)
        # gating and value branch
        g = self.gate(x)
        v = self.value(x)
        # aggregation
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        x = x + shortcut
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:8162-8236 ----
class MogaBlock(nn.Module):
    """A block of MogaNet.

    Args:
        embed_dims (int): Number of input channels.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        act_type (str): The activation type for projections and FFNs.
            Defaults to 'GELU'.
        norm_cfg (str): The type of normalization layer. Defaults to 'BN'.
        init_value (float): Init value for Layer Scale. Defaults to 1e-5.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for the gating branch.
            Defaults to 'SiLU'.
    """

    def __init__(self,
                 embed_dims,
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_type='GELU',
                 norm_type='BN',
                 init_value=1e-5,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False,
                ):
        super(MogaBlock, self).__init__()
        self.out_channels = embed_dims

        self.norm1 = nn.BatchNorm2d(embed_dims)

        # spatial attention
        self.attn = MultiOrderGatedAggregation(
            embed_dims,
            attn_dw_dilation=attn_dw_dilation,
            attn_channel_split=attn_channel_split,
            attn_act_type=attn_act_type,
            attn_force_fp32=attn_force_fp32,
        )
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(embed_dims)

        # channel MLP
        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        self.mlp = ChannelAggregationFFN(  # DWConv + Channel Aggregation FFN
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_type=act_type,
            ffn_drop=drop_rate,
        )

        # init layer scale
        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

    def forward(self, x):
        # spatial
        identity = x
        x = self.layer_scale_1 * self.attn(self.norm1(x))
        x = identity + self.drop_path(x)
        # channel
        identity = x
        x = self.layer_scale_2 * self.mlp(self.norm2(x))
        x = identity + self.drop_path(x)
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:8238-8242 ----
class C3k_MogaBlock(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MogaBlock(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:8244-8247 ----
class C3k2_MogaBlock(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MogaBlock(self.c, self.c, 2, shortcut, g) if c3k else MogaBlock(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:1765-1769 ----
class C3k_OREPA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_OREPA(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:1771-1774 ----
class C3k2_OREPA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_OREPA(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_OREPA(self.c, self.c, shortcut, g) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:5167-5180 ----
class PKIModule_CAA(nn.Module):
    def __init__(self, ch, h_kernel_size = 11, v_kernel_size = 11) -> None:
        super().__init__()
        
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, ch)
        self.h_conv = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)
        self.v_conv = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)
        self.conv2 = Conv(ch, ch)
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor

# ---- 原样迁移自 nn/extra_modules/block.py:5182-5216 ----
class PKIModule(nn.Module):
    def __init__(self, inc, ouc, kernel_sizes=(3, 5, 7, 9, 11), expansion=1.0, with_caa=True, caa_kernel_size=11, add_identity=True) -> None:
        super().__init__()
        hidc = make_divisible(int(ouc * expansion), 8)
        
        self.pre_conv = Conv(inc, hidc)
        self.dw_conv = nn.ModuleList(nn.Conv2d(hidc, hidc, kernel_size=k, padding=autopad(k), groups=hidc) for k in kernel_sizes)
        self.pw_conv = Conv(hidc, hidc)
        self.post_conv = Conv(hidc, ouc)
        
        if with_caa:
            self.caa_factor = PKIModule_CAA(hidc, caa_kernel_size, caa_kernel_size)
        else:
            self.caa_factor = None
        
        self.add_identity = add_identity and inc == ouc
    
    def forward(self, x):
        x = self.pre_conv(x)
        
        y = x
        x = self.dw_conv[0](x)
        x = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv[1:]], dim=0), dim=0)
        x = self.pw_conv(x)
        
        if self.caa_factor is not None:
            y = self.caa_factor(y)
        if self.add_identity:
            y = x * y
            x = x + y
        else:
            x = x * y

        x = self.post_conv(x)
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:5218-5222 ----
class C3k_PKIModule(C3k):
    def __init__(self, c1, c2, n=1, kernel_sizes=(3, 5, 7, 9, 11), expansion=1.0, with_caa=True, caa_kernel_size=11, add_identity=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, True, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(PKIModule(c_, c_, kernel_sizes, expansion, with_caa, caa_kernel_size, add_identity) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:5224-5227 ----
class C3k2_PKIModule(C3k2):
    def __init__(self, c1, c2, n=1, kernel_sizes=(3, 5, 7, 9, 11), expansion=1.0, with_caa=True, caa_kernel_size=11, add_identity=True, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_PKIModule(self.c, self.c, 2, kernel_sizes, expansion, with_caa, caa_kernel_size, add_identity) if c3k else PKIModule(self.c, self.c, kernel_sizes, expansion, with_caa, caa_kernel_size, add_identity) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/hcfnet.py:64-81 ----
class ECA(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1):
        super(ECA, self).__init__()
        k=int(abs((math.log(in_channel,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=out.view(x.size(0),1,x.size(1))
        out=self.conv(out)
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x

# ---- 原样迁移自 nn/extra_modules/hcfnet.py:22-62 ----
class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size*patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True)) 
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P*P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output

# ---- 原样迁移自 nn/extra_modules/hcfnet.py:9-20 ----
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x

# ---- 原样迁移自 nn/extra_modules/hcfnet.py:84-114 ----
class PPA(nn.Module):
    def __init__(self, in_features, filters) -> None:
         super().__init__()

         self.skip = Conv(in_features, filters, act=False)
         self.c1 = Conv(filters, filters, 3)
         self.c2 = Conv(filters, filters, 3)
         self.c3 = Conv(filters, filters, 3)
         self.sa = SpatialAttentionModule()
         self.cn = ECA(filters)
         self.lga2 = LocalGlobalAttention(filters, 2)
         self.lga4 = LocalGlobalAttention(filters, 4)

         self.drop = nn.Dropout2d(0.1)
         self.bn1 = nn.BatchNorm2d(filters)
         self.silu = nn.SiLU()

    def forward(self, x):
        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.silu(x)
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:5310-5314 ----
class C3k_PPA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(PPA(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:5316-5319 ----
class C3k2_PPA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_PPA(self.c, self.c, 2, shortcut, g) if c3k else PPA(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/metaformer.py:203-217 ----
class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer: https://arxiv.org/abs/2111.11418
    Modfiled for [B, H, W, C] input
    """
    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        y = x.permute(0, 3, 1, 2)
        y = self.pool(y)
        y = y.permute(0, 2, 3, 1)
        return y - x

# ---- 原样迁移自 nn/extra_modules/block.py:8737-8743 ----
class C3k_PoolingFormer(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MetaFormerBlock(
                dim=c_, token_mixer=Pooling, norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False)
            ) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:8745-8750 ----
class C3k2_PoolingFormer(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_PoolingFormer(self.c, self.c, n, shortcut, g) if c3k else MetaFormerBlock(
                dim=self.c, token_mixer=Pooling, norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False)
            ) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:9680-9691 ----
class SAB(nn.Module):
    def __init__(self):
        super(SAB, self).__init__()
        kernel_size = 5
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, kernel_size)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale

# ---- 原样迁移自 nn/extra_modules/block.py:9693-9722 ----
class RAB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=True):
        super(RAB, self).__init__()
        kernel_size = 3
        stride = 1
        padding = 1
        reduction_2 = 2
        layers = []
        layers.append(nn.Conv2d(in_channels// reduction_2, out_channels// reduction_2, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels// reduction_2, out_channels// reduction_2, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        self.res = nn.Sequential(*layers)
        self.conv1x1_1 = nn.Conv2d(in_channels, in_channels // reduction_2, 1)
        self.conv1x1_2 = nn.Conv2d(in_channels // reduction_2, in_channels, 1)
        self.sab = SAB()

    def forward(self, x):
        x_d = self.conv1x1_1(x)
        x1 = x_d + self.res(x_d)
        x2 = x1 + self.res(x1)
        x3 = x2 + self.res(x2)

        x3_1 = x1 + x3
        x4 = x3_1 + self.res(x3_1)
        x4_1 = x_d + x4

        x5 = self.sab(self.conv1x1_2(x4_1))
        x5_1 = x + x5

        return x5_1

# ---- 原样迁移自 nn/extra_modules/block.py:9724-9728 ----
class C3k_RAB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RAB(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:9730-9733 ----
class C3k2_RAB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_RAB(self.c, self.c, 2, shortcut, g) if c3k else RAB(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:1793-1797 ----
class C3k_REPVGGOREPA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_REPVGGOREPA(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:1799-1802 ----
class C3k2_REPVGGOREPA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_REPVGGOREPA(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_REPVGGOREPA(self.c, self.c, shortcut, g) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/MaIR.py:520-584 ----
class RMB(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            inp_shape: tuple = None,
            shift_size=0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            ssm_ratio: float = 2.,
            mlp_ratio=1.5,
            scan_len=8,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = VMM(d_model=hidden_dim, d_state=d_state,expand=ssm_ratio,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.conv_blk = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim)
        
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.hidden_dim = hidden_dim

        self.shift_size = shift_size
        self.scan_len = scan_len

        self.inp_shape = inp_shape
        self._generate_ids()

    def _generate_ids(self):
        H,W = self.inp_shape

        xs_scan_ids, xs_inverse_ids = mair_ids_generate(inp_shape=(1, 1, H, W), scan_len=self.scan_len)# [B,H,W,C]
        self.xs_scan_ids = xs_scan_ids
        self.xs_inverse_ids = xs_inverse_ids

        xs_shift_scan_ids, xs_shift_inverse_ids = mair_shift_ids_generate(inp_shape=(1, 1, H, W), scan_len=self.scan_len, shift_len=self.scan_len//2)# [B,H,W,C]
        self.xs_shift_scan_ids = xs_shift_scan_ids
        self.xs_shift_inverse_ids = xs_shift_inverse_ids

        del xs_scan_ids, xs_inverse_ids, xs_shift_scan_ids, xs_shift_inverse_ids

    def forward(self, input):
        input = input.permute(0, 2, 3, 1).contiguous()

        self.xs_shift_scan_ids = self.xs_shift_scan_ids.to(input.device)
        self.xs_shift_inverse_ids = self.xs_shift_inverse_ids.to(input.device)
        self.xs_scan_ids = self.xs_scan_ids.to(input.device)
        self.xs_inverse_ids = self.xs_inverse_ids.to(input.device)

        x = self.ln_1(input)
        if self.shift_size > 0:
            x = input*self.skip_scale + self.drop_path(self.self_attention(x, (self.xs_shift_scan_ids, self.xs_shift_inverse_ids)))
        else:
            x = input*self.skip_scale + self.drop_path(self.self_attention(x, (self.xs_scan_ids, self.xs_inverse_ids)))
        
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x))

        x = x.permute(0, 3, 1, 2).contiguous()

        return x

# ---- 原样迁移自 nn/extra_modules/block.py:13903-13907 ----
class C3k_RMB(C3k):
    def __init__(self, c1, c2, n=1, size=None, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RMB(c_, size, shift_size=(0 if _ % 2 == 0 else 2)) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:13909-13912 ----
class C3k2_RMB(C3k2):
    def __init__(self, c1, c2, n=1, size=None, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_RMB(self.c, self.c, 2, size, shortcut, g) if c3k else RMB(self.c, size, shift_size=(0 if _ % 2 == 0 else 2)) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:5027-5034 ----
class RepViTBlock_EMA(RepViTBlock):
    def __init__(self, inp, oup, use_se=True):
        super().__init__(inp, oup, use_se)
        
        self.token_mixer = nn.Sequential(
            RepVGGDW(inp),
            EMA(inp) if use_se else nn.Identity(),
        )

# ---- 原样迁移自 nn/extra_modules/block.py:5058-5062 ----
class C3k_RVB_EMA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepViTBlock_EMA(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:5064-5067 ----
class C3k2_RVB_EMA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_RVB_EMA(self.c, self.c, 2, shortcut, g) if c3k else RepViTBlock_EMA(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/metaformer.py:89-100 ----
class RandomMixing(nn.Module):
    def __init__(self, num_tokens=196, **kwargs):
        super().__init__()
        self.random_matrix = nn.parameter.Parameter(
            data=torch.softmax(torch.rand(num_tokens, num_tokens), dim=-1), 
            requires_grad=False)
    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H*W, C)
        x = torch.einsum('mn, bnc -> bmc', self.random_matrix, x)
        x = x.reshape(B, H, W, C)
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:8722-8728 ----
class C3k_RandomMixing(C3k):
    def __init__(self, c1, c2, n=1, num_tokens=196, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MetaFormerBlock(
                dim=c_, token_mixer=partial(RandomMixing, num_tokens=num_tokens), norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False)
            ) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:8730-8735 ----
class C3k2_RandomMixing(C3k2):
    def __init__(self, c1, c2, n=1, num_tokens=196, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_RandomMixing(self.c, self.c, n, num_tokens, shortcut, g) if c3k else MetaFormerBlock(
                dim=self.c, token_mixer=partial(RandomMixing, num_tokens=num_tokens), norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False)
            ) for _ in range(n))

# ---- 原样迁移自 nn/backbone/rmt.py:36-100 ----
class RelPos2d(nn.Module):

    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)
        
    def generate_2d_decay(self, H: int, W: int):
        '''
        generate 2d decay mask, the result is (HW)*(HW)
        '''
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H*W, 2) #(H*W 2)
        mask = grid[:, None, :] - grid[None, :, :] #(H*W H*W 2)
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]  #(n H*W H*W)
        return mask
    
    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :] #(l l)
        mask = mask.abs() #(l l)
        mask = mask * self.decay[:, None, None]  #(n l l)
        return mask
    
    def forward(self, slen: Tuple[int], activate_recurrent=False, chunkwise_recurrent=False):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        if activate_recurrent:

            retention_rel_pos = self.decay.exp()

        elif chunkwise_recurrent:
            mask_h = self.generate_1d_decay(slen[0])
            mask_w = self.generate_1d_decay(slen[1])

            retention_rel_pos = (mask_h, mask_w)

        else:
            mask = self.generate_2d_decay(slen[0], slen[1]) #(n l l)
            retention_rel_pos = mask

        return retention_rel_pos

# ---- 原样迁移自 nn/backbone/rmt.py:20-33 ----
class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = x.permute(0, 3, 1, 2) #(b c h w)
        x = self.conv(x) #(b c h w)
        x = x.permute(0, 2, 3, 1) #(b h w c)
        return x

# ---- 原样迁移自 nn/backbone/rmt.py:235-278 ----
class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn=F.gelu,
        dropout=0.0,
        activation_dropout=0.0,
        layernorm_eps=1e-6,
        subln=False,
        subconv=False
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = activation_fn
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        if self.dwconv is not None:
            residual = x
            x = self.dwconv(x)
            x = x + residual
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x

# ---- 原样迁移自 nn/backbone/rmt.py:178-233 ----
class MaSA(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim*self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        rel_pos: mask: (n l l)
        '''
        bsz, h, w, _ = x.size()
        mask = rel_pos
        
        assert h*w == mask.size(1)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        qr = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d1)
        kr = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d1)


        qr = qr.flatten(2, 3) #(b n l d1)
        kr = kr.flatten(2, 3) #(b n l d1)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4) #(b n h w d2)
        vr = vr.flatten(2, 3) #(b n l d2)
        qk_mat = qr @ kr.transpose(-1, -2) #(b n l l)
        qk_mat = qk_mat + mask  #(b n l l)
        qk_mat = torch.softmax(qk_mat, -1) #(b n l l)
        output = torch.matmul(qk_mat, vr) #(b n l d2)
        output = output.transpose(1, 2).reshape(bsz, h, w, -1) #(b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output

# ---- 原样迁移自 nn/backbone/rmt.py:102-176 ----
class MaSAd(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)


        self.out_proj = nn.Linear(embed_dim*self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        '''
        x: (b h w c)
        mask_h: (n h h)
        mask_w: (n w w)
        '''
        bsz, h, w, _ = x.size()

        mask_h, mask_w = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        qr = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4) #(b n h w d1)
        kr = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4) #(b n h w d1)


        '''
        qr: (b n h w d1)
        kr: (b n h w d1)
        v: (b h w n*d2)
        '''
        
        qr_w = qr.transpose(1, 2) #(b h n w d1)
        kr_w = kr.transpose(1, 2) #(b h n w d1)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4) #(b h n w d2)

        qk_mat_w = qr_w @ kr_w.transpose(-1, -2) #(b h n w w)
        qk_mat_w = qk_mat_w + mask_w  #(b h n w w)
        qk_mat_w = torch.softmax(qk_mat_w, -1) #(b h n w w)
        v = torch.matmul(qk_mat_w, v) #(b h n w d2)


        qr_h = qr.permute(0, 3, 1, 2, 4) #(b w n h d1)
        kr_h = kr.permute(0, 3, 1, 2, 4) #(b w n h d1)
        v = v.permute(0, 3, 2, 1, 4) #(b w n h d2)

        qk_mat_h = qr_h @ kr_h.transpose(-1, -2) #(b w n h h)
        qk_mat_h = qk_mat_h + mask_h  #(b w n h h)
        qk_mat_h = torch.softmax(qk_mat_h, -1) #(b w n h h)
        output = torch.matmul(qk_mat_h, v) #(b w n h d2)
        
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1) #(b h w n*d2)
        output = output + lepe
        output = self.out_proj(output)
        return output

# ---- 原样迁移自 nn/backbone/rmt.py:280-314 ----
class RetBlock(nn.Module):
    def __init__(self, retention: str, embed_dim: int, num_heads: int, ffn_dim: int, drop_path=0., layerscale=False, layer_init_values=1e-5):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.retention_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        assert retention in ['chunk', 'whole']
        if retention == 'chunk':
            self.retention = MaSAd(embed_dim, num_heads)
        else:
            self.retention = MaSA(embed_dim, num_heads)
        self.drop_path = DropPath(drop_path)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.pos = DWConv2d(embed_dim, 3, 1, 1)

        if layerscale:
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim),requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim),requires_grad=True)

    def forward(
            self,
            x: torch.Tensor, 
            incremental_state=None,
            chunkwise_recurrent=False,
            retention_rel_pos=None
        ):
        x = x + self.pos(x)
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.final_layer_norm(x)))
        else:
            x = x + self.drop_path(self.retention(self.retention_layer_norm(x), retention_rel_pos, chunkwise_recurrent, incremental_state))
            x = x + self.drop_path(self.ffn(self.final_layer_norm(x)))
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:5107-5127 ----
class C3k_RetBlock(C3k):
    def __init__(self, c1, c2, n=1, retention='chunk', num_heads=8, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.retention = retention
        self.Relpos = RelPos2d(c_, num_heads, 2, 4)
        self.m = nn.Sequential(*(RetBlock(retention, c_, num_heads, c_) for _ in range(n)))
    
    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        b, c, h, w = x.size()
        rel_pos = self.Relpos((h, w), chunkwise_recurrent=self.retention == 'chunk')
        
        cv1 = self.cv1(x)
        for idx, layer in enumerate(self.m):
            if idx == 0:
                cv1 = layer(cv1.permute(0, 2, 3, 1), None, self.retention == 'chunk', rel_pos)
            else:
                cv1 = layer(cv1, None, self.retention == 'chunk', rel_pos)
        cv2 = self.cv2(x)
        return self.cv3(torch.cat((cv1.permute(0, 3, 1, 2), cv2), 1))

# ---- 原样迁移自 nn/extra_modules/block.py:5129-5151 ----
class C3k2_RetBlock(C3k2):
    def __init__(self, c1, c2, n=1, retention='chunk', num_heads=8, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.retention = retention
        if c3k:
            self.Relpos = None
        else:
            self.Relpos = RelPos2d(self.c, num_heads, 2, 4)
        self.m = nn.ModuleList(C3k_RetBlock(self.c, self.c, 2, retention, num_heads, shortcut, g) if c3k else RetBlock(retention, self.c, num_heads, self.c) for _ in range(n))
    
    def forward(self, x):
        """Forward pass through C2f layer."""
        b, c, h, w = x.size()
        if self.Relpos is not None:
            rel_pos = self.Relpos((h, w), chunkwise_recurrent=self.retention == 'chunk')
        
        y = list(self.cv1(x).chunk(2, 1))
        for layer in self.m:
            if self.Relpos is not None:
                y.append(layer(y[-1].permute(0, 2, 3, 1), None, self.retention == 'chunk', rel_pos).permute(0, 3, 1, 2))
            else:
                y.append(layer(y[-1]))
        return self.cv2(torch.cat(y, 1))

# ---- 原样迁移自 nn/extra_modules/semnet.py:118-189 ----
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        
        self.mamba2 = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        
    
    @autocast(enabled=False)
    def forward(self, x, pe, mask):
        x_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
            self.to(x.dtype)
        B, C = x.shape[:2]
        B, C, H, W = x.shape

        reversed_x1 = x.clone()

        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        
        x2 = x.transpose(-1,-2)
  
        reversed_x2 = x2.clone()

        reversed_x1[:,:,1::2,:] = x[:,:,1::2,:].flip(-1)
        
        reversed_x2[:,:,1::2,:] = x2[:,:,1::2,:].flip(-1)

        #### add positional embedding
        x1_flat = reversed_x1.reshape(B, C, n_tokens).transpose(-1, -2)
        
        x1_flat = x1_flat + pe[:n_tokens, :]

        x2_flat = reversed_x2.reshape(B, C, n_tokens).transpose(-1, -2)
        
        x2_flat = x2_flat + pe[:n_tokens, :]
        ###### end adding positional embedding

        x1_norm = self.norm(x1_flat)
        x1_mamba = self.mamba(x1_norm)
        
        x2_norm = self.norm2(x2_flat)
        x2_mamba = self.mamba2(x2_norm)

        out1 = x1_mamba.transpose(-1, -2).reshape(B, C, H, W)
        out1_clone2 = out1
        out1_clone2[:,:,1::2,:] = out1[:,:,1::2,:].flip(-1)
        
        out2 = x2_mamba.transpose(-1, -2).reshape(B, C, W, H)
        out2_clone2 = out2
        out2_clone2[:,:,1::2,:] = out2[:,:,1::2,:].flip(-1)
        out2_clone2 = out2_clone2.transpose(-1,-2)
        
        out = out1_clone2 + out2_clone2
        
        return out

# ---- 原样迁移自 nn/extra_modules/semnet.py:191-198 ----
def PositionalEncoding(d_model, max_len=5000):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe.cuda()

# ---- 原样迁移自 nn/extra_modules/semnet.py:201-222 ----
class SBSM(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(SBSM, self).__init__()
        self.dim = dim
        
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        ##### Try Mamba
        self.attn = MambaLayer(dim)
        #####

        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.ffn = SEFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x_spatial = x
        B, C, H, W = x.size()
        pos = PositionalEncoding(self.dim, H*W).to(x.device)
        x = x + self.attn(self.norm1(x), pos, None).to(x_spatial.dtype)
        x = x + self.ffn(self.norm2(x), x_spatial)

        return x

# ---- 原样迁移自 nn/extra_modules/block.py:13371-13375 ----
class C3k_SBSM(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(SBSM(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:13377-13380 ----
class C3k2_SBSM(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_SBSM(self.c, self.c, n, shortcut, g) if c3k else SBSM(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:9103-9146 ----
class SFHF_FFN(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(SFHF_FFN, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2
        # PW first or DW first?
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim*2, 1),
        )

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=1,
                      groups=self.dim_sp),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=5, padding=2,
                      groups=self.dim_sp),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=7, padding=3,
                      groups=self.dim_sp),
        )

        self.gelu = nn.GELU()
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
        )


    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim_sp, dim=1))
        x[1] = self.conv1_1(x[1])
        x[2] = self.conv1_2(x[2])
        x[3] = self.conv1_3(x[3])
        x = torch.cat(x, dim=1)
        x = self.gelu(x)
        x = self.conv_fina(x)


        return x

# ---- 原样迁移自 nn/extra_modules/block.py:9170-9207 ----
class SFHF_FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=4):
        # bn_layer not used
        super(SFHF_FourierUnit, self).__init__()
        self.groups = groups
        self.bn = nn.BatchNorm2d(out_channels * 2)

        self.fdc = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2 * self.groups,
                                                        kernel_size=1, stride=1, padding=0, groups=self.groups, bias=True)
        self.weight = nn.Sequential(
             nn.Conv2d(in_channels=in_channels * 2, out_channels=self.groups, kernel_size=1, stride=1, padding=0),
             nn.Softmax(dim=1)
        )

        self.fpe = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3,
                                        padding=1, stride=1, groups=in_channels * 2,bias=True)

    def forward(self, x):
        batch, c, h, w = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        ffted = rearrange(ffted, 'b c h w d -> b (c d) h w').contiguous()
        ffted = self.bn(ffted)
        ffted = self.fpe(ffted) + ffted
        dy_weight = self.weight(ffted)
        ffted = self.fdc(ffted).view(batch, self.groups, 2*c, h, -1)  # (batch, c*2, h, w/2+1)
        ffted = torch.einsum('ijkml,ijml->ikml', ffted, dy_weight)
        ffted = F.gelu(ffted)
        ffted = rearrange(ffted, 'b (c d) h w -> b c h w d', d=2).contiguous()
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output

# ---- 原样迁移自 nn/extra_modules/block.py:9210-9233 ----
class TokenMixer_For_Gloal(nn.Module):
    def __init__(
            self,
            dim
    ):
        super(TokenMixer_For_Gloal, self).__init__()
        self.dim = dim
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU()
        )
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU()
        )
        self.FFC = SFHF_FourierUnit(self.dim*2, self.dim*2)

    def forward(self, x):
        x = self.conv_init(x)
        x0 = x
        x = self.FFC(x)
        x = self.conv_fina(x+x0)

        return x

# ---- 原样迁移自 nn/extra_modules/block.py:9149-9167 ----
class TokenMixer_For_Local(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(TokenMixer_For_Local, self).__init__()
        self.dim = dim
        self.dim_sp = dim//2

        self.CDilated_1 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=1, dilation=1, groups=self.dim_sp)
        self.CDilated_2 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=2, dilation=2, groups=self.dim_sp)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        cd1 = self.CDilated_1(x1)
        cd2 = self.CDilated_2(x2)
        x = torch.cat([cd1, cd2], dim=1)

        return x

# ---- 原样迁移自 nn/extra_modules/block.py:9236-9278 ----
class SFHF_Mixer(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer_for_local=TokenMixer_For_Local,
            token_mixer_for_gloal=TokenMixer_For_Gloal,
    ):
        super(SFHF_Mixer, self).__init__()
        self.dim = dim
        self.mixer_local = token_mixer_for_local(dim=self.dim,)
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim,)

        self.ca_conv = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2*dim, 2*dim//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*dim//2, 2*dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.gelu = nn.GELU()
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, 2*dim, 1),
        )



    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim, dim=1))
        x_local = self.mixer_local(x[0])
        x_gloal = self.mixer_gloal(x[1])
        x = torch.cat([x_local, x_gloal], dim=1)
        x = self.gelu(x)
        x = self.ca(x) * x
        x = self.ca_conv(x)



        return x

# ---- 原样迁移自 nn/extra_modules/block.py:9281-9309 ----
class SFHF_Block(nn.Module):
    def __init__(
            self,
            dim,
            norm_layer=nn.BatchNorm2d,
            token_mixer=SFHF_Mixer,
    ):
        super(SFHF_Block, self).__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mixer = token_mixer(dim=self.dim)
        self.ffn = SFHF_FFN(dim=self.dim)

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        copy = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = x * self.beta + copy

        copy = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x * self.gamma + copy

        return x

# ---- 原样迁移自 nn/extra_modules/block.py:9311-9315 ----
class C3k_SFHF(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(SFHF_Block(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:9317-9320 ----
class C3k2_SFHF(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_SFHF(self.c, self.c, 2, shortcut, g) if c3k else SFHF_Block(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:8261-8270 ----
class SHSABlock_FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.SiLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:8348-8356 ----
class SHSABlock(torch.nn.Module):
    def __init__(self, dim, qk_dim=16, pdim=32):
        super().__init__()
        self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups = dim, bn_weight_init = 0))
        self.mixer = Residual(SHSA(dim, qk_dim, pdim))
        self.ffn = Residual(SHSABlock_FFN(dim, int(dim * 2)))
    
    def forward(self, x):
        return self.ffn(self.mixer(self.conv(x)))

# ---- 原样迁移自 nn/extra_modules/block.py:8368-8372 ----
class C3k_SHSA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(SHSABlock(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:8374-8377 ----
class C3k2_SHSA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_SHSA(self.c, self.c, 2, shortcut, g) if c3k else SHSABlock(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:8573-8603 ----
class E_MLP(nn.Module):
    def __init__(self, feature_size, forward_expansion, dropout):
        super(E_MLP, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_size, forward_expansion * feature_size),
            nn.GELU(),
            nn.Linear(forward_expansion * feature_size, feature_size)
        )
        self.linear1 = nn.Linear(feature_size, forward_expansion * feature_size)
        self.act = nn.GELU()
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(in_channels=forward_expansion * feature_size, out_channels=forward_expansion * feature_size, kernel_size=3, padding=1, groups=1)

        # pixelwise convolution
        self.pixelwise_conv = nn.Conv2d(in_channels=forward_expansion * feature_size, out_channels=forward_expansion * feature_size, kernel_size=3, padding=1)

        self.linear2 = nn.Linear(forward_expansion * feature_size, feature_size)

    def forward(self, x):
        b, hw, c = x.size()
        feature_size = int(math.sqrt(hw))

        x = self.linear1(x)
        x = self.act(x)
        x = rearrange(x, 'b (h w) (c) -> b c h w', h=feature_size, w=feature_size)
        x = self.depthwise_conv(x)
        x = self.pixelwise_conv(x)
        x = rearrange(x, 'b c h w -> b (h w) (c)', h=feature_size, w=feature_size)
        out = self.linear2(x)

        return out

# ---- 原样迁移自 nn/extra_modules/block.py:8436-8549 ----
class Modulator(nn.Module):
    def __init__(self, in_ch, out_ch, with_pos=True):
        super(Modulator, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.rate = [1, 6, 12, 18]
        self.with_pos = with_pos
        self.patch_size = 2
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.CA_fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // 16, in_ch, bias=False),
            nn.Sigmoid(),
        )

        # Pixel Attention
        self.PA_conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.PA_bn = nn.BatchNorm2d(in_ch)
        self.sigmoid = nn.Sigmoid()

        # Spatial Attention
        self.SA_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=rate, dilation=rate),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch)
            ) for rate in self.rate
        ])
        self.SA_out_conv = nn.Conv2d(len(self.rate) * out_ch, out_ch, 1)

        self.output_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self._init_weights()

        self.pj_conv = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=self.patch_size + 1,
                         stride=self.patch_size, padding=self.patch_size // 2)
        self.pos_conv = nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1, groups=self.out_ch, bias=True)
        self.layernorm = nn.LayerNorm(self.out_ch, eps=1e-6)

    def forward(self, x):
        res = x
        pa = self.PA(x)
        ca = self.CA(x)

        # Softmax(PA @ CA)
        pa_ca = torch.softmax(pa @ ca, dim=-1)

        # Spatial Attention
        sa = self.SA(x)

        # (Softmax(PA @ CA)) @ SA
        out = pa_ca @ sa
        out = self.norm(self.output_conv(out))
        out = out + self.bias
        synergistic_attn = out + res
        return synergistic_attn

    # def forward(self, x):
    #     pa_out = self.pa(x)
    #     ca_out = self.ca(x)
    #     sa_out = self.sa(x)
    #     # Concatenate along channel dimension
    #     combined_out = torch.cat([pa_out, ca_out, sa_out], dim=1)
    #
    #     return self.norm(self.output_conv(combined_out))



    def PE(self, x):
        proj = self.pj_conv(x)

        if self.with_pos:
            pos = proj * self.sigmoid(self.pos_conv(proj))

        pos = pos.flatten(2).transpose(1, 2)  # BCHW -> BNC
        embedded_pos = self.layernorm(pos)

        return embedded_pos

    def PA(self, x):
        attn = self.PA_conv(x)
        attn = self.PA_bn(attn)
        attn = self.sigmoid(attn)
        return x * attn

    def CA(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.CA_fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    def SA(self, x):
        sa_outs = [block(x) for block in self.SA_blocks]
        sa_out = torch.cat(sa_outs, dim=1)
        sa_out = self.SA_out_conv(sa_out)
        return sa_out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# ---- 原样迁移自 nn/extra_modules/block.py:8551-8571 ----
class SMA(nn.Module):
    def __init__(self, feature_size, num_heads, dropout):
        super(SMA, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_size, num_heads=num_heads, dropout=dropout)
        self.combined_modulator = Modulator(feature_size, feature_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, value, key, query):
        MSA = self.attention(query, key, value)[0]

        # 将输出转换为适合AttentionBlock的输入格式
        batch_size, seq_len, feature_size = MSA.shape
        MSA = MSA.permute(0, 2, 1).view(batch_size, feature_size, int(seq_len**0.5), int(seq_len**0.5))
        # 通过CombinedModulator进行multi-attn fusion
        synergistic_attn = self.combined_modulator.forward(MSA)


        # 将输出转换回 (batch_size, seq_len, feature_size) 格式
        x = synergistic_attn.view(batch_size, feature_size, -1).permute(0, 2, 1)

        return x

# ---- 原样迁移自 nn/extra_modules/block.py:8605-8622 ----
class SMAFormerBlock(nn.Module):
    def __init__(self, ch_out, heads=8, dropout=0.1, forward_expansion=2):
        super(SMAFormerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(ch_out)
        self.norm2 = nn.LayerNorm(ch_out)
        self.synergistic_multi_attention = SMA(ch_out, heads, dropout)
        self.e_mlp = E_MLP(ch_out, forward_expansion, dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.flatten(2).permute(0, 2, 1)
        value, key, query, res = x, x, x, x
        attention = self.synergistic_multi_attention(query, key, value)
        query = self.dropout(self.norm1(attention + res))
        feed_forward = self.e_mlp(query)
        out = self.dropout(self.norm2(feed_forward + query))
        return out.permute(0, 2, 1).reshape((b, c, h, w))

# ---- 原样迁移自 nn/extra_modules/block.py:8644-8648 ----
class C3k_SMAFB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(SMAFormerBlock(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:8650-8653 ----
class C3k2_SMAFB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_SMAFB(self.c, self.c, 2, shortcut, g) if c3k else SMAFormerBlock(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/UniConvNet.py:120-188 ----
class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a1 = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 4, 7, padding=3, groups=dim // 4)
        )
        self.v1 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.v11 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.v12 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.conv3_1 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim//4)

        self.norm2 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")
        self.a2 = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 2, 9, padding=4, groups=dim // 2)
        )
        self.v2 = nn.Conv2d(dim//2, dim//2, 1)
        self.v21 = nn.Conv2d(dim // 2, dim // 2, 1)
        self.v22 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.proj2 = nn.Conv2d(dim // 2, dim // 4, 1)
        self.conv3_2 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        self.norm3 = LayerNorm(dim * 3 // 4, eps=1e-6, data_format="channels_first")
        self.a3 = nn.Sequential(
            nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 11, padding=5, groups=dim * 3 // 4)
        )
        self.v3 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v31 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v32 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.proj3 = nn.Conv2d(dim * 3 // 4, dim // 4, 1)
        self.conv3_3 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        self.dim = dim

    def forward(self, x):

        x = self.norm1(x)
        x_split = torch.split(x, self.dim // 4, dim=1)
        a = self.a1(x_split[0])
        mul = a * self.v1(x_split[0])
        mul = self.v11(mul)
        x1 = self.conv3_1(self.v12(x_split[1]))
        x1 = x1 + a
        x1 = torch.cat((x1, mul), dim=1)

        x1 = self.norm2(x1)
        a = self.a2(x1)
        mul = a * self.v2(x1)
        mul = self.v21(mul)
        x2 = self.conv3_2(self.v22(x_split[2]))
        x2 = x2 + self.proj2(a)
        x2 = torch.cat((x2, mul), dim=1)

        x2 = self.norm3(x2)
        a = self.a3(x2)
        mul = a * self.v3(x2)
        mul = self.v31(mul)
        x3 = self.conv3_3(self.v32(x_split[3]))
        x3 = x3 + self.proj3(a)
        x = torch.cat((x3, mul), dim=1)

        return x

# ---- 原样迁移自 nn/extra_modules/UniConvNet.py:62-92 ----
class MLPLayer(nn.Module):
    r""" MLP layer of InternImage
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 # act_layer='GELU',
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ---- 原样迁移自 nn/extra_modules/UniConvNet.py:191-229 ----
class UniConvBlock(nn.Module):
    def __init__(self, dim,
                 drop=0.,
                 drop_path=0.,
                 mlp_ratio=4,
                 layer_scale_init_value=1e-5,
                 core_op=None
                 ):
        super().__init__()
        if core_op is None:
            assert opsm is not None, "DCNv3 未编译:先执行 compile_module/ops_dcnv3 下的编译步骤"
            core_op = getattr(opsm, 'DCNv3')

        self.attn = ConvMod(dim)
        self.mlp = MLPLayer(in_features=dim,
                            hidden_features=int(dim * mlp_ratio),
                            drop=drop)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True)
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True)
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                        requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = build_norm_layer(dim, 'LN')
        self.norm2 = build_norm_layer(dim, 'LN')
        self.dcn = core_op(
            channels=dim,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=dim // 8,
            offset_scale=1.0,
        )

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x.permute(0, 2, 3, 1)
        x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x.permute(0, 3, 1, 2)

# ---- 原样迁移自 nn/extra_modules/block.py:15269-15273 ----
class C3k_UniConvBlock(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(UniConvBlock(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:15275-15278 ----
class C3k2_UniConvBlock(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_UniConvBlock(self.c, self.c, 2, shortcut, g) if c3k else UniConvBlock(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:7460-7499 ----
class gConvBlock(nn.Module):
	def __init__(self, dim, kernel_size=3, gate_act=nn.Sigmoid, net_depth=8):
		super().__init__()
		self.dim = dim

		self.net_depth = net_depth
		self.kernel_size = kernel_size

		self.norm_layer = nn.BatchNorm2d(dim)
        
		self.Wv = nn.Sequential(
			nn.Conv2d(dim, dim, 1),
			nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, padding_mode='reflect')
		)

		self.Wg = nn.Sequential(
			nn.Conv2d(dim, dim, 1),
			gate_act() if gate_act in [nn.Sigmoid, nn.Tanh] else gate_act(inplace=True)
		)

		self.proj = nn.Conv2d(dim, dim, 1)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			gain = (8 * self.net_depth) ** (-1/4)    # self.net_depth ** (-1/2), the deviation seems to be too small, a bigger one may be better
			fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
			std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
			trunc_normal_(m.weight, std=std)

			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, X):
		iden = X
		X = self.norm_layer(X)
		out = self.Wv(X) * self.Wg(X)
		out = self.proj(out)
		return out + iden

# ---- 原样迁移自 nn/extra_modules/block.py:7501-7505 ----
class C3k_gConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(gConvBlock(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:7507-7510 ----
class C3k2_gConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_gConv(self.c, self.c, 2, shortcut, n) if c3k else gConvBlock(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:3727-3748 ----
class DualConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, g=4):
        """
        Initialize the DualConv class.
        :param input_channels: the number of input channels
        :param output_channels: the number of output channels
        :param stride: convolution stride
        :param g: the value of G used in DualConv
        """
        super(DualConv, self).__init__()
        # Group Convolution
        self.gc = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=g, bias=False)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, input_data):
        """
        Define how DualConv processes the input images or input feature maps.
        :param input_data: input images or input feature maps
        :return: return output feature maps
        """
        return self.gc(input_data) + self.pwc(input_data)

# ---- 原样迁移自 nn/extra_modules/block.py:3750-3756 ----
class EDLAN(nn.Module):
    def __init__(self, c, g=4) -> None:
        super().__init__()
        self.m = nn.Sequential(DualConv(c, c, 1, g=g), DualConv(c, c, 1, g=g))
    
    def forward(self, x):
        return self.m(x)

# ---- 原样迁移自 nn/extra_modules/block.py:3758-3777 ----
class CSP_EDLAN(nn.Module):
    # CSP Efficient Dual Layer Aggregation Networks
    def __init__(self, c1, c2, n=1, g=4, e=0.5) -> None:
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(EDLAN(self.c, g=g) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

# ---- 原样迁移自 nn/extra_modules/block.py:4659-4759 ----
class RepConvN(nn.Module):
    """RepConv is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process"""
        if hasattr(self, 'conv'):
            return self.forward_fuse(x)
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.c1
        groups = self.g
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')

# ---- 原样迁移自 nn/extra_modules/block.py:4761-4771 ----
class RepNBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

# ---- 原样迁移自 nn/extra_modules/block.py:4791-4802 ----
class RepNCSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

# ---- 原样迁移自 nn/extra_modules/block.py:5229-5248 ----
class RepNCSPELAN4_CAA(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3//2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3//2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3+(2*c4), c2, 1, 1)
        self.caa = CAA(c3+(2*c4))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(self.caa(torch.cat(y, 1)))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(self.caa(torch.cat(y, 1)))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- RepNBottleneck ----
    try:
        module = RepNBottleneck(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'RepNBottleneck  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'RepNBottleneck  自测跳过: {e}' + RESET)
    # ---- RepNCSP ----
    try:
        module = RepNCSP(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'RepNCSP  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'RepNCSP  自测跳过: {e}' + RESET)
    # ---- RepNCSPELAN4_CAA ----
    try:
        module = RepNCSPELAN4_CAA(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'RepNCSPELAN4_CAA  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'RepNCSPELAN4_CAA  自测跳过: {e}' + RESET)

