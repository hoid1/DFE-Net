'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:10578-10583 , nn/extra_modules/block.py:10585-10589 , nn/extra_modules/block.py:10591-10594 , nn/extra_modules/block.py:10612-10616 , nn/extra_modules/block.py:10618-10621 , nn/extra_modules/block.py:11012-11016 , nn/extra_modules/block.py:11018-11021 , nn/extra_modules/block.py:11443-11447 , nn/extra_modules/block.py:11449-11452 , nn/extra_modules/block.py:11914-11918 , nn/extra_modules/block.py:11920-11923 , nn/extra_modules/block.py:12039-12043 , nn/extra_modules/block.py:12045-12048 , nn/extra_modules/block.py:12894-12898 , nn/extra_modules/block.py:12900-12903 , nn/extra_modules/block.py:13057-13061 , nn/extra_modules/block.py:13063-13066 , nn/extra_modules/block.py:13118-13122 , nn/extra_modules/block.py:13124-13127 , nn/extra_modules/block.py:13133-13137 , nn/extra_modules/block.py:13139-13142 , nn/extra_modules/block.py:13603-13607 , nn/extra_modules/block.py:13609-13612 , nn/extra_modules/block.py:13614-13618 , nn/extra_modules/block.py:13620-13623 , nn/extra_modules/block.py:14796-14800 , nn/extra_modules/block.py:14802-14805 , nn/extra_modules/block.py:15359-15363 , nn/extra_modules/block.py:15365-15368 , nn/extra_modules/block.py:15404-15408 , nn/extra_modules/block.py:15410-15413 , nn/extra_modules/block.py:15415-15419 , nn/extra_modules/block.py:15421-15424 , nn/extra_modules/block.py:15430-15434 , nn/extra_modules/block.py:15436-15439 , nn/extra_modules/block.py:15445-15450 , nn/extra_modules/block.py:15452-15456 , nn/extra_modules/block.py:15546-15550 , nn/extra_modules/block.py:15552-15555 , nn/extra_modules/block.py:15988-15992 , nn/extra_modules/block.py:15994-15997 , nn/extra_modules/block.py:16102-16106 , nn/extra_modules/block.py:16108-16111 , nn/extra_modules/block.py:16117-16121 , nn/extra_modules/block.py:16123-16126 , nn/extra_modules/block.py:16132-16136 , nn/extra_modules/block.py:16138-16141 , nn/extra_modules/block.py:16147-16151 , nn/extra_modules/block.py:16153-16156 , nn/extra_modules/block.py:16214-16218 , nn/extra_modules/block.py:16220-16223 , nn/extra_modules/block.py:16225-16229 , nn/extra_modules/block.py:16231-16234 , nn/extra_modules/block.py:16240-16244 , nn/extra_modules/block.py:16246-16249 , nn/extra_modules/block.py:16266-16270 , nn/extra_modules/block.py:16272-16275 , nn/extra_modules/block.py:16277-16281 , nn/extra_modules/block.py:16283-16286 , nn/extra_modules/block.py:16292-16296 , nn/extra_modules/block.py:16298-16301 , nn/extra_modules/block.py:16307-16311 , nn/extra_modules/block.py:16313-16316 , nn/extra_modules/block.py:16322-16326 , nn/extra_modules/block.py:16328-16331 , nn/extra_modules/block.py:16359-16363 , nn/extra_modules/block.py:16365-16368 , nn/extra_modules/block.py:16415-16419 , nn/extra_modules/block.py:16421-16424 , nn/extra_modules/block.py:2489-2493 , nn/extra_modules/block.py:2495-2498 , nn/extra_modules/block.py:2739-2743 , nn/extra_modules/block.py:2745-2748 , nn/extra_modules/block.py:2927-2931 , nn/extra_modules/block.py:2933-2936 , nn/extra_modules/block.py:4579-4583 , nn/extra_modules/block.py:4585-4588 , nn/extra_modules/block.py:5036-5040 , nn/extra_modules/block.py:5042-5045 , nn/extra_modules/block.py:670-674 , nn/extra_modules/block.py:676-679 , nn/extra_modules/block.py:681-686 , nn/extra_modules/block.py:688-692 , nn/extra_modules/block.py:694-697 , nn/extra_modules/block.py:7304-7308 , nn/extra_modules/block.py:7311-7314 , nn/extra_modules/block.py:9838-9842 , nn/extra_modules/block.py:9844-9847
A1 组：0526 版 CSP 包装壳，内核在 2107 版中已存在，此处仅迁入壳本体。
签名 (c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True) 与 base_modules + repeat_modules 契约一致，
旧 yaml 可一字不改直接使用。（等价写法：C3k2_Block, [ch, {'module': <内核>}, ...]）
本组包含：C3k2_AP, C3k2_CAMixer, C3k2_CFBlock, C3k2_CNCM, C3k2_CSSC, C3k2_DBlock, C3k2_DRG, C3k2_DSEBlock, C3k2_DWR, C3k2_EBlock, C3k2_ELGCA, C3k2_EMBC, C3k2_ESC, C3k2_EVA, C3k2_EfficientVIM, C3k2_FAT, C3k2_FMB, C3k2_Faster, C3k2_FourierSR, C3k2_FrequencyCM, C3k2_GLGM, C3k2_HFRB, C3k2_IDWC, C3k2_IEL, C3k2_IRA, C3k2_JDPM, C3k2_LEGM, C3k2_LFE, C3k2_LaSEA, C3k2_MAC, C3k2_MSBlock, C3k2_MSInit, C3k2_PConv, C3k2_PFG, C3k2_PartialNetBlock, C3k2_RCB, C3k2_RFGM, C3k2_RVB, C3k2_SFEB, C3k2_SMB, C3k2_SPJFB, C3k2_Strip, C3k2_iRMB
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch

import torch.nn as nn


from ultralytics.nn.modules.conv import Conv


class MBConv(nn.Module):
    """C3k_EMBC / C3k2_EMBC 用的 MBConv：1x1升维 -> 3x3 depthwise -> 1x1投影，
    shortcut=True 且 inp==oup 时做残差相加。"""

    def __init__(self, inp, oup, shortcut=True, e=4, g=1):
        super().__init__()
        c_ = int(inp * e)
        self.cv1 = Conv(inp, c_, 1)
        self.cv2 = Conv(c_, c_, 3, g=c_)   # depthwise
        self.cv3 = Conv(c_, oup, 1, act=False)
        self.add = shortcut and inp == oup

    def forward(self, x):
        y = self.cv3(self.cv2(self.cv1(x)))
        return x + y if self.add else y


def _get_MBConv():
    """保留函数名以兼容下方两处调用点。"""
    return MBConv
from ultralytics.nn.extra_modules.module.APBottleneck import APBottleneck
from ultralytics.nn.extra_modules.module.CFBlock import CFBlock
from ultralytics.nn.extra_modules.module.CNCM import CNCM
from ultralytics.nn.extra_modules.module.CSSC import CSSC
from ultralytics.nn.extra_modules.module.DBlock import DBlock
from ultralytics.nn.extra_modules.module.DRG import DRG
from ultralytics.nn.extra_modules.module.DSEBlock import DSEBlock
from ultralytics.nn.extra_modules.module.DWR import DWR
from ultralytics.nn.extra_modules.module.EBlock import EBlock
from ultralytics.nn.extra_modules.module.ESC import ESCBlock
from ultralytics.nn.extra_modules.module.EVA import EVA
from ultralytics.nn.extra_modules.module.FATBlock import FAT_Block
from ultralytics.nn.extra_modules.module.FMB import FMB
from ultralytics.nn.extra_modules.module.FourierSR import FourierSR
from ultralytics.nn.extra_modules.module.FrequencyCM import FrequencyCM
from ultralytics.nn.extra_modules.module.GLGM import GLGM
from ultralytics.nn.extra_modules.module.HFRB import HFRB
from ultralytics.nn.extra_modules.module.IDWB import InceptionDWConv2d
from ultralytics.nn.extra_modules.module.IEL import IEL
from ultralytics.nn.extra_modules.module.IRA import IRA
from ultralytics.nn.extra_modules.module.JDPM import JDPM
from ultralytics.nn.extra_modules.module.LEGM import LEGM
from ultralytics.nn.extra_modules.module.LFE import LFE
from ultralytics.nn.extra_modules.module.LaSEA import LaSEA
from ultralytics.nn.extra_modules.module.MAC import MAC
from ultralytics.nn.extra_modules.module.MSBlock import MSBlock
from ultralytics.nn.extra_modules.module.MSInit import MSInit
from ultralytics.nn.extra_modules.module.PFG import PFG
from ultralytics.nn.extra_modules.module.PartialNetBlock import PartialNetBlock
from ultralytics.nn.extra_modules.module.RCB import RepConvBlock
from ultralytics.nn.extra_modules.module.RFGM import RFGM
from ultralytics.nn.extra_modules.module.RepViTBlock import RepViTBlock
from ultralytics.nn.extra_modules.module.SFEB import SFEB
from ultralytics.nn.extra_modules.module.SPJFB import SPJFrequencyBlock
from ultralytics.nn.extra_modules.module.StripBlock import StripBlock
from ultralytics.nn.extra_modules.module.camixer import CAMixer
from ultralytics.nn.extra_modules.module.efficientVIM import EfficientViMBlock
from ultralytics.nn.extra_modules.module.elgca import ELGCA_EncoderBlock
from ultralytics.nn.extra_modules.module.fasterblock import Faster_Block, Partial_conv3
from ultralytics.nn.extra_modules.module.iRMB import iRMB
from ultralytics.nn.extra_modules.module.sparse_mamba_block import SparseMambaBlock
from ultralytics.nn.modules.block import Bottleneck, C3k, C3k2


# ---- 原样迁移自 nn/extra_modules/block.py:10578-10583 ----
class Bottleneck_IDWC(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = InceptionDWConv2d(c1)
        self.cv2 = InceptionDWConv2d(c2)

# ---- 原样迁移自 nn/extra_modules/block.py:681-686 ----
class Bottleneck_PConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Partial_conv3(c1)
        self.cv2 = Partial_conv3(c2)

# ---- 原样迁移自 nn/extra_modules/block.py:11443-11447 ----
class C3k_AP(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(APBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:11449-11452 ----
class C3k2_AP(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_AP(self.c, self.c, 2, shortcut, g) if c3k else APBottleneck(self.c, self.c, shortcut, g) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:10612-10616 ----
class C3k_CAMixer(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(CAMixer(c_, window_size=4) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:10618-10621 ----
class C3k2_CAMixer(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_CAMixer(self.c, self.c, 2, shortcut, g) if c3k else CAMixer(self.c, window_size=4) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:15359-15363 ----
class C3k_CFBlock(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(CFBlock(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:15365-15368 ----
class C3k2_CFBlock(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_CFBlock(self.c, self.c, 2, shortcut, g) if c3k else CFBlock(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:15415-15419 ----
class C3k_CNCM(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(CNCM(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:15421-15424 ----
class C3k2_CNCM(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_CNCM(self.c, self.c, 2, shortcut, g) if c3k else CNCM(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:15404-15408 ----
class C3k_CSSC(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(CSSC(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:15410-15413 ----
class C3k2_CSSC(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_CSSC(self.c, self.c, 2, shortcut, g) if c3k else CSSC(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:13614-13618 ----
class C3k_DBlock(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DBlock(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:13620-13623 ----
class C3k2_DBlock(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DBlock(self.c, self.c, n, shortcut, g) if c3k else DBlock(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:16102-16106 ----
class C3k_DRG(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DRG(c_, c_, 3, 1) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:16108-16111 ----
class C3k2_DRG(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DRG(self.c, self.c, 2, shortcut, g) if c3k else DRG(self.c, self.c, 3, 1) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:16266-16270 ----
class C3k_DSEBlock(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DSEBlock(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:16272-16275 ----
class C3k2_DSEBlock(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DSEBlock(self.c, self.c, 2, shortcut, g) if c3k else DSEBlock(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:2927-2931 ----
class C3k_DWR(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DWR(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:2933-2936 ----
class C3k2_DWR(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DWR(self.c, self.c, 2, shortcut, g) if c3k else DWR(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:13603-13607 ----
class C3k_EBlock(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(EBlock(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:13609-13612 ----
class C3k2_EBlock(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_EBlock(self.c, self.c, n, shortcut, g) if c3k else EBlock(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:11914-11918 ----
class C3k_ELGCA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(ELGCA_EncoderBlock(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:11920-11923 ----
class C3k2_ELGCA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_ELGCA(self.c, self.c, 2, shortcut, g) if c3k else ELGCA_EncoderBlock(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:2739-2743 ----
class C3k_EMBC(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        MBConv = _get_MBConv()
        self.m = nn.Sequential(*(MBConv(c_, c_, shortcut) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:2745-2748 ----
class C3k2_EMBC(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        MBConv = _get_MBConv()
        self.m = nn.ModuleList(C3k_EMBC(self.c, self.c, 2, shortcut, g) if c3k else MBConv(self.c, self.c, shortcut) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:14796-14800 ----
class C3k_ESC(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(ESCBlock(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:14802-14805 ----
class C3k2_ESC(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_ESC(self.c, self.c, 2, shortcut, g) if c3k else ESCBlock(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:15445-15450 ----
class C3k_EVA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        # kernel_size可选7,11,23,35
        self.m = nn.Sequential(*(EVA(c_, kernel_size=7) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:15452-15456 ----
class C3k2_EVA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        # kernel_size可选7,11,23,35
        self.m = nn.ModuleList(C3k_EVA(self.c, self.c, 2, shortcut, g) if c3k else EVA(self.c, kernel_size=7) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:12894-12898 ----
class C3k_EfficientVIM(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(EfficientViMBlock(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:12900-12903 ----
class C3k2_EfficientVIM(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_EfficientVIM(self.c, self.c, n, shortcut, g) if c3k else EfficientViMBlock(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:13057-13061 ----
class C3k_FAT(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(FAT_Block(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:13063-13066 ----
class C3k2_FAT(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_FAT(self.c, self.c, n, shortcut, g) if c3k else FAT_Block(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:7304-7308 ----
class C3k_FMB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(FMB(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:7311-7314 ----
class C3k2_FMB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_FMB(self.c, self.c, 2, shortcut, n) if c3k else FMB(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:670-674 ----
class C3k_Faster(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Faster_Block(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:676-679 ----
class C3k2_Faster(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_Faster(self.c, self.c, 2, shortcut, g) if c3k else Faster_Block(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:16307-16311 ----
class C3k_FourierSR(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(FourierSR(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:16313-16316 ----
class C3k2_FourierSR(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_FourierSR(self.c, self.c, 2, shortcut, g) if c3k else FourierSR(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:16322-16326 ----
class C3k_FrequencyCM(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(FrequencyCM(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:16328-16331 ----
class C3k2_FrequencyCM(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_FrequencyCM(self.c, self.c, 2, shortcut, g) if c3k else FrequencyCM(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:16117-16121 ----
class C3k_GLGM(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GLGM(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:16123-16126 ----
class C3k2_GLGM(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_GLGM(self.c, self.c, 2, shortcut, g) if c3k else GLGM(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:15430-15434 ----
class C3k_HFRB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(HFRB(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:15436-15439 ----
class C3k2_HFRB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_HFRB(self.c, self.c, 2, shortcut, g) if c3k else HFRB(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:10585-10589 ----
class C3k_IDWC(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_IDWC(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:10591-10594 ----
class C3k2_IDWC(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_IDWC(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_IDWC(self.c, self.c, shortcut, g) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:15546-15550 ----
class C3k_IEL(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(IEL(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:15552-15555 ----
class C3k2_IEL(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_IEL(self.c, self.c, 2, shortcut, g) if c3k else IEL(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:16359-16363 ----
class C3k_IRA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(IRA(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:16365-16368 ----
class C3k2_IRA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_IRA(self.c, self.c, 2, shortcut, g) if c3k else IRA(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:11012-11016 ----
class C3k_JDPM(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(JDPM(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:11018-11021 ----
class C3k2_JDPM(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_JDPM(self.c, self.c, 2, shortcut, g) if c3k else JDPM(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:13118-13122 ----
class C3k_LEGM(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(LEGM(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:13124-13127 ----
class C3k2_LEGM(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_LEGM(self.c, self.c, n, shortcut, g) if c3k else LEGM(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:9838-9842 ----
class C3k_LFE(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(LFE(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:9844-9847 ----
class C3k2_LFE(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_LFE(self.c, self.c, 2, shortcut, g) if c3k else LFE(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:16277-16281 ----
class C3k_LaSEA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(LaSEA(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:16283-16286 ----
class C3k2_LaSEA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_LaSEA(self.c, self.c, 2, shortcut, g) if c3k else LaSEA(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:16132-16136 ----
class C3k_MAC(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MAC(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:16138-16141 ----
class C3k2_MAC(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MAC(self.c, self.c, 2, shortcut, g) if c3k else MAC(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:2489-2493 ----
class C3k_MSBlock(C3k):
    def __init__(self, c1, c2, n=1, kernel_sizes=[1, 3, 3], in_expand_ratio=3., mid_expand_ratio=2., layers_num=3, in_down_ratio=2., shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MSBlock(c_, c_, kernel_sizes, in_expand_ratio, mid_expand_ratio, layers_num, in_down_ratio) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:2495-2498 ----
class C3k2_MSBlock(C3k2):
    def __init__(self, c1, c2, n=1, kernel_sizes=[1, 3, 3], in_expand_ratio=3., mid_expand_ratio=2., layers_num=3, in_down_ratio=2., c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MSBlock(self.c, self.c, 2, kernel_sizes, in_expand_ratio, mid_expand_ratio, layers_num, in_down_ratio, shortcut, g) if c3k else MSBlock(self.c, self.c, kernel_sizes, in_expand_ratio, mid_expand_ratio, layers_num, in_down_ratio) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:16225-16229 ----
class C3k_MSInit(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(MSInit(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:16231-16234 ----
class C3k2_MSInit(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MSInit(self.c, self.c, 2, shortcut, g) if c3k else MSInit(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:688-692 ----
class C3k_PConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_PConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:694-697 ----
class C3k2_PConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_PConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_PConv(self.c, self.c, shortcut, g) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:16214-16218 ----
class C3k_PFG(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(PFG(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:16220-16223 ----
class C3k2_PFG(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_PFG(self.c, self.c, 2, shortcut, g) if c3k else PFG(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:15988-15992 ----
class C3k_PartialNetBlock(C3k):
    def __init__(self, c1, c2, n=1, version='v1', shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(PartialNetBlock(c_, c_, version) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:15994-15997 ----
class C3k2_PartialNetBlock(C3k2):
    def __init__(self, c1, c2, n=1, version='v1', c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_PartialNetBlock(self.c, self.c, 2, version, shortcut, g) if c3k else PartialNetBlock(self.c, self.c, version) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:13133-13137 ----
class C3k_RCB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepConvBlock(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:13139-13142 ----
class C3k2_RCB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_RCB(self.c, self.c, n, shortcut, g) if c3k else RepConvBlock(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:16415-16419 ----
class C3k_RFGM(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RFGM(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:16421-16424 ----
class C3k2_RFGM(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_RFGM(self.c, self.c, 2, shortcut, g) if c3k else RFGM(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:5036-5040 ----
class C3k_RVB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepViTBlock(c_, c_, False) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:5042-5045 ----
class C3k2_RVB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_RVB(self.c, self.c, 2, shortcut, g) if c3k else RepViTBlock(self.c, self.c, False) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:16292-16296 ----
class C3k_SFEB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(SFEB(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:16298-16301 ----
class C3k2_SFEB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_SFEB(self.c, self.c, 2, shortcut, g) if c3k else SFEB(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:16240-16244 ----
class C3k_SMB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(SparseMambaBlock(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:16246-16249 ----
class C3k2_SMB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_SMB(self.c, self.c, 2, shortcut, g) if c3k else SparseMambaBlock(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:16147-16151 ----
class C3k_SPJFB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(SPJFrequencyBlock(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:16153-16156 ----
class C3k2_SPJFB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_SPJFB(self.c, self.c, 2, shortcut, g) if c3k else C3k_SPJFB(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:12039-12043 ----
class C3k_Strip(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(StripBlock(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:12045-12048 ----
class C3k2_Strip(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_Strip(self.c, self.c, 2, shortcut, g) if c3k else StripBlock(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:4579-4583 ----
class C3k_iRMB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(iRMB(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:4585-4588 ----
class C3k2_iRMB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_iRMB(self.c, self.c, 2, shortcut, g) if c3k else iRMB(self.c, self.c) for _ in range(n))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- C3k2_Strip ----
    try:
        module = C3k2_Strip(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_Strip  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_Strip  自测跳过: {e}' + RESET)
    # ---- C3k_iRMB ----
    try:
        module = C3k_iRMB(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k_iRMB  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k_iRMB  自测跳过: {e}' + RESET)
    # ---- C3k2_iRMB ----
    try:
        module = C3k2_iRMB(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_iRMB  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_iRMB  自测跳过: {e}' + RESET)

