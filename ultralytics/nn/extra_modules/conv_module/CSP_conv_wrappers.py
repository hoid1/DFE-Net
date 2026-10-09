'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:1003-1006 , nn/extra_modules/block.py:1008-1013 , nn/extra_modules/block.py:1015-1019 , nn/extra_modules/block.py:1021-1024 , nn/extra_modules/block.py:1315-1320 , nn/extra_modules/block.py:1322-1326 , nn/extra_modules/block.py:1328-1331 , nn/extra_modules/block.py:13629-13634 , nn/extra_modules/block.py:13636-13640 , nn/extra_modules/block.py:13642-13645 , nn/extra_modules/block.py:13880-13885 , nn/extra_modules/block.py:13887-13891 , nn/extra_modules/block.py:13893-13896 , nn/extra_modules/block.py:13962-13967 , nn/extra_modules/block.py:13969-13973 , nn/extra_modules/block.py:13975-13978 , nn/extra_modules/block.py:14759-14764 , nn/extra_modules/block.py:14766-14770 , nn/extra_modules/block.py:14772-14775 , nn/extra_modules/block.py:14807-14812 , nn/extra_modules/block.py:14814-14818 , nn/extra_modules/block.py:14820-14823 , nn/extra_modules/block.py:1525-1535 , nn/extra_modules/block.py:15310-15315 , nn/extra_modules/block.py:15317-15321 , nn/extra_modules/block.py:15323-15326 , nn/extra_modules/block.py:15332-15337 , nn/extra_modules/block.py:15339-15343 , nn/extra_modules/block.py:15345-15348 , nn/extra_modules/block.py:1537-1541 , nn/extra_modules/block.py:1543-1546 , nn/extra_modules/block.py:15462-15474 , nn/extra_modules/block.py:15491-15496 , nn/extra_modules/block.py:15498-15502 , nn/extra_modules/block.py:16177-16182 , nn/extra_modules/block.py:1618-1624 , nn/extra_modules/block.py:16184-16188 , nn/extra_modules/block.py:16190-16193 , nn/extra_modules/block.py:1626-1630 , nn/extra_modules/block.py:1632-1635 , nn/extra_modules/block.py:16337-16342 , nn/extra_modules/block.py:16344-16348 , nn/extra_modules/block.py:16350-16353 , nn/extra_modules/block.py:4250-4256 , nn/extra_modules/block.py:4258-4262 , nn/extra_modules/block.py:4264-4267 , nn/extra_modules/block.py:4973-4977 , nn/extra_modules/block.py:4979-4983 , nn/extra_modules/block.py:4985-4988 , nn/extra_modules/block.py:5287-5293 , nn/extra_modules/block.py:5295-5299 , nn/extra_modules/block.py:5301-5304 , nn/extra_modules/block.py:6236-6243 , nn/extra_modules/block.py:6245-6249 , nn/extra_modules/block.py:6251-6254 , nn/extra_modules/block.py:7020-7025 , nn/extra_modules/block.py:7027-7031 , nn/extra_modules/block.py:7033-7036 , nn/extra_modules/block.py:972-977 , nn/extra_modules/block.py:979-983 , nn/extra_modules/block.py:985-988 , nn/extra_modules/block.py:990-995 , nn/extra_modules/block.py:997-1001
A1 组：0526 版 CSP 包装壳，内核在 2107 版中已存在，此处仅迁入壳本体。
签名 (c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True) 与 base_modules + repeat_modules 契约一致，
旧 yaml 可一字不改直接使用。（等价写法：C3k2_Block, [ch, {'module': <内核>}, ...]）
本组包含：C3k2_CKConv, C3k2_ConvAttn, C3k2_Converse, C3k2_DBB, C3k2_DCNv2, C3k2_DEConv, C3k2_DEGConv, C3k2_DSA, C3k2_DeepDBB, C3k2_DySnakeConv, C3k2_DynamicConv, C3k2_FADC, C3k2_FDConv, C3k2_FourierConv, C3k2_GCConv, C3k2_RMBC, C3k2_SFSConv, C3k2_SWC, C3k2_ScConv, C3k2_WDBB, C3k2_WTConv
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch

import torch.nn as nn
from ultralytics.nn.extra_modules.conv_module.CKConv import CKConv
from ultralytics.nn.extra_modules.conv_module.ConvAttn import ConvAttn
from ultralytics.nn.extra_modules.conv_module.Converse2D import Converse2D
from ultralytics.nn.extra_modules.conv_module.DEGConv import DEGConv
from ultralytics.nn.extra_modules.conv_module.DSA import DSA
from ultralytics.nn.extra_modules.conv_module.DynamicConv import DynamicConv
from ultralytics.nn.extra_modules.conv_module.FADC import AdaptiveDilatedConv
from ultralytics.nn.extra_modules.conv_module.FDConv import FDConv
from ultralytics.nn.extra_modules.conv_module.FourierConv import FourierConv
from ultralytics.nn.extra_modules.conv_module.RMBC import RepMBConv
from ultralytics.nn.extra_modules.conv_module.SFSConv import SFS_Conv
from ultralytics.nn.extra_modules.conv_module.ScConv import ScConv
from ultralytics.nn.extra_modules.conv_module.ShiftwiseConv import ReparamLargeKernelConv
from ultralytics.nn.extra_modules.conv_module.dbb import DiverseBranchBlock
from ultralytics.nn.extra_modules.conv_module.dcnv2 import DCNv2
from ultralytics.nn.extra_modules.conv_module.deconv import DEConv
from ultralytics.nn.extra_modules.conv_module.deepdbb import DeepDiverseBranchBlock
from ultralytics.nn.extra_modules.conv_module.dynamic_snake_conv import DySnakeConv
from ultralytics.nn.extra_modules.conv_module.gcconv import GCConv
from ultralytics.nn.extra_modules.conv_module.wdbb import WideDiverseBranchBlock
from ultralytics.nn.extra_modules.conv_module.wtconv2d import WTConv2d
from ultralytics.nn.modules.block import Bottleneck, C3k, C3k2
from ultralytics.nn.modules.conv import Conv


# ---- 原样迁移自 nn/extra_modules/block.py:16337-16342 ----
class Bottleneck_CKConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = CKConv(c1, c1)
        self.cv2 = CKConv(c2, c2)

# ---- 原样迁移自 nn/extra_modules/block.py:14807-14812 ----
class Bottleneck_ConvAttn(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvAttn(c1)
        self.cv2 = ConvAttn(c2)

# ---- 原样迁移自 nn/extra_modules/block.py:15310-15315 ----
class Bottleneck_Converse(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Converse2D(c1, c1, 3)
        self.cv2 = Converse2D(c2, c2, 3)

# ---- 原样迁移自 nn/extra_modules/block.py:972-977 ----
class Bottleneck_DBB(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DiverseBranchBlock(c1, c_, k[0], 1)
        self.cv2 = DiverseBranchBlock(c_, c2, k[1], 1, groups=g)

# ---- 原样迁移自 nn/extra_modules/block.py:1618-1624 ----
class Bottleneck_DCNV2(Bottleneck):
    """Standard bottleneck with DCNV2."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DCNv2(c_, c2, k[1], 1)

# ---- 原样迁移自 nn/extra_modules/block.py:6236-6243 ----
class Bottleneck_DEConv(Bottleneck):
    """Standard bottleneck with DCNV3."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        # self.cv1 = DEConv(c_)
        self.cv2 = DEConv(c_)

# ---- 原样迁移自 nn/extra_modules/block.py:16177-16182 ----
class Bottleneck_DEGConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DEGConv(c1, c1)
        self.cv2 = DEGConv(c2, c2)

# ---- 原样迁移自 nn/extra_modules/block.py:13880-13885 ----
class Bottleneck_DSA(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        # self.cv1 = DSA(c1)
        self.cv2 = DSA(c2)

# ---- 原样迁移自 nn/extra_modules/block.py:1008-1013 ----
class Bottleneck_DeepDBB(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DeepDiverseBranchBlock(c1, c_, k[0], 1)
        self.cv2 = DeepDiverseBranchBlock(c_, c2, k[1], 1, groups=g)

# ---- 原样迁移自 nn/extra_modules/block.py:1525-1535 ----
class Bottleneck_DySnakeConv(Bottleneck):
    """Standard bottleneck with DySnakeConv."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DySnakeConv(c_, c2, k[1])
        self.cv3 = Conv(c2 * 3, c2, k=1)
    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))

# ---- 原样迁移自 nn/extra_modules/block.py:4973-4977 ----
class Bottleneck_DynamicConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DynamicConv(c2, c2, 3)

# ---- 原样迁移自 nn/extra_modules/block.py:5287-5293 ----
class Bottleneck_FADC(Bottleneck):
    """Standard bottleneck with FADC."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = AdaptiveDilatedConv(in_channels=c_, out_channels=c2, kernel_size=k[1], stride=1, padding=1)

# ---- 原样迁移自 nn/extra_modules/block.py:13629-13634 ----
class Bottleneck_FDConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = FDConv(c1, c_)
        self.cv2 = FDConv(c_, c2)

# ---- 原样迁移自 nn/extra_modules/block.py:14759-14764 ----
class Bottleneck_FourierConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), size=None, e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = FourierConv(c1, c_, size)
        self.cv2 = FourierConv(c_, c2, size)

# ---- 原样迁移自 nn/extra_modules/block.py:15332-15337 ----
class Bottleneck_GCConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = GCConv(c1, c1, 3)
        self.cv2 = GCConv(c2, c2, 3)

# ---- 原样迁移自 nn/extra_modules/block.py:15462-15474 ----
class Bottleneck_RepMBConv(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        self.cv1 = RepMBConv(c1)
        self.cv2 = RepMBConv(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

# ---- 原样迁移自 nn/extra_modules/block.py:13962-13967 ----
class Bottleneck_SFSConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = SFS_Conv(c1, c_)
        self.cv2 = SFS_Conv(c_, c2)

# ---- 原样迁移自 nn/extra_modules/block.py:4250-4256 ----
class Bottleneck_SWC(Bottleneck):
    """Standard bottleneck with DilatedReparamBlock."""

    def __init__(self, c1, c2, kernel_size, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = ReparamLargeKernelConv(c2, c2, kernel_size, groups=(c2 // 16))

# ---- 原样迁移自 nn/extra_modules/block.py:1315-1320 ----
class Bottleneck_ScConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = ScConv(c2)

# ---- 原样迁移自 nn/extra_modules/block.py:990-995 ----
class Bottleneck_WDBB(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = WideDiverseBranchBlock(c1, c_, k[0], 1)
        self.cv2 = WideDiverseBranchBlock(c_, c2, k[1], 1, groups=g)

# ---- 原样迁移自 nn/extra_modules/block.py:7020-7025 ----
class Bottleneck_WTConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        # self.cv1 = WTConv2d(c1, c2)
        self.cv2 = WTConv2d(c2, c2)

# ---- 原样迁移自 nn/extra_modules/block.py:16344-16348 ----
class C3k_CKConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_CKConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:16350-16353 ----
class C3k2_CKConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_CKConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_CKConv(self.c, self.c, shortcut) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:14814-14818 ----
class C3k_ConvAttn(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_ConvAttn(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:14820-14823 ----
class C3k2_ConvAttn(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_ConvAttn(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_ConvAttn(self.c, self.c, shortcut) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:15317-15321 ----
class C3k_Converse(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_Converse(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:15323-15326 ----
class C3k2_Converse(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_Converse(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_Converse(self.c, self.c, shortcut) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:979-983 ----
class C3k_DBB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DBB(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:985-988 ----
class C3k2_DBB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DBB(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DBB(self.c, self.c, shortcut, g) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:1626-1630 ----
class C3k_DCNv2(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DCNV2(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:1632-1635 ----
class C3k2_DCNv2(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DCNv2(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DCNV2(self.c, self.c, shortcut, g) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:6245-6249 ----
class C3k_DEConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DEConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:6251-6254 ----
class C3k2_DEConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DEConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DEConv(self.c, self.c, shortcut, g) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:16184-16188 ----
class C3k_DEGConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DEGConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:16190-16193 ----
class C3k2_DEGConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DEGConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DEGConv(self.c, self.c, shortcut) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:13887-13891 ----
class C3k_DSA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DSA(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:13893-13896 ----
class C3k2_DSA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DSA(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DSA(self.c, self.c, shortcut, g) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:1015-1019 ----
class C3k_DeepDBB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DeepDBB(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:1021-1024 ----
class C3k2_DeepDBB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DeepDBB(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DeepDBB(self.c, self.c, shortcut, g) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:1537-1541 ----
class C3k_DySnakeConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DySnakeConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:1543-1546 ----
class C3k2_DySnakeConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DySnakeConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DySnakeConv(self.c, self.c, shortcut, g) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:4979-4983 ----
class C3k_DynamicConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DynamicConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:4985-4988 ----
class C3k2_DynamicConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DynamicConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DynamicConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:5295-5299 ----
class C3k_FADC(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_FADC(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:5301-5304 ----
class C3k2_FADC(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_FADC(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_FADC(self.c, self.c, shortcut, g) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:13636-13640 ----
class C3k_FDConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_FDConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:13642-13645 ----
class C3k2_FDConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_FDConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_FDConv(self.c, self.c, shortcut, g) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:14766-14770 ----
class C3k_FourierConv(C3k):
    def __init__(self, c1, c2, n=1, feat_size=None, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_FourierConv(c_, c_, shortcut, g, k=(k, k), size=feat_size, e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:14772-14775 ----
class C3k2_FourierConv(C3k2):
    def __init__(self, c1, c2, n=1, feat_size=None, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_FourierConv(self.c, self.c, 2, feat_size, shortcut, g) if c3k else Bottleneck_FourierConv(self.c, self.c, shortcut, g, size=feat_size) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:15339-15343 ----
class C3k_GCConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_GCConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:15345-15348 ----
class C3k2_GCConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_GCConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_GCConv(self.c, self.c, shortcut) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:15491-15496 ----
class C3k_RMBC(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        # kernel_size可选7,11,23,35
        self.m = nn.Sequential(*(Bottleneck_RepMBConv(c_, c_, shortcut=shortcut, g=g) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:15498-15502 ----
class C3k2_RMBC(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        # kernel_size可选7,11,23,35
        self.m = nn.ModuleList(C3k_RMBC(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_RepMBConv(self.c, self.c, shortcut=shortcut, g=g) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:13969-13973 ----
class C3k_SFSConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_SFSConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:13975-13978 ----
class C3k2_SFSConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_SFSConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_SFSConv(self.c, self.c, shortcut, g) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:4258-4262 ----
class C3k_SWC(C3k):
    def __init__(self, c1, c2, n=1, kernel_size=13, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_SWC(c_, c_, kernel_size, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:4264-4267 ----
class C3k2_SWC(C3k2):
    def __init__(self, c1, c2, n=1, kernel_size=13, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_SWC(self.c, self.c, 2, kernel_size, shortcut, g) if c3k else Bottleneck_SWC(self.c, self.c, kernel_size, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:1322-1326 ----
class C3k_ScConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_ScConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:1328-1331 ----
class C3k2_ScConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_ScConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_ScConv(self.c, self.c, shortcut, g) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:997-1001 ----
class C3k_WDBB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_WDBB(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:1003-1006 ----
class C3k2_WDBB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_WDBB(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_WDBB(self.c, self.c, shortcut, g) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:7027-7031 ----
class C3k_WTConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_WTConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:7033-7036 ----
class C3k2_WTConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_WTConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_WTConv(self.c, self.c, shortcut, g) for _ in range(n))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- C3k2_WDBB ----
    try:
        module = C3k2_WDBB(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_WDBB  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_WDBB  自测跳过: {e}' + RESET)
    # ---- C3k_WTConv ----
    try:
        module = C3k_WTConv(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k_WTConv  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k_WTConv  自测跳过: {e}' + RESET)
    # ---- C3k2_WTConv ----
    try:
        module = C3k2_WTConv(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_WTConv  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_WTConv  自测跳过: {e}' + RESET)

