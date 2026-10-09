'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/backbone/lsnet.py:397-405 , nn/extra_modules/block.py:12843-12877 , nn/extra_modules/block.py:12879-12883 , nn/extra_modules/block.py:12885-12888 , nn/extra_modules/block.py:13478-13512 , nn/extra_modules/block.py:13514-13518 , nn/extra_modules/block.py:13520-13523 , nn/extra_modules/block.py:13648-13682 , nn/extra_modules/block.py:13684-13688 , nn/extra_modules/block.py:13690-13693 , nn/extra_modules/block.py:13833-13867 , nn/extra_modules/block.py:13869-13873 , nn/extra_modules/block.py:13875-13878 , nn/extra_modules/block.py:13980-14014 , nn/extra_modules/block.py:14016-14020 , nn/extra_modules/block.py:14022-14025
二次创新(2)：CVPR2025 MambaOut 的 GatedCNNBlock 分别融合 UniRepLKNet / LSNet 的 LSConv /
CVPR2025 FDConv / DSA / SFSConv（旧库 md #290/#319/#334/#338/#347）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

from timm.layers import DropPath
import torch.nn as nn
from functools import partial
import torch
from ultralytics.nn.extra_modules.conv_module.LSConv_kernel import SKA
from ultralytics.nn.extra_modules.conv_module.LSConv_kernel import LKP
from ultralytics.nn.extra_modules.block.RepHMS import UniRepLKNetBlock
from ultralytics.nn.extra_modules.conv_module.DSA import DSA
from ultralytics.nn.extra_modules.conv_module.FDConv import FDConv
from ultralytics.nn.extra_modules.conv_module.SFSConv import SFS_Conv
from ultralytics.nn.extra_modules.module.mambaout import LayerNormGeneral
from ultralytics.nn.modules.block import C3k, C3k2


# ---- 原样迁移自 nn/extra_modules/block.py:13833-13867 ----
class GatedDSABlock_BCHW(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args: 
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve practical efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and 
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(LayerNormGeneral,eps=1e-6,normalized_dim=(1, 2, 3)), 
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        self.norm = norm_layer((dim, 1, 1))
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 1)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        # self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.conv = DSA(conv_channels, kernel_size=kernel_size)
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x # [B, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        # c = c.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        # c = c.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=1))
        x = self.drop_path(x)
        return x + shortcut

# ---- 原样迁移自 nn/extra_modules/block.py:13869-13873 ----
class C3k_MambaOut_DSA(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GatedDSABlock_BCHW(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:13875-13878 ----
class C3k2_MambaOut_DSA(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MambaOut_DSA(self.c, self.c, n, shortcut, g) if c3k else GatedDSABlock_BCHW(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:13648-13682 ----
class GatedFDConvBlock_BCHW(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args: 
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve practical efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and 
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(LayerNormGeneral,eps=1e-6,normalized_dim=(1, 2, 3)), 
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        self.norm = norm_layer((dim, 1, 1))
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 1)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        # self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.conv = FDConv(conv_channels, conv_channels, kernel_size=kernel_size)
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x # [B, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        # c = c.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        # c = c.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=1))
        x = self.drop_path(x)
        return x + shortcut

# ---- 原样迁移自 nn/extra_modules/block.py:13684-13688 ----
class C3k_MambaOut_FDConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GatedFDConvBlock_BCHW(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:13690-13693 ----
class C3k2_MambaOut_FDConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MambaOut_FDConv(self.c, self.c, n, shortcut, g) if c3k else GatedFDConvBlock_BCHW(self.c) for _ in range(n))

# ---- 原样迁移自 nn/backbone/lsnet.py:397-405 ----
class LSConv(nn.Module):
    def __init__(self, dim):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska(x, self.lkp(x))) + x

# ---- 原样迁移自 nn/extra_modules/block.py:13478-13512 ----
class GatedLSBlock_BCHW(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args: 
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve practical efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and 
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(LayerNormGeneral,eps=1e-6,normalized_dim=(1, 2, 3)), 
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        self.norm = norm_layer((dim, 1, 1))
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 1)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        # self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.conv = LSConv(conv_channels)
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x # [B, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        # c = c.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        # c = c.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=1))
        x = self.drop_path(x)
        return x + shortcut

# ---- 原样迁移自 nn/extra_modules/block.py:13514-13518 ----
class C3k_MambaOut_LSConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GatedLSBlock_BCHW(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:13520-13523 ----
class C3k2_MambaOut_LSConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MambaOut_LSConv(self.c, self.c, n, shortcut, g) if c3k else GatedLSBlock_BCHW(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:13980-14014 ----
class GatedSFSCBlock_BCHW(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args: 
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve practical efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and 
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(LayerNormGeneral,eps=1e-6,normalized_dim=(1, 2, 3)), 
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        self.norm = norm_layer((dim, 1, 1))
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 1)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        # self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.conv = SFS_Conv(conv_channels, conv_channels)
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x # [B, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        # c = c.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        # c = c.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=1))
        x = self.drop_path(x)
        return x + shortcut

# ---- 原样迁移自 nn/extra_modules/block.py:14016-14020 ----
class C3k_MambaOut_SFSC(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GatedSFSCBlock_BCHW(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:14022-14025 ----
class C3k2_MambaOut_SFSC(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MambaOut_SFSC(self.c, self.c, n, shortcut, g) if c3k else GatedSFSCBlock_BCHW(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:12843-12877 ----
class GatedUniRepLKBlock_BCHW(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args: 
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve practical efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and 
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(LayerNormGeneral,eps=1e-6,normalized_dim=(1, 2, 3)), 
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        self.norm = norm_layer((dim, 1, 1))
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 1)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        # self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.conv = UniRepLKNetBlock(conv_channels, kernel_size=kernel_size)
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x # [B, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        # c = c.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        # c = c.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=1))
        x = self.drop_path(x)
        return x + shortcut

# ---- 原样迁移自 nn/extra_modules/block.py:12879-12883 ----
class C3k_MambaOut_UniRepLK(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GatedUniRepLKBlock_BCHW(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:12885-12888 ----
class C3k2_MambaOut_UniRepLK(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_MambaOut_UniRepLK(self.c, self.c, n, shortcut, g) if c3k else GatedUniRepLKBlock_BCHW(self.c) for _ in range(n))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- GatedUniRepLKBlock_BCHW ----
    try:
        module = GatedUniRepLKBlock_BCHW(in_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'GatedUniRepLKBlock_BCHW  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'GatedUniRepLKBlock_BCHW  自测跳过: {e}' + RESET)
    # ---- C3k_MambaOut_UniRepLK ----
    try:
        module = C3k_MambaOut_UniRepLK(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k_MambaOut_UniRepLK  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k_MambaOut_UniRepLK  自测跳过: {e}' + RESET)
    # ---- C3k2_MambaOut_UniRepLK ----
    try:
        module = C3k2_MambaOut_UniRepLK(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_MambaOut_UniRepLK  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_MambaOut_UniRepLK  自测跳过: {e}' + RESET)

