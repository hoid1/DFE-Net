'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:13695-13720 , nn/extra_modules/block.py:13722-13776 , nn/extra_modules/block.py:13778-13783 , nn/extra_modules/block.py:13785-13789 , nn/extra_modules/block.py:13791-13794 , nn/extra_modules/block.py:13796-13800 , nn/extra_modules/block.py:13802-13805
二次创新(2)：CVPR2023 PConv + CVPR2025 FDConv（旧库 md #335/#336）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

from timm.layers import DropPath
import torch.nn as nn
import torch
from ultralytics.nn.extra_modules.conv_module.FDConv import FDConv
from ultralytics.nn.modules.block import Bottleneck, C3k, C3k2
from ultralytics.nn.modules.conv import Conv


# ---- 原样迁移自 nn/extra_modules/block.py:13695-13720 ----
class Partial_FDConv(nn.Module):
    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = FDConv(self.dim_conv3, self.dim_conv3, 3, 1)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:13778-13783 ----
class Bottleneck_PFDConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Partial_FDConv(c1)
        self.cv2 = Partial_FDConv(c2)

# ---- 原样迁移自 nn/extra_modules/block.py:13722-13776 ----
class FasterFDConv(nn.Module):
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

        self.spatial_mixing = Partial_FDConv(
            dim,
            n_div,
            pconv_fw_type
        )
        
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
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x

# ---- 原样迁移自 nn/extra_modules/block.py:13796-13800 ----
class C3k_FasterFD(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(FasterFDConv(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:13802-13805 ----
class C3k2_FasterFD(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_FasterFD(self.c, self.c, 2, shortcut, g) if c3k else FasterFDConv(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:13785-13789 ----
class C3k_PFDConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_PFDConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:13791-13794 ----
class C3k2_PFDConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_PFDConv(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_PFDConv(self.c, self.c, shortcut, g) for _ in range(n))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- C3k2_FasterFD ----
    try:
        module = C3k2_FasterFD(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_FasterFD  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_FasterFD  自测跳过: {e}' + RESET)
    # ---- C3k_PFDConv ----
    try:
        module = C3k_PFDConv(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k_PFDConv  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k_PFDConv  自测跳过: {e}' + RESET)
    # ---- C3k2_PFDConv ----
    try:
        module = C3k2_PFDConv(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_PFDConv  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_PFDConv  自测跳过: {e}' + RESET)

