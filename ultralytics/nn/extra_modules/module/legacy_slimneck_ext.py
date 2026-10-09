'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:1054-1065 , nn/extra_modules/block.py:1081-1089 , nn/extra_modules/block.py:1113-1117 , nn/extra_modules/block.py:1119-1124 , nn/extra_modules/block.py:5047-5051 , nn/extra_modules/block.py:5053-5056 , nn/extra_modules/block.py:7078-7108 , nn/extra_modules/block.py:7149-7155
0526 版 slim-neck 与 RepViT 变体扩展（VoVGSCSPC / VoVGSCSPns / GSConvns / C3k2_RVB_SE / multiRCM）。
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
import torch
from ultralytics.nn.extra_modules.module.RepViTBlock import RepViTBlock
from ultralytics.nn.extra_modules.neck.SlimNeck import GSBottleneck, GSBottleneckC, GSConv, VoVGSCSP
from ultralytics.nn.modules.block import C3k, C3k2


# ---- 原样迁移自 nn/extra_modules/block.py:5047-5051 ----
class C3k_RVB_SE(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepViTBlock(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:5053-5056 ----
class C3k2_RVB_SE(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_RVB_SE(self.c, self.c, 2, shortcut, g) if c3k else RepViTBlock(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:1054-1065 ----
class GSConvns(GSConv):
    # GSConv with a normative-shuffle https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__(c1, c2, k, s, p, g, act=True)
        c_ = c2 // 2
        self.shuf = nn.Conv2d(c_ * 2, c2, 1, 1, 0, bias=False)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # normative-shuffle, TRT supported
        return nn.ReLU()(self.shuf(x2))

# ---- 原样迁移自 nn/extra_modules/block.py:1081-1089 ----
class GSBottleneckns(GSBottleneck):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__(c1, c2, k, s, e)
        c_ = int(c2*e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConvns(c1, c_, 1, 1),
            GSConvns(c_, c2, 3, 1, act=False))

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

# ---- 原样迁移自 nn/extra_modules/block.py:1119-1124 ----
class VoVGSCSPC(VoVGSCSP):
    # cheap VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2)
        c_ = int(c2 * 0.5)  # hidden channels
        self.gsb = GSBottleneckC(c_, c_, 1, 1)

# ---- 原样迁移自 nn/extra_modules/block.py:1113-1117 ----
class VoVGSCSPns(VoVGSCSP):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.gsb = nn.Sequential(*(GSBottleneckns(c_, c_, e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:7149-7155 ----
class multiRCM(nn.Module):
    def __init__(self, dim, n=3) -> None:
        super().__init__()
        self.mrcm = nn.Sequential(*[RCA(dim, 3, 2, square_kernel_size=1) for _ in range(n)])
    
    def forward(self, x):
        return self.mrcm(x)

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- VoVGSCSPC ----
    try:
        module = VoVGSCSPC(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'VoVGSCSPC  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'VoVGSCSPC  自测跳过: {e}' + RESET)
    # ---- VoVGSCSPns ----
    try:
        module = VoVGSCSPns(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'VoVGSCSPns  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'VoVGSCSPns  自测跳过: {e}' + RESET)
    # ---- multiRCM ----
    try:
        module = multiRCM(in_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'multiRCM  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'multiRCM  自测跳过: {e}' + RESET)

