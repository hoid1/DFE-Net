'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:14781-14785 , nn/extra_modules/block.py:14787-14790 , nn/extra_modules/block.py:15184-15188 , nn/extra_modules/block.py:15190-15193 , nn/extra_modules/block.py:15199-15203 , nn/extra_modules/block.py:15205-15208 , nn/extra_modules/block.py:15254-15258 , nn/extra_modules/block.py:15260-15263 , nn/extra_modules/block.py:15561-15565 , nn/extra_modules/block.py:15567-15570 , nn/extra_modules/block.py:16162-16166 , nn/extra_modules/block.py:16168-16171 , nn/extra_modules/block.py:16199-16203 , nn/extra_modules/block.py:16205-16208 , nn/extra_modules/block.py:16251-16255 , nn/extra_modules/block.py:16257-16260 , nn/extra_modules/block.py:16389-16393 , nn/extra_modules/block.py:16395-16398 , nn/extra_modules/block.py:16400-16404 , nn/extra_modules/block.py:16406-16409
A1 组：0526 版 CSP 包装壳，内核在 2107 版中已存在，此处仅迁入壳本体。
签名 (c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True) 与 base_modules + repeat_modules 契约一致，
旧 yaml 可一字不改直接使用。（等价写法：C3k2_Block, [ch, {'module': <内核>}, ...]）
本组包含：C3k2_CSI, C3k2_GLSS2D, C3k2_GLVSS, C3k2_GradMamba, C3k2_PatchMamba, C3k2_SFMB, C3k2_SMS, C3k2_TVIM, C3k2_TransMixer, C3k2_VSSD
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch

import torch.nn as nn
from ultralytics.nn.extra_modules.mamba.CSI import CSI
from ultralytics.nn.extra_modules.mamba.GLSS2D import GLSS2D
from ultralytics.nn.extra_modules.mamba.GLVSS import GL_VSS
from ultralytics.nn.extra_modules.mamba.GradMamba import GradMamba
from ultralytics.nn.extra_modules.mamba.PatchMamba import PatchMamba
from ultralytics.nn.extra_modules.mamba.SFMB import SFMB
from ultralytics.nn.extra_modules.mamba.TinyViM import TViMBlock
from ultralytics.nn.extra_modules.mamba.TransMixer import TransMixerModule
from ultralytics.nn.extra_modules.mamba.VSSD import VMAMBA2Block
from ultralytics.nn.extra_modules.mamba.sparse_state_space import SparseStateSpace
from ultralytics.nn.modules.block import C3k, C3k2


# ---- 原样迁移自 nn/extra_modules/block.py:15254-15258 ----
class C3k_CSI(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(CSI(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:15260-15263 ----
class C3k2_CSI(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_CSI(self.c, self.c, 2, shortcut, g) if c3k else CSI(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:16162-16166 ----
class C3k_GLSS2D(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GLSS2D(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:16168-16171 ----
class C3k2_GLSS2D(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_GLSS2D(self.c, self.c, 2, shortcut, g) if c3k else GLSS2D(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:14781-14785 ----
class C3k_GLVSS(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GL_VSS(c_, c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:14787-14790 ----
class C3k2_GLVSS(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_GLVSS(self.c, self.c, 2, shortcut, g) if c3k else GL_VSS(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:16389-16393 ----
class C3k_GradMamba(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GradMamba(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:16395-16398 ----
class C3k2_GradMamba(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_GradMamba(self.c, self.c, 2, shortcut, g) if c3k else GradMamba(self.c, self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:16400-16404 ----
class C3k_PatchMamba(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(PatchMamba(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:16406-16409 ----
class C3k2_PatchMamba(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_PatchMamba(self.c, self.c, 2, shortcut, g) if c3k else PatchMamba(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:15561-15565 ----
class C3k_SFMB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(SFMB(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:15567-15570 ----
class C3k2_SFMB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_SFMB(self.c, self.c, 2, shortcut, g) if c3k else SFMB(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:16251-16255 ----
class C3k_SMS(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(SparseStateSpace(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:16257-16260 ----
class C3k2_SMS(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_SMS(self.c, self.c, 2, shortcut, g) if c3k else SparseStateSpace(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:15199-15203 ----
class C3k_TVIM(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(TViMBlock(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:15205-15208 ----
class C3k2_TVIM(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_TVIM(self.c, self.c, 2, shortcut, g) if c3k else TViMBlock(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:16199-16203 ----
class C3k_TransMixer(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(TransMixerModule(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:16205-16208 ----
class C3k2_TransMixer(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_TransMixer(self.c, self.c, 2, shortcut, g) if c3k else TransMixerModule(self.c) for _ in range(n))

# ---- 原样迁移自 nn/extra_modules/block.py:15184-15188 ----
class C3k_VSSD(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(VMAMBA2Block(c_) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:15190-15193 ----
class C3k2_VSSD(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_VSSD(self.c, self.c, 2, shortcut, g) if c3k else VMAMBA2Block(self.c) for _ in range(n))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- C3k2_TransMixer ----
    try:
        module = C3k2_TransMixer(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_TransMixer  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_TransMixer  自测跳过: {e}' + RESET)
    # ---- C3k_VSSD ----
    try:
        module = C3k_VSSD(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k_VSSD  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k_VSSD  自测跳过: {e}' + RESET)
    # ---- C3k2_VSSD ----
    try:
        module = C3k2_VSSD(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C3k2_VSSD  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C3k2_VSSD  自测跳过: {e}' + RESET)

