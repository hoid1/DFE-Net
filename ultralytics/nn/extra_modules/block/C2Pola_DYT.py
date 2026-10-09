'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/transformer.py:515-528 , nn/extra_modules/transformer.py:530-534
二次创新(2)：ICLR2025 PolaLinearAttention + CVPR2025 DyT（旧库 md #286）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch

import torch.nn as nn
from ultralytics.nn.extra_modules.norm.dyt import DynamicTanh
from ultralytics.nn.extra_modules.transformer.PolaLinearAttention import PolaLinearAttention
from ultralytics.nn.modules.block import C2PSA, PSABlock


# ---- 原样迁移自 nn/extra_modules/transformer.py:515-528 ----
class Polalock_DYT(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.attn = PolaLinearAttention(c, hw=[20, 20], num_heads=num_heads)
    
    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        BS, C, H, W = x.size()
        x = x + self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous() if self.add else self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        return x

# ---- 原样迁移自 nn/extra_modules/transformer.py:530-534 ----
class C2Pola_DYT(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(Polalock_DYT(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- C2Pola_DYT ----
    try:
        module = C2Pola_DYT(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'C2Pola_DYT  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'C2Pola_DYT  自测跳过: {e}' + RESET)

