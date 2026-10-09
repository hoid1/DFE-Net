'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/transformer.py:1511-1515 , nn/extra_modules/transformer.py:1517-1521
0526 版 C2PSA-SDTA 变体（旧库 yaml: yolo11-C2PSA-SDTA.yaml）。
签名 (c1, c2, n=1, e=0.5)，走 base_modules + repeat_modules。
内核 SDTA 直接复用 2107 版 transformer/SDTA.py —— 已 AST 比对，与 0526 版语义一致。
包含：PSABlock_SDTA, C2PSA_SDTA
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import C2PSA, PSABlock
from ultralytics.nn.extra_modules.transformer.SDTA import SDTA

# ---- 原样迁移自 nn/extra_modules/transformer.py:1511-1515 ----
class PSABlock_SDTA(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)

        self.attn = SDTA(c)

# ---- 原样迁移自 nn/extra_modules/transformer.py:1517-1521 ----
class C2PSA_SDTA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)

        self.m = nn.Sequential(*(PSABlock_SDTA(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
