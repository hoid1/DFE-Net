'''
自研模块：Source-confidence Adaptive Fusion (SRAF)

设计动机：SOEP layer 16 的三路 Concat（SPD 相位特征 / 上采样语义特征 /
backbone P3 侧向特征）把可靠性完全不同的三路特征等权拼接，交由下游卷积
自行排序，代价是通道数被推高。SRAF 用每路的局部统计量生成竞争性空间
权重，并将门控头的 bias 初始化为偏向骨干侧向特征（P3 融合场景中通常
最稳定的一路），使模块初始状态接近直通，训练只需做增量修正。

Data flow:
    [x_1, ..., x_n] -> 各路 1x1 对齐到 ouc
                    -> 各路 (通道均值, 梯度幅值) 共 2n 维描述子
                    -> 1x1 -> n 个 logits -> 沿源维 softmax
                    -> 加权求和
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class SRAF(nn.Module):
    """Source-confidence Adaptive Fusion.

    Args:
        inc: 各输入源的通道数列表，如 [c1, c2, c3]。
        ouc: 输出通道数。
        dominant_index: 初始化时偏向的源下标（门控 bias 会向此源倾斜）。
    """

    def __init__(self, inc: list, ouc: int, dominant_index: int = -1) -> None:
        super().__init__()
        self.n = len(inc)
        self.ouc = ouc

        self.align = nn.ModuleList([Conv(c, ouc, 1) for c in inc])

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x, persistent=False)
        self.register_buffer('sobel_y', sobel_y, persistent=False)

        self.gate_head = nn.Conv2d(2 * self.n, self.n, kernel_size=1, bias=True)

        with torch.no_grad():
            bias = torch.full((self.n,), -2.0)
            dom = dominant_index if dominant_index >= 0 else self.n + dominant_index
            bias[dom] = 2.0
            self.gate_head.bias.copy_(bias)
            self.gate_head.weight.zero_()

    def _grad_mag(self, x: torch.Tensor) -> torch.Tensor:
        xm = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(xm, self.sobel_x, padding=1)
        gy = F.conv2d(xm, self.sobel_y, padding=1)
        return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)

    def forward(self, x: list) -> torch.Tensor:
        aligned = [align(xi) for align, xi in zip(self.align, x)]

        descs = []
        for feat in aligned:
            mean = feat.mean(dim=1, keepdim=True)
            grad = self._grad_mag(feat)
            descs.append(torch.cat([mean, grad], dim=1))
        desc = torch.cat(descs, dim=1)

        logits = self.gate_head(desc)
        alpha = torch.softmax(logits, dim=1)

        out = sum(aligned[i] * alpha[:, i:i + 1] for i in range(self.n))
        return out


if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, height, width = 1, 80, 80
    c_list = [64, 128, 64]
    inputs = [torch.randn((batch_size, c, height, width)).to(device) for c in c_list]

    module = SRAF(c_list, 128, dominant_index=-1).to(device)
    outputs = module(inputs)
    print(GREEN + f'inputs:{[tuple(i.shape) for i in inputs]} outputs.size:{outputs.size()}' + RESET)

    with torch.no_grad():
        alpha_check = torch.softmax(module.gate_head.bias, dim=0)
        print(YELLOW + f'初始 softmax 权重（应偏向最后一路）: {alpha_check.tolist()}' + RESET)

    print(ORANGE)
    try:
        from calflops import calculate_flops
        calculate_flops(model=module, args=[inputs], output_as_string=True,
                        output_precision=4, print_detailed=True)
    except Exception as e:
        print('calflops 不可用:', e)
    print(RESET)
