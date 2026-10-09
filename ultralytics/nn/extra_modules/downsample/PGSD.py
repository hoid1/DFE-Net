'''
自研模块：Phase-Gated Sub-pixel Downsampling (PGSD)

设计动机：SPDConv 将 2x2 子格的四个采样相位等权 concat 后统一卷积，
对近周期纹理（如 NEU-DET 中 crazing / rolled-in_scale 类的弥散边界）
四个相位携带的是同一纹理的不同采样偏移，等权处理会把混叠噪声原样
带入下一层。PGSD 让四个相位先各自做深度卷积提取局部结构，再由
"每相位的全局统计量"生成竞争性权重，抑制噪声相位、放大信息相位，
并保留一条相位均值的残差以保证退化时不劣于普通平均池化。

Data flow:
    x -> 四相位子采样 (与 SPDConv 相同的取样方式)
      -> 逐相位 depthwise 3x3（不跨通道混合）
      -> 每相位统计量 (通道均值, 局部方差) 共 8 维 -> 1x1 -> 4 相位权重 -> softmax
      -> 按相位加权后 4C -> pointwise 1x1 投影到 ouc
      -> + 相位均值 1x1 残差
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class PGSD(nn.Module):
    """Phase-Gated Sub-pixel Downsampling."""

    def __init__(self, inc: int, ouc: int) -> None:
        super().__init__()
        self.inc = inc

        self.dw_phases = nn.ModuleList([
            nn.Conv2d(inc, inc, kernel_size=3, padding=1, groups=inc, bias=False)
            for _ in range(4)
        ])

        self.gate_head = nn.Conv2d(8, 4, kernel_size=1, bias=True)
        self.proj = Conv(inc * 4, ouc, 1)
        self.res_proj = Conv(inc, ouc, 1)

    @staticmethod
    def _phase_split(x: torch.Tensor):
        return [
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2],
        ]

    def _phase_stats(self, feat: torch.Tensor) -> torch.Tensor:
        mean = feat.mean(dim=(1, 2, 3), keepdim=True)
        var = feat.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        return torch.cat([mean, var], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phases = self._phase_split(x)
        feats = [dw(p) for dw, p in zip(self.dw_phases, phases)]

        stats = torch.cat([self._phase_stats(f) for f in feats], dim=1)
        logits = self.gate_head(stats)
        alpha = torch.softmax(logits, dim=1)

        weighted = torch.cat(
            [f * alpha[:, i:i + 1] for i, f in enumerate(feats)], dim=1
        )

        out = self.proj(weighted)
        phase_mean = sum(feats) / 4.0
        out = out + self.res_proj(phase_mean)
        return out


if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 256, 160, 160
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    module = PGSD(in_channel, out_channel).to(device)
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)

    print(ORANGE)
    try:
        from calflops import calculate_flops
        calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                        output_as_string=True, output_precision=4, print_detailed=True)
    except Exception as e:
        print('calflops 不可用:', e)
    print(RESET)
