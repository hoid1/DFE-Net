'''
自研模块：Boundary-Aware Refinement (BAR)

设计动机：CSPOmniKernel（含 31x31 深度卷积 + FFT 频域调制）与自研 HPFGA
在"大核空间聚合 + 频率线索"上高度重叠（详见 noOK 消融：删除 CSPOmniKernel
后 mAP50-95 在噪声带内持平、mAP75 反升），继续堆叠同类机制收益有限。
NEU-DET 上表现最差的类别（crazing / rolled-in_scale / scratches）共同
特征是边界弥散、局部对比度低。BAR 用边界置信图对局部反差做可学习的
非锐化掩蔽增强，不含 FFT，可与其余模块共用 AMP。

Data flow:
    x -> Sobel 边界置信图 B(x) （sigmoid 归一化到 (0,1)）
      -> 局部反差 x - avgpool_k(x)
      -> out = x + gamma * B(x) * (x - avgpool_k(x))，gamma 逐通道可学习、初始为 0
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F


class BAR(nn.Module):
    """Boundary-Aware Refinement."""

    def __init__(self, dim: int, k: int = 3) -> None:
        super().__init__()
        self.dim = dim
        self.k = k
        pad = k // 2

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x, persistent=False)
        self.register_buffer('sobel_y', sobel_y, persistent=False)

        self.edge_bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.avgpool = nn.AvgPool2d(kernel_size=k, stride=1, padding=pad)

    def _depthwise_filter(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        weight = kernel.repeat(self.dim, 1, 1, 1)
        return F.conv2d(x, weight, padding=1, groups=self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = self._depthwise_filter(x, self.sobel_x)
        gy = self._depthwise_filter(x, self.sobel_y)
        edge = torch.sigmoid(torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6) + self.edge_bias)

        local_contrast = x - self.avgpool(x)
        return x + self.gamma * edge * local_contrast


if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, channel, height, width = 1, 320, 80, 80
    inputs = torch.randn((batch_size, channel, height, width)).to(device)

    module = BAR(channel).to(device)
    outputs = module(inputs)
    print(GREEN + f'inputs.size:{inputs.size()} outputs.size:{outputs.size()}' + RESET)
    print(YELLOW + f'gamma 初始应全为 0（退化为恒等映射）: {module.gamma.abs().max().item()}' + RESET)

    print(ORANGE)
    try:
        from calflops import calculate_flops
        calculate_flops(model=module, input_shape=(batch_size, channel, height, width),
                        output_as_string=True, output_precision=4, print_detailed=True)
    except Exception as e:
        print('calflops 不可用:', e)
    print(RESET)
