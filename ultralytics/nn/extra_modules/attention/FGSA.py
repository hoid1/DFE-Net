'''
新增自研模块（第三改进点）：FGSA —— Frequency-Gated Strip Attention
作者：本次对话新设计，非 0526 迁移代码

设计动机
--------
项目现有两个改进点：
  改进点一 C3k2-HPFGA（backbone）：低层多核方向卷积 + 空间/频域双域门控
  改进点二 SOEP（neck）：CSPOmniKernel，FFT 通道调制 + 大核空间混合，聚焦小目标
两者都没有触碰 C2PSA —— 也就是骨干末端 P5（20x20，n 尺度下 self.c=128）用的
标准多头自注意力（ultralytics/nn/modules/block.py: Attention / PSABlock / C2PSA）。
在所有已跑过的 8xx 个变体（legacy0526/attention、legacy0526/csp-block 里的
C2PSA-XXX、C2BRA/C2DA/C2Pola/C2TSSA 等）中，这个标准 C2PSA 节点本身其实
从没有被“针对性结构改动”过，大多数变体是整体替换成另一种注意力/线性注意力/
可变形注意力，而不是在保留其感受野-全局建模能力的前提下做局部增强。这也是
为什么很多三模块融合后 mAP50 反而掉到 85% 以下的一个可能原因：三个模块的
功能大量重叠或互相抵消（比如同样在做全局token混合/同样在做FFT），而不是
互补。

FGSA 的策略是"小改动、强互补"：
  1) 频谱通道门控（Spectral Channel Gate）
     对 attention 的 V 分支做一次全局 FFT 能量统计（每通道幅度谱均值，
     O(1) 大小，不是逐像素频域卷积），过一个 SE 风格的 1x1-ReLU-1x1-Sigmoid
     瓶颈，得到每通道的门控权重去重标定 V。直觉：钢材表面缺陷
     （crazing裂纹/scratches划痕/rolled-in_scale轧入氧化皮）在频域上有
     明显的方向性高频能量，而背景/噪声通道往往是弥散的低频能量，用频谱
     统计做一次通道级的“通道选择”，比单纯空间 softmax 注意力更容易把
     携带缺陷边缘信息的通道保留下来。这与 HPFGA 的四叉门控是不同粒度、
     不同位置（HPFGA 在 backbone 浅层做逐像素方向核门控，FGSA 在 P5
     语义层做逐通道全局门控），因此是互补而非重复。
  2) 条带位置编码（Strip Positional Encoding）
     把原版 3x3 depthwise 的位置编码 pe 换成 1xK + Kx1 depthwise 条带卷积
     （K 默认 5），用于捕捉 scratches / rolled-in_scale 等细长方向性缺陷的
     局部先验，弥补 softmax 全局注意力对局部形状不敏感的问题。

两个改动都只作用于 C2PSA 内部，不改变通道数、不新增网络层级，因此可以在
现有 yolo11-csp-HPFGA-SOEP.yaml 中把第 10 层的 `C2PSA` 直接替换为
`C2PSA_FGSA`，其余结构（HPFGA、SOEP/CSPOmniKernel）原样保留。

参数量 / 计算量预算（n 尺度，self.c=128, num_heads=2 时的单个 block 估算）：
  原版 Attention 卷积参数量  ≈ qkv(128*256) + proj(128*128) + pe(128*9)
                              = 32768 + 16384 + 1152 ≈ 50304
  新增部分：
    strip pe (1x5+5x1 dw)    = 128*5*2 = 1280   （比原 pe 多约 128）
    spectral gate (SE, r=8)  = 128*16 + 16*128 + bias(16+128) ≈ 4240
  单个 block 增量 ≈ 4368，相对该 block 原参数量 +8.7%，
  相对全网（HPFGA+SOEP 融合后约 2.77M 参数）增量 <0.4%。
  FLOPs 增量同理：新增计算集中在 1x1 conv 和一次全局 FFT（P5 分辨率仅
  20x20，FFT 代价可忽略），整体新增 GFLOPs 预计 <1%，远低于用户要求的
  20% 上限。（以上是解析估算，请用文件末尾的 __main__ 用 calflops 在你的
  实际 GPU/CPU 环境里跑一遍，拿到与 HPFGA.py / SOEP.py 一致口径的精确数字）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2PSA, PSABlock


class FGSA_Attention(nn.Module):
    """Frequency-Gated Strip Attention.

    Drop-in replacement for ultralytics.nn.modules.block.Attention. Same
    qkv/proj/scale computation as the vanilla module (so it keeps global
    self-attention capacity), plus:
      - a spectral (FFT amplitude) channel gate applied to V before the
        attention-weighted aggregation
      - a strip-shaped (1xK + Kx1) depthwise positional branch replacing the
        vanilla 3x3 depthwise PE

    Args:
        dim: input/output channel dimension (== C2PSA.self.c)
        num_heads: number of attention heads
        attn_ratio: key-dim ratio, same semantics as vanilla Attention
        strip_k: kernel size of the strip positional branch (odd, default 5)
        gate_reduction: SE-style bottleneck reduction ratio for the gate
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5,
                 strip_k: int = 5, gate_reduction: int = 8) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)

        pad = strip_k // 2
        self.pe_h = nn.Conv2d(dim, dim, kernel_size=(1, strip_k), padding=(0, pad),
                               groups=dim, bias=False)
        self.pe_v = nn.Conv2d(dim, dim, kernel_size=(strip_k, 1), padding=(pad, 0),
                               groups=dim, bias=False)

        gate_dim = max(dim // gate_reduction, 8)
        self.gate_fc = nn.Sequential(
            nn.Conv2d(dim, gate_dim, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_dim, dim, 1, bias=True),
            nn.Sigmoid(),
        )

    def _spectral_gate(self, v_spatial: torch.Tensor) -> torch.Tensor:
        """Global per-channel FFT-amplitude energy -> SE-style channel gate."""
        amp = torch.fft.rfft2(v_spatial.float(), norm='ortho').abs()
        energy = amp.mean(dim=(2, 3), keepdim=True).to(v_spatial.dtype)  # (B,C,1,1)
        return self.gate_fc(energy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        v_spatial = v.reshape(B, C, H, W)
        gate = self._spectral_gate(v_spatial)
        v_spatial = v_spatial * gate
        v = v_spatial.reshape(B, self.num_heads, self.head_dim, N)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        pe = self.pe_v(self.pe_h(v_spatial))
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + pe
        x = self.proj(x)
        return x


class FGSA_PSABlock(PSABlock):
    """PSABlock with FGSA_Attention instead of vanilla Attention."""

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4,
                 shortcut: bool = True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        self.attn = FGSA_Attention(c, num_heads=num_heads, attn_ratio=attn_ratio)


class C2PSA_FGSA(C2PSA):
    """C2PSA variant using FGSA_Attention.

    Signature kept identical to C2PSA: (c1, c2, n=1, e=0.5) so it is a
    drop-in replacement in yaml (base_modules + repeat_modules contract,
    same as C2ASSA / C2BRA / C2PSA_AFFN etc. in this project).
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(
            *(FGSA_PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
              for _ in range(n))
        )


if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # n 尺度下 layer10 的真实形状：c1=c2=256, self.c=128 (e=0.5)
    batch_size, channel, height, width = 2, 256, 20, 20
    inputs = torch.randn((batch_size, channel, height, width)).to(device)

    baseline = C2PSA(channel, channel, n=2, e=0.5).to(device)
    module = C2PSA_FGSA(channel, channel, n=2, e=0.5).to(device)

    out_base = baseline(inputs)
    out_new = module(inputs)
    print(GREEN + f'C2PSA      out: {out_base.shape}' + RESET)
    print(GREEN + f'C2PSA_FGSA out: {out_new.shape}' + RESET)

    p_base = sum(p.numel() for p in baseline.parameters())
    p_new = sum(p.numel() for p in module.parameters())
    print(ORANGE + f'C2PSA params      : {p_base:,}' + RESET)
    print(ORANGE + f'C2PSA_FGSA params : {p_new:,}  (+{p_new - p_base:,}, +{(p_new / p_base - 1) * 100:.2f}%)' + RESET)

    try:
        from calflops import calculate_flops
        print(ORANGE)
        flops_b, macs_b, _ = calculate_flops(model=baseline, input_shape=(batch_size, channel, height, width),
                                              output_as_string=True, output_precision=4, print_detailed=False)
        flops_n, macs_n, _ = calculate_flops(model=module, input_shape=(batch_size, channel, height, width),
                                              output_as_string=True, output_precision=4, print_detailed=False)
        print(f'C2PSA      FLOPs: {flops_b}  MACs: {macs_b}')
        print(f'C2PSA_FGSA FLOPs: {flops_n}  MACs: {macs_n}')
        print(RESET)
    except Exception as e:
        print('calflops 不可用，仅报告参数量:', e)
