'''
================================================================================
RG-SOEP: Radial-frequency Gated Small Object Enhance Pyramid
        —— 对自研 SOEP（SPDConv + Concat + CSPOmniKernel）的二次创新
================================================================================

【原版 SOEP 的三个问题】
  P1. 三路等权 Concat：SPD 上来的 P2 细节支噪声大、语义弱，与 P4 语义支、P3 主干支
      直接按通道拼接后由 C3k2 隐式加权，缺少显式的"分支竞争"，高分辨率噪声会稀释
      定位线索 —— 表现为融合后 mAP75 反而下降。
  P2. OmniKernel 的 fca 分支只用 GAP 出来的每通道标量去乘整张频谱，再取 abs()。
      数学上 |IFFT(a·X)| ≈ |a|·|x|，退化成一个普通通道注意力，并没有做真正的
      "频带选择"；而 abs() 还会丢弃相位，破坏定位信息。
  P3. 计算量：cv1/cv2 两个全通道 1×1 + 31×31 稠密深度卷积，占了 SOEP 一半以上的
      FLOPs；且 FFT 在 fp16 下不稳定，必须关掉 amp（训练变慢、显存翻倍）。

【本文件的三个改进】
  C1. SGFF —— 通道-空间联合竞争门控融合（替代 Concat）
      · 三支先 1×1 对齐到同一宽度；细节支额外做"高频提纯"（去掉低频均值分量）
      · 由联合上下文同时生成 逐像素 softmax 竞争权重 和 逐通道 softmax 竞争权重
      · 以主干 P3 支作为残差锚点，保证不弱于原始路径
      · 输出宽度可控（默认 512→n 时 128），比原来 320 通道更省，下游 C3k2 也变便宜

  C2. CSPRFA —— 径向频带门控 + 条带/空洞解耦大核（替代 CSPOmniKernel）
      · RFBG：把 rfft2 频谱按 归一化径向频率 软划分为 低/中/高 三带，
        每带每通道一个"内容自适应增益"，这才是真正的频带选择；
        用 rfft2/irfft2 直接得到实数输出，不再 abs()，相位完整保留
      · GDConv：按卷积定理 IFFT(FFT(k)·FFT(v)) 实现内容自适应的全局循环卷积，
        替换原 FGM 里"空间域 × 频域"的越域相乘
      · 大核解耦：31×31 稠密 dw → (1×11+11×1) + 空洞(1×7+7×1,d=3) + 3×3(d=4) + 1×1
        每通道乘法 1024 → 46，约 22× 便宜，等效感受野仍 ≥19，再叠加 GDConv 的全局感受野
        （长条带正好匹配 scratches / rolled-in_scale，空洞分支匹配 crazing 的网状纹理）
      · 真·CSP：去掉全通道 cv1，只对 e 比例的分支做变换
      · AMP 安全：所有 FFT 局部关闭 autocast 并升到 fp32，可以正常开 amp 训练

  C3. SPDConvLite —— 空间到深度下采样轻量化（可选）
      · 4C→C 的 3×3 稠密卷积 → 1×1 压缩 + 3×3 深度卷积 + 1×1，约 8× 便宜

作者接口约定（与仓库 parse_model 对齐）：
  SGFF        → 注册进 featurefusion 分支：  __init__(inc: list, ouc: int, ...)
  CSPRFA      → 注册进 legacy0526_dim 分支： __init__(dim: int, e=0.25, ...)
  SPDConvLite → 注册进 downsample_modules： __init__(inc: int, ouc: int)
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

__all__ = ['SGFF', 'CSPRFA', 'RFBG', 'GDConv', 'DecoupledLargeKernel', 'SPDConvLite']


# --------------------------------------------------------------------------- #
# 基础件
# --------------------------------------------------------------------------- #
def _gn_groups(c, prefer=8):
    """为 GroupNorm 自动挑一个能整除通道数的分组数。"""
    for g in range(min(prefer, c), 0, -1):
        if c % g == 0:
            return g
    return 1


class ConvNormAct(nn.Module):
    """Conv + GroupNorm + SiLU。用 GroupNorm 而非 BN，小 batch 训练更稳。"""

    def __init__(self, c1, c2, k=1, s=1, d=1, g=1, act=True):
        super().__init__()
        p = d * (k // 2)
        if c1 % g or c2 % g:
            g = 1
        self.conv = nn.Conv2d(c1, c2, k, s, p, dilation=d, groups=g, bias=False)
        self.norm = nn.GroupNorm(_gn_groups(c2), c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


def _fp32_fft_ctx(x):
    """返回一个上下文管理器：在其中关闭 autocast，保证 FFT 在 fp32 下计算。"""
    return torch.autocast(device_type=x.device.type, enabled=False)


# --------------------------------------------------------------------------- #
# C2-①  RFBG：径向频带门控 (Radial Frequency Band Gating)
# --------------------------------------------------------------------------- #
class RFBG(nn.Module):
    """
    把特征的 2D 频谱按归一化径向频率软划分成 低频/中频/高频 三个带，
    每个带、每个通道给一个由全局上下文预测出来的增益，再做逆变换。

    与原 OmniKernel 的 fca 相比：
      · 原版：X_fft * a(每通道一个标量) → 对所有频率一视同仁，等价于通道缩放
      · 本版：X_fft * (g_low·M_low + g_mid·M_mid + g_high·M_high) → 真正的频带选择
      · 用 rfft2/irfft2，输出天然是实数，无需 abs()，相位不丢
    """

    def __init__(self, dim, reduction=4, t_low=0.25, t_high=0.55, soft=0.06):
        super().__init__()
        self.dim = dim
        self.t_low, self.t_high, self.soft = t_low, t_high, soft

        hidden = max(dim // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # 由全局上下文预测 3 个频带 × dim 通道 的增益
        self.gate = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, dim * 3, 1, bias=True),
        )
        # 与数据无关的频带先验（可学习），初始为 0 → 增益初始为 1，等价于恒等变换
        self.band_prior = nn.Parameter(torch.zeros(3, dim, 1, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

        self._mask_cache = {}   # 普通字典，不进 state_dict

    def _bands(self, h, w, device, dtype):
        key = (h, w, device, dtype)
        if key in self._mask_cache:
            m = self._mask_cache[key]
            if m.device != device or m.dtype != dtype:
                m = m.to(device=device, dtype=dtype)
                self._mask_cache[key] = m
            return m
        fy = torch.fft.fftfreq(h, device=device, dtype=dtype).view(-1, 1)   # (H,1)
        fx = torch.fft.rfftfreq(w, device=device, dtype=dtype).view(1, -1)  # (1,W//2+1)
        r = torch.sqrt(fy ** 2 + fx ** 2)
        r = r / (r.max() + 1e-6)                                            # 归一化到 [0,1]
        hi = torch.sigmoid((r - self.t_high) / self.soft)
        lo = 1.0 - torch.sigmoid((r - self.t_low) / self.soft)
        mid = (1.0 - lo - hi).clamp_min(0.0)                                # 三带之和 ≈ 1
        m = torch.stack([lo, mid, hi], 0).unsqueeze(1)                      # (3,1,H,W2)
        if len(self._mask_cache) > 8:
            self._mask_cache.clear()
        self._mask_cache[key] = m
        return m

    def forward(self, x):
        b, c, h, w = x.shape
        g = self.gate(self.pool(x)).view(b, 3, c, 1, 1)
        g = 1.0 + torch.tanh(g + self.band_prior.unsqueeze(0))              # 增益 ∈ (0,2)，初始 =1

        with _fp32_fft_ctx(x):
            xf = x.float()
            gf = g.float()
            spec = torch.fft.rfft2(xf, norm='ortho')                        # (B,C,H,W2) complex
            masks = self._bands(h, w, x.device, torch.float32)              # (3,1,H,W2)
            mod = (gf * masks.unsqueeze(0)).sum(dim=1)                      # (B,C,H,W2)
            out = torch.fft.irfft2(spec * mod, s=(h, w), norm='ortho')      # 实数输出，相位保留
        return out.to(x.dtype)


# --------------------------------------------------------------------------- #
# C2-②  GDConv：基于卷积定理的全局动态卷积（替换原 FGM）
# --------------------------------------------------------------------------- #
class GDConv(nn.Module):
    """
    原 FGM 做的是  x1 * FFT(x2)  —— 空间域张量直接乘频域张量，量纲/域都不一致。
    本模块按卷积定理改写为  IFFT( FFT(k) · FFT(v) )，即 k 与 v 的循环卷积：
    卷积核由特征自身生成 → 内容自适应的全局感受野，且有明确的信号处理解释。
    alpha 初始为 0，模块启动时等价于恒等映射，训练更稳。
    """

    def __init__(self, dim):
        super().__init__()
        self.k = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        h, w = x.shape[-2:]
        k, v = self.k(x), self.v(x)
        with _fp32_fft_ctx(x):
            kf = torch.fft.rfft2(k.float(), norm='ortho')
            vf = torch.fft.rfft2(v.float(), norm='ortho')
            out = torch.fft.irfft2(kf * vf, s=(h, w), norm='ortho')
        return out.to(x.dtype) * self.alpha + x * self.beta


# --------------------------------------------------------------------------- #
# C2-③  解耦大核：条带 + 空洞，代替 31×31 稠密深度卷积
# --------------------------------------------------------------------------- #
class DecoupledLargeKernel(nn.Module):
    """
    每通道乘法量对比（dim 归一化后）：
        原 OmniKernel : 31×31 + 1×31 + 31×1 + 1×1 = 1024
        本模块        : (11+11) + (7+7) + 9 + 1   = 46      ≈ 22× 更便宜
    分支语义：
        strip  —— 短条带，对应 scratches 这类细长划痕
        dstrip —— 空洞条带(d=3, 等效跨度 19)，对应 rolled-in_scale 的长程条纹
        dila   —— 3×3 空洞(d=4, 等效 9×9)，对应 crazing 的网状重复纹理
        point  —— 1×1，保留原始逐点响应
    """

    def __init__(self, dim, k=11, dk=7, dilation=3, ddil=4):
        super().__init__()
        self.s_h = nn.Conv2d(dim, dim, (1, k), padding=(0, k // 2), groups=dim)
        self.s_v = nn.Conv2d(dim, dim, (k, 1), padding=(k // 2, 0), groups=dim)
        pd = dilation * (dk // 2)
        self.d_h = nn.Conv2d(dim, dim, (1, dk), padding=(0, pd), dilation=(1, dilation), groups=dim)
        self.d_v = nn.Conv2d(dim, dim, (dk, 1), padding=(pd, 0), dilation=(dilation, 1), groups=dim)
        self.dila = nn.Conv2d(dim, dim, 3, padding=ddil, dilation=ddil, groups=dim)
        self.point = nn.Conv2d(dim, dim, 1, groups=dim)

    def forward(self, x):
        strip = self.s_v(self.s_h(x))       # 级联 → 十字形 11×11 感受野
        dstrip = self.d_v(self.d_h(x))      # 级联 → 等效 19×19
        return strip + dstrip + self.dila(x) + self.point(x)


# --------------------------------------------------------------------------- #
# C2   RFA / CSPRFA：改进后的 OmniKernel 主体与 CSP 包装
# --------------------------------------------------------------------------- #
class RFA(nn.Module):
    """Radial-Frequency Aggregation —— 对应原 OmniKernel。"""

    def __init__(self, dim):
        super().__init__()
        self.in_conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU())
        self.out_conv = nn.Conv2d(dim, dim, 1)

        self.lk = DecoupledLargeKernel(dim)          # 空间：解耦大核
        self.rfbg = RFBG(dim)                        # 频域：径向频带门控
        self.gd = GDConv(dim)                        # 频域：全局动态卷积

        # 空间通道注意力（保留原 sca 思路，但接在频带门控之后）
        self.sca_pool = nn.AdaptiveAvgPool2d(1)
        self.sca_conv = nn.Conv2d(dim, dim, 1, bias=True)

        self.act = nn.SiLU(inplace=True)
        # 频域支的残差缩放，初始较小，避免训练早期频域分支扰乱空间主路
        self.gamma = nn.Parameter(torch.full((dim, 1, 1), 0.1))

    def forward(self, x):
        y = self.in_conv(x)

        f = self.rfbg(y)                             # 频带选择
        f = self.sca_conv(self.sca_pool(f)) * f      # 通道重标定
        f = self.gd(f)                               # 全局动态卷积

        out = x + self.lk(y) + self.gamma * f
        return self.out_conv(self.act(out))


class CSPRFA(nn.Module):
    """
    真·CSP 包装：只对 e 比例的通道跑重计算分支，其余直通。
    相比原 CSPOmniKernel 去掉了全通道 cv1（省下 dim²·H·W 的 1×1），
    仅保留融合用的 cv2。
    """

    def __init__(self, dim, e=0.25, use_cv1=False):
        super().__init__()
        self.dim = dim
        c = max(int(dim * e) // 8 * 8, 8)
        c = min(c, dim - 8) if dim > 8 else dim
        self.c = c
        self.cv1 = Conv(dim, dim, 1) if use_cv1 else nn.Identity()
        self.m = RFA(c)
        self.cv2 = Conv(dim, dim, 1)

    def forward(self, x):
        y = self.cv1(x)
        a, b = torch.split(y, [self.c, self.dim - self.c], dim=1)
        return self.cv2(torch.cat((self.m(a), b), 1))


# --------------------------------------------------------------------------- #
# C1   SGFF：通道-空间联合竞争门控融合（替代 Concat）
# --------------------------------------------------------------------------- #
class SGFF(nn.Module):
    """
    Selective Gated Feature Fusion.

    yaml 里输入顺序需与 `[[-1, -2, 4], 1, SGFF, [512]]` 一致，即
        idx0 = SPDConv 出来的 P2 细节支
        idx1 = 上采样来的 P4 语义支
        idx2 = backbone P3 主干支（残差锚点）
    可用 detail_idx / base_idx 显式指定。

    流程：
      1) 各支 1×1 对齐到 ouc；
      2) 细节支做高频提纯 d ← d + w·(d − blur(d))，抑制 SPD 带来的低频冗余；
      3) 由三支联合上下文同时预测 逐像素 竞争权重(softmax over branch) 与
         逐通道 竞争权重(softmax over branch)，两者相乘得到最终门控；
      4) 加权求和 → 3×3 深度卷积细化 → 与主干支残差相加。
    """

    def __init__(self, inc, ouc, reduction=4, detail_idx=0, base_idx=2):
        super().__init__()
        assert isinstance(inc, (list, tuple)) and len(inc) >= 2, 'SGFF 需要多路输入'
        self.n = len(inc)
        self.ouc = ouc
        self.detail_idx = detail_idx % self.n
        self.base_idx = base_idx % self.n

        self.align = nn.ModuleList([ConvNormAct(c, ouc, 1) for c in inc])

        # 细节支高频提纯：x - avgpool_blur(x)
        self.hf_w = nn.Parameter(torch.zeros(ouc, 1, 1))
        self.hf_conv = nn.Conv2d(ouc, ouc, 3, padding=1, groups=ouc, bias=False)

        hidden = max(ouc // reduction, 8)
        self.ctx = ConvNormAct(ouc * self.n, hidden, 1)
        self.spatial_gate = nn.Conv2d(hidden, self.n, 3, padding=1)          # (B,n,H,W)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, ouc * self.n, 1),
        )                                                                     # (B,n*C,1,1)

        self.proj = nn.Sequential(
            nn.Conv2d(ouc, ouc, 3, padding=1, groups=ouc, bias=False),
            nn.GroupNorm(_gn_groups(ouc), ouc),
            nn.SiLU(inplace=True),
            nn.Conv2d(ouc, ouc, 1, bias=False),
        )
        self.gamma = nn.Parameter(torch.full((ouc, 1, 1), 0.5))

    def forward(self, xs):
        assert len(xs) == self.n, f'SGFF 期望 {self.n} 路输入，实际 {len(xs)} 路'
        feats = [a(x) for a, x in zip(self.align, xs)]

        # 细节支高频提纯
        d = feats[self.detail_idx]
        hf = d - F.avg_pool2d(d, 3, stride=1, padding=1)
        feats[self.detail_idx] = d + self.hf_w * self.hf_conv(hf)

        b, c, h, w = feats[0].shape
        ctx = self.ctx(torch.cat(feats, dim=1))
        ws = self.spatial_gate(ctx).softmax(dim=1)                            # 逐像素竞争
        wc = self.channel_gate(ctx).view(b, self.n, c, 1, 1).softmax(dim=1)   # 逐通道竞争

        out = 0
        for i, f in enumerate(feats):
            out = out + f * ws[:, i:i + 1] * wc[:, i]
        out = out * self.n                                                    # 幅度补偿

        return self.proj(out) * self.gamma + feats[self.base_idx]


# --------------------------------------------------------------------------- #
# C3   SPDConvLite：轻量空间到深度下采样（可选）
# --------------------------------------------------------------------------- #
class SPDConvLite(nn.Module):
    """原 SPDConv 用 3×3 稠密卷积把 4C 压到 C，是 SOEP 里第二大的 FLOPs 来源。
    这里拆成 1×1 压缩 + 3×3 深度卷积 + 1×1 混合，约 8× 便宜，精度基本无损。"""

    def __init__(self, inc, ouc):
        super().__init__()
        self.cv1 = Conv(inc * 4, ouc, 1)
        self.dw = Conv(ouc, ouc, 3, g=ouc)
        self.cv2 = Conv(ouc, ouc, 1)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                       x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        return self.cv2(self.dw(self.cv1(x)))


# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    GREEN, RESET = "\033[92m", "\033[0m"
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(2, 128, 80, 80).to(dev)
    m = CSPRFA(128).to(dev)
    print(GREEN + f'CSPRFA  in {tuple(x.shape)} -> out {tuple(m(x).shape)}' + RESET)

    xs = [torch.randn(2, 64, 80, 80).to(dev),
          torch.randn(2, 128, 80, 80).to(dev),
          torch.randn(2, 128, 80, 80).to(dev)]
    f = SGFF([64, 128, 128], 128).to(dev)
    print(GREEN + f'SGFF    -> out {tuple(f(xs).shape)}' + RESET)

    s = SPDConvLite(64, 64).to(dev)
    print(GREEN + f'SPDLite in (2,64,160,160) -> out '
                  f'{tuple(s(torch.randn(2,64,160,160).to(dev)).shape)}' + RESET)

    for name, mod in [('CSPRFA', m), ('SGFF', f), ('SPDConvLite', s)]:
        print(f'{name}: {sum(p.numel() for p in mod.parameters()):,} params')
