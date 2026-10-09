'''
本文件由 0526 版迁移而来（代码逐字保留，未作任何修改）
来源：nn/backbone/lsnet.py:1-16 , 17-210 , 211-233 , 376-396
CVPR2025 LSNet 的 LSConv 内核（SKA / LKP / triton 算子），供 md #319 MambaOut-LSConv 使用。
新版已删除 nn/backbone 目录，故把这段自包含的内核单独迁到 conv_module 下。
triton 不可用时会自动回落到 except 分支，不影响其他模块。
'''

import os, sys, functools
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.autograd import Function

import torch, math, tqdm
import itertools
from timm.layers import SqueezeExcite, trunc_normal_
try:
    from torch.amp import custom_fwd, custom_bwd
    import inspect
    if 'device_type' in inspect.signature(custom_fwd).parameters:
        custom_fwd = functools.partial(custom_fwd, device_type='cuda')
        custom_bwd = functools.partial(custom_bwd, device_type='cuda')
except:
    from torch.cuda.amp import custom_fwd, custom_bwd
from torch.autograd import Function
try:
    import triton
    import triton.language as tl
except:
    pass

__all__ = ['lsnet_t', 'lsnet_s', 'lsnet_b', 'SKA']


try:
    def _grid(numel: int, bs: int) -> tuple:
        return (triton.cdiv(numel, bs),)

    @triton.jit
    def _idx(i, n: int, c: int, h: int, w: int):
        ni = i // (c * h * w)
        ci = (i // (h * w)) % c
        hi = (i // w) % h
        wi = i % w
        m = i < (n * c * h * w)
        return ni, ci, hi, wi, m

    @triton.jit
    def ska_fwd(
        x_ptr, w_ptr, o_ptr,
        n, ic, h, w, ks, pad, wc,
        BS: tl.constexpr,
        CT: tl.constexpr, AT: tl.constexpr
    ):
        pid = tl.program_id(0)
        start = pid * BS
        offs = start + tl.arange(0, BS)

        ni, ci, hi, wi, m = _idx(offs, n, ic, h, w)
        val = tl.zeros((BS,), dtype=AT)

        for kh in range(ks):
            hin = hi - pad + kh
            hb = (hin >= 0) & (hin < h)
            for kw in range(ks):
                win = wi - pad + kw
                b = hb & (win >= 0) & (win < w)

                x_off = ((ni * ic + ci) * h + hin) * w + win
                w_off = ((ni * wc + ci % wc) * ks * ks + (kh * ks + kw)) * h * w + hi * w + wi

                x_val = tl.load(x_ptr + x_off, mask=m & b, other=0.0).to(CT)
                w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(CT)
                val += tl.where(b & m, x_val * w_val, 0.0).to(AT)

        tl.store(o_ptr + offs, val.to(CT), mask=m)

    @triton.jit
    def ska_bwd_x(
        go_ptr, w_ptr, gi_ptr,
        n, ic, h, w, ks, pad, wc,
        BS: tl.constexpr,
        CT: tl.constexpr, AT: tl.constexpr
    ):
        pid = tl.program_id(0)
        start = pid * BS
        offs = start + tl.arange(0, BS)

        ni, ci, hi, wi, m = _idx(offs, n, ic, h, w)
        val = tl.zeros((BS,), dtype=AT)

        for kh in range(ks):
            ho = hi + pad - kh
            hb = (ho >= 0) & (ho < h)
            for kw in range(ks):
                wo = wi + pad - kw
                b = hb & (wo >= 0) & (wo < w)

                go_off = ((ni * ic + ci) * h + ho) * w + wo
                w_off = ((ni * wc + ci % wc) * ks * ks + (kh * ks + kw)) * h * w + ho * w + wo

                go_val = tl.load(go_ptr + go_off, mask=m & b, other=0.0).to(CT)
                w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(CT)
                val += tl.where(b & m, go_val * w_val, 0.0).to(AT)

        tl.store(gi_ptr + offs, val.to(CT), mask=m)

    @triton.jit
    def ska_bwd_w(
        go_ptr, x_ptr, gw_ptr,
        n, wc, h, w, ic, ks, pad,
        BS: tl.constexpr,
        CT: tl.constexpr, AT: tl.constexpr
    ):
        pid = tl.program_id(0)
        start = pid * BS
        offs = start + tl.arange(0, BS)

        ni, ci, hi, wi, m = _idx(offs, n, wc, h, w)

        for kh in range(ks):
            hin = hi - pad + kh
            hb = (hin >= 0) & (hin < h)
            for kw in range(ks):
                win = wi - pad + kw
                b = hb & (win >= 0) & (win < w)
                w_off = ((ni * wc + ci) * ks * ks + (kh * ks + kw)) * h * w + hi * w + wi

                val = tl.zeros((BS,), dtype=AT)
                steps = (ic - ci + wc - 1) // wc
                for s in range(tl.max(steps, axis=0)):
                    cc = ci + s * wc
                    cm = (cc < ic) & m & b

                    x_off = ((ni * ic + cc) * h + hin) * w + win
                    go_off = ((ni * ic + cc) * h + hi) * w + wi

                    x_val = tl.load(x_ptr + x_off, mask=cm, other=0.0).to(CT)
                    go_val = tl.load(go_ptr + go_off, mask=cm, other=0.0).to(CT)
                    val += tl.where(cm, x_val * go_val, 0.0).to(AT)

                tl.store(gw_ptr + w_off, val.to(CT), mask=m)
    
    class SkaFn(Function):
        @staticmethod
        @custom_fwd
        def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            ks = int(math.sqrt(w.shape[2]))
            pad = (ks - 1) // 2
            ctx.ks, ctx.pad = ks, pad
            n, ic, h, width = x.shape
            wc = w.shape[1]
            o = torch.empty(n, ic, h, width, device=x.device, dtype=x.dtype)
            numel = o.numel()

            x = x.contiguous()
            w = w.contiguous()

            grid = lambda meta: _grid(numel, meta["BS"])

            ct = tl.float16 if x.dtype == torch.float16 else (tl.float32 if x.dtype == torch.float32 else tl.float64)
            at = tl.float32 if x.dtype == torch.float16 else ct

            ska_fwd[grid](x, w, o, n, ic, h, width, ks, pad, wc, BS=1024, CT=ct, AT=at)

            ctx.save_for_backward(x, w)
            ctx.ct, ctx.at = ct, at
            return o

        @staticmethod
        @custom_bwd
        def backward(ctx, go: torch.Tensor) -> tuple:
            ks, pad = ctx.ks, ctx.pad
            x, w = ctx.saved_tensors
            n, ic, h, width = x.shape
            wc = w.shape[1]

            go = go.contiguous()
            gx = gw = None
            ct, at = ctx.ct, ctx.at

            if ctx.needs_input_grad[0]:
                gx = torch.empty_like(x)
                numel = gx.numel()
                ska_bwd_x[lambda meta: _grid(numel, meta["BS"])](go, w, gx, n, ic, h, width, ks, pad, wc, BS=1024, CT=ct, AT=at)

            if ctx.needs_input_grad[1]:
                gw = torch.empty_like(w)
                numel = gw.numel() // w.shape[2]
                ska_bwd_w[lambda meta: _grid(numel, meta["BS"])](go, x, gw, n, wc, h, width, ic, ks, pad, BS=1024, CT=ct, AT=at)

            return gx, gw, None, None
except:
    def _idx(i, n: int, c: int, h: int, w: int):
        pass

    def ska_fwd(
        x_ptr, w_ptr, o_ptr,
        n, ic, h, w, ks, pad, wc,
        BS,
        CT, AT
    ):
        pass

    def ska_bwd_x(
        go_ptr, w_ptr, gi_ptr,
        n, ic, h, w, ks, pad, wc,
        BS,
        CT, AT
    ):
        pass

    def ska_bwd_w(
        go_ptr, x_ptr, gw_ptr,
        n, wc, h, w, ic, ks, pad,
        BS,
        CT, AT
    ):
        pass

class SKA(torch.nn.Module):
        def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            return SkaFn.apply(x, w) # type: ignore

class SKA(torch.nn.Module):
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return SkaFn.apply(x, w) # type: ignore


# ---- 原样迁移自 nn/backbone/lsnet.py:211-233 ----
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)
        
        self.sks = sks
        self.groups = groups
        self.dim = dim
        
    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w

