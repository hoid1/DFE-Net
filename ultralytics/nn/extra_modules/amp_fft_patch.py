# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
AMP(混合精度) 与 FFT 的兼容补丁。

问题
----
开启 amp=True 训练时,特征图是 fp16。cuFFT 在半精度下只支持边长为 2 的幂的信号
(64/128/256...),而检测网络的特征图常见 80x80、40x40、20x20,于是报错:

    RuntimeError: cuFFT only supports dimensions whose sizes are powers of two
    when computing in half precision, but got a signal size of [80, 80]

注意:只写 `with torch.autocast(enabled=False)` 是没用的。
autocast 只影响"块内新产生"的算子,不会改变已经是 fp16 的输入张量。
必须显式把张量转成 fp32(或把 complex32 转成 complex64)。

做法
----
在 torch.fft 的正/逆变换外面包一层:
输入是 fp16/bf16 就先转 fp32,是 complex32 就先转 complex64,再做变换。
输出保持 fp32/complex64,后续卷积/线性层由 autocast 自动再降回 fp16,
显存和速度基本不受影响。

torch.complex 也一并处理,因为半精度实部/虚部会拼出 complex32,
同样会踩 cuFFT 的半精度限制。

关闭方式
--------
设置环境变量 ULTRA_NO_FP32_FFT=1 即可跳过本补丁。
"""

import functools
import os

import torch

__all__ = ("enable_fp32_fft", "fp32_fft_forward", "to_fft_safe_dtype")

_PATCH_FLAG = "_ultra_fp32_fft_patched"

_HALF_REAL = (torch.float16, torch.bfloat16)

# 需要打补丁的变换函数(不存在的会自动跳过)
_FFT_FUNCS = (
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfft2",
    "irfft2",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
    "hfft2",
    "ihfft2",
    "hfftn",
    "ihfftn",
)


def to_fft_safe_dtype(t):
    """把 FFT 不支持的半精度类型抬到 fp32 / complex64,其余原样返回。"""
    if not torch.is_tensor(t):
        return t
    if t.dtype in _HALF_REAL:
        return t.float()
    if t.dtype == torch.complex32:
        return t.to(torch.complex64)
    return t


def _device_type(t):
    return t.device.type if torch.is_tensor(t) else "cuda"


def _wrap_fft(fn):
    """包装单个 torch.fft.* 函数。"""
    if getattr(fn, _PATCH_FLAG, False):
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if args:
            first = args[0]
            args = (to_fft_safe_dtype(first),) + args[1:]
        elif "input" in kwargs:
            first = kwargs["input"]
            kwargs["input"] = to_fft_safe_dtype(first)
        else:
            return fn(*args, **kwargs)

        with torch.autocast(device_type=_device_type(first), enabled=False):
            return fn(*args, **kwargs)

    setattr(wrapper, _PATCH_FLAG, True)
    return wrapper


def _wrap_complex(fn):
    """包装 torch.complex,避免产生 complex32。"""
    if getattr(fn, _PATCH_FLAG, False):
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        args = tuple(a.float() if torch.is_tensor(a) and a.dtype in _HALF_REAL else a for a in args)
        for k in ("real", "imag"):
            v = kwargs.get(k)
            if torch.is_tensor(v) and v.dtype in _HALF_REAL:
                kwargs[k] = v.float()
        return fn(*args, **kwargs)

    setattr(wrapper, _PATCH_FLAG, True)
    return wrapper


def fp32_fft_forward(func):
    """
    装饰器:给自研模块的 forward 用,整段强制走 fp32。

    用法:
        @fp32_fft_forward
        def forward(self, x):
            ...
    """

    @functools.wraps(func)
    def wrapper(self, x, *args, **kwargs):
        if torch.is_tensor(x) and x.dtype in _HALF_REAL:
            with torch.autocast(device_type=x.device.type, enabled=False):
                return func(self, x.float(), *args, **kwargs)
        return func(self, x, *args, **kwargs)

    return wrapper


def enable_fp32_fft(verbose=False):
    """安装补丁。重复调用无副作用。"""
    if os.environ.get("ULTRA_NO_FP32_FFT", "") == "1":
        return False
    if getattr(torch.fft, _PATCH_FLAG, False):
        return False

    patched = []
    for name in _FFT_FUNCS:
        fn = getattr(torch.fft, name, None)
        if fn is None:
            continue
        setattr(torch.fft, name, _wrap_fft(fn))
        patched.append(name)

    if hasattr(torch, "complex"):
        torch.complex = _wrap_complex(torch.complex)
        patched.append("complex")

    setattr(torch.fft, _PATCH_FLAG, True)

    if verbose:
        print(f"[amp_fft_patch] fp32-FFT enabled for: {', '.join(patched)}")
    return True
