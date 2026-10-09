'''
本文件由BiliBili：魔傀面具整理
ultralytics/nn/module_images/CVPR2023-SMPConv.md
论文链接：https://arxiv.org/pdf/2304.02330

---- 修复说明 ----
迁移时这里被误放成了另一套不兼容的 SMPConv 实现（4 参数构造、forward 里带了一个
本不存在的 conv1x1 投影），和真正从 0526 版逐字迁移过来的调用方
（extra_modules/module/SMPCGLU.py、extra_modules/conv_module/SMPCNN.py）用的
6 参数 API 完全对不上：

    SMPConv(inc, kernel_size, n_points, 1, padding=kernel_size // 2, groups=1)
    TypeError: SMPConv.__init__() got an unexpected keyword argument \'padding\'

现改为 0526 原版实现（nn/extra_modules/SMPConv.py 全文件，含 get_conv2d /
conv_bn / conv_bn_relu 等配套 helper，均未作任何修改），SMPConv 不再自带
conv1x1（0526 原版没有这一层，输入输出通道数由调用方保证一致）。
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath

try:
    from depthwise_conv2d_implicit_gemm import _DepthWiseConv2dImplicitGEMMFP16, _DepthWiseConv2dImplicitGEMMFP32
except ImportError as e:
    pass

__all__ = ['SMPConv', 'get_conv2d', 'get_bn', 'enable_sync_bn', 'conv_bn', 'conv_bn_relu', 'fuse_bn']


def rel_pos(kernel_size):
    tensors = [torch.linspace(-1, 1, steps=kernel_size) for _ in range(2)]
    kernel_coord = torch.stack(torch.meshgrid(*tensors), dim=-0)
    kernel_coord = kernel_coord.unsqueeze(0)
    return kernel_coord


class SMPConv(nn.Module):
    def __init__(self, planes, kernel_size, n_points, stride, padding, groups):
        super().__init__()

        self.planes = planes
        self.kernel_size = kernel_size
        self.n_points = n_points
        self.init_radius = 2 * (2 / kernel_size)

        kernel_coord = rel_pos(kernel_size)
        self.register_buffer('kernel_coord', kernel_coord)

        weight_coord = torch.empty(1, n_points, 2)
        nn.init.trunc_normal_(weight_coord, std=0.2, a=-1., b=1.)
        self.weight_coord = nn.Parameter(weight_coord)

        self.radius = nn.Parameter(torch.empty(1, n_points).unsqueeze(-1).unsqueeze(-1))
        self.radius.data.fill_(value=self.init_radius)

        weights = torch.empty(1, planes, n_points)
        trunc_normal_(weights, std=.02)
        self.weights = nn.Parameter(weights)

    def forward(self, x):
        kernels = self.make_kernels().unsqueeze(1)
        x = x.contiguous()
        kernels = kernels.contiguous()

        if x.dtype == torch.float32:
            x = _DepthWiseConv2dImplicitGEMMFP32.apply(x, kernels)
        elif x.dtype == torch.float16:
            x = _DepthWiseConv2dImplicitGEMMFP16.apply(x, kernels)
        else:
            raise TypeError("Only support fp32 and fp16, get {}".format(x.dtype))
        return x

    def make_kernels(self):
        diff = self.weight_coord.unsqueeze(-2) - self.kernel_coord.reshape(1, 2, -1).transpose(1, 2)
        diff = diff.transpose(2, 3).reshape(1, self.n_points, 2, self.kernel_size, self.kernel_size)
        diff = F.relu(1 - torch.sum(torch.abs(diff), dim=2) / self.radius)

        kernels = torch.matmul(self.weights, diff.reshape(1, self.n_points, -1))
        kernels = kernels.reshape(1, self.planes, *self.kernel_coord.shape[2:])
        kernels = kernels.squeeze(0)
        kernels = torch.flip(kernels.permute(0, 2, 1), dims=(1,))
        return kernels

    def radius_clip(self, min_radius=1e-3, max_radius=1.):
        r = self.radius.data
        r = r.clamp(min_radius, max_radius)
        self.radius.data = r


def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, n_points=None):
    if n_points != None and in_channels == out_channels and out_channels == groups and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        return SMPConv(in_channels, kernel_size, n_points, stride, padding, groups)
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding, dilation=dilation, groups=groups, bias=bias)


use_sync_bn = False


def enable_sync_bn():
    global use_sync_bn
    use_sync_bn = True


def get_bn(channels):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, n_points=None):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False,
                                          n_points=n_points))
    result.add_module('bn', get_bn(out_channels))
    return result


def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, n_points=None):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=groups, dilation=dilation,
                      n_points=n_points)
    result.add_module('nonlinear', nn.ReLU())
    return result


def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std
