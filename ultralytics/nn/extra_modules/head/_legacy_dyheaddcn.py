'''
本文件由 0526 版迁移而来 —— 检测头本体，逐字保留，未作任何修改
来源：nn/extra_modules/DCNv4_op/DCNv4/functions/dcnv4_func.py:62-129 , nn/extra_modules/DCNv4_op/DCNv4/modules/dcnv4.py:155-237 , nn/extra_modules/DCNv4_op/DCNv4/modules/dcnv4.py:18-26 , nn/extra_modules/block.py:373-456 , nn/extra_modules/block.py:463-537 , nn/extra_modules/head.py:88-91 , nn/extra_modules/head.py:93-96 , nn/extra_modules/ops_dcnv3/functions/dcnv3_func.py:23-107 , nn/extra_modules/ops_dcnv3/modules/dcnv3.py:221-338 , nn/extra_modules/ops_dcnv3/modules/dcnv3.py:340-422
D 组：DyHead + DCNv3 / DCNv4（旧库 md #89 等）。
注意：需先编译 compile_module/ops_dcnv3 与 compile_module/DCNv4_op，且依赖 mmcv/mmengine。

请勿直接在 yaml 中引用本文件的类，yaml 请使用同目录 DyHeadDCN.py 中的同名适配类。
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn.functional as F
from torch.autograd import Function
from mmcv.cnn import build_activation_layer
from torch.nn.init import constant_
from mmengine.model import constant_init
from torch.cuda.amp import custom_bwd
from torch.cuda.amp import custom_fwd
import pkg_resources
try:
    dcn_version = float(pkg_resources.get_distribution('DCNv3').version)
except Exception:
    dcn_version = 0.0
from DCNv4 import ext
import DCNv3 as _dcnv3_ext
from ultralytics.nn.extra_modules.DCNv4_op.DCNv4.functions.dcnv4_func import find_spec_bwd
from ultralytics.nn.extra_modules.DCNv4_op.DCNv4.functions.dcnv4_func import findspec
import math
import torch.nn as nn
from mmengine.model import normal_init
from torch.autograd.function import once_differentiable
import torch
import warnings
from torch.nn.init import xavier_uniform_
from ultralytics.nn.extra_modules.featurefusion.SFSFusion import _is_power_of_2
from ultralytics.nn.extra_modules.head.DyHead import Detect_DyHead, DyReLU
from ultralytics.nn.modules.conv import Conv


# ---- 原样迁移自 nn/extra_modules/DCNv4_op/DCNv4/modules/dcnv4.py:18-26 ----
class CenterFeatureScaleModule(nn.Module):
    def forward(self,
                query,
                center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query,
                                        weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale

# ---- 原样迁移自 nn/extra_modules/ops_dcnv3/functions/dcnv3_func.py:23-107 ----
class DCNv3Function(Function):
    @staticmethod
    @custom_fwd
    def forward(
            ctx, input, offset, mask,
            kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w, dilation_h, dilation_w,
            group, group_channels, offset_scale, im2col_step, remove_center):
        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w
        ctx.group = group
        ctx.group_channels = group_channels
        ctx.offset_scale = offset_scale
        ctx.im2col_step = im2col_step
        ctx.remove_center = remove_center

        args = [
            input, offset, mask, kernel_h,
            kernel_w, stride_h, stride_w, pad_h,
            pad_w, dilation_h, dilation_w, group,
            group_channels, offset_scale, ctx.im2col_step
        ]
        if remove_center or dcn_version > 1.0:
            args.append(remove_center)

        output = _dcnv3_ext.dcnv3_forward(*args)
        ctx.save_for_backward(input, offset, mask)

        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        input, offset, mask = ctx.saved_tensors

        args = [
            input, offset, mask, ctx.kernel_h,
            ctx.kernel_w, ctx.stride_h, ctx.stride_w, ctx.pad_h,
            ctx.pad_w, ctx.dilation_h, ctx.dilation_w, ctx.group,
            ctx.group_channels, ctx.offset_scale, grad_output.contiguous(), ctx.im2col_step
        ]
        if ctx.remove_center or dcn_version > 1.0:
            args.append(ctx.remove_center)

        grad_input, grad_offset, grad_mask = \
            _dcnv3_ext.dcnv3_backward(*args)

        return grad_input, grad_offset, grad_mask, \
            None, None, None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def symbolic(g, input, offset, mask, kernel_h, kernel_w, stride_h,
                 stride_w, pad_h, pad_w, dilation_h, dilation_w, group,
                 group_channels, offset_scale, im2col_step, remove_center):
        """Symbolic function for mmdeploy::DCNv3.

        Returns:
            DCNv3 op for onnx.
        """
        return g.op(
            'mmdeploy::TRTDCNv3',
            input,
            offset,
            mask,
            kernel_h_i=int(kernel_h),
            kernel_w_i=int(kernel_w),
            stride_h_i=int(stride_h),
            stride_w_i=int(stride_w),
            pad_h_i=int(pad_h),
            pad_w_i=int(pad_w),
            dilation_h_i=int(dilation_h),
            dilation_w_i=int(dilation_w),
            group_i=int(group),
            group_channels_i=int(group_channels),
            offset_scale_f=float(offset_scale),
            im2col_step_i=int(im2col_step),
            remove_center=int(remove_center),
        )

# ---- 原样迁移自 nn/extra_modules/ops_dcnv3/modules/dcnv3.py:221-338 ----
class DCNv3(nn.Module):
    def __init__(
        self,
        channels=64,
        kernel_size=3,
        stride=1,
        pad=1,
        dilation=1,
        group=4,
        offset_scale=1.0,
        center_feature_scale=False,
        remove_center=False,
    ):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        if self.remove_center and self.kernel_size % 2 == 0:
            raise ValueError('remove_center is only compatible with odd kernel size.')

        self.dw_conv = Conv(channels, channels, dw_kernel_size, g=channels)
        self.offset = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center) * 2)
        self.mask = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center))
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()
        
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        N, H, W, _ = input.shape

        x = self.input_proj(input)
        x_proj = x
        dtype = x.dtype

        x1 = input.permute(0, 3, 1, 2)
        x1 = self.dw_conv(x1).permute(0, 2, 3, 1)
        offset = self.offset(x1)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1)
        mask = mask.reshape(N, H, W, -1).type(dtype)
        
        x = DCNv3Function.apply(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256,
            self.remove_center)
        
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)

        return x

# ---- 原样迁移自 nn/extra_modules/ops_dcnv3/modules/dcnv3.py:340-422 ----
class DCNv3_DyHead(nn.Module):
    def __init__(
        self,
        channels=64,
        kernel_size=3,
        stride=1,
        pad=1,
        dilation=1,
        group=4,
        offset_scale=1.0,
        center_feature_scale=False,
        remove_center=False,
    ):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        if self.remove_center and self.kernel_size % 2 == 0:
            raise ValueError('remove_center is only compatible with odd kernel size.')

        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()
        
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, x, offset, mask):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        x = DCNv3Function.apply(
            x.permute(0, 2, 3, 1).contiguous(), offset.contiguous(), mask.contiguous(),
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256,
            self.remove_center)
        
        x = self.output_proj(x)
        return x.permute(0, 3, 1, 2)

# ---- 原样迁移自 nn/extra_modules/DCNv4_op/DCNv4/functions/dcnv4_func.py:62-129 ----
class DCNv4Function(Function):
    @staticmethod
    @custom_fwd
    def forward(
            ctx, input, offset_mask,
            kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w, dilation_h, dilation_w,
            group, group_channels, offset_scale,
            im2col_step, remove_center):

        forward_d_stride, forward_block_thread = findspec(input.shape[0], input.shape[1], input.shape[2], group, group_channels)
        backward_d_stride, backward_block_thread = find_spec_bwd(input.shape[0], input.shape[1], input.shape[2], group, group_channels)

        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w
        ctx.group = group
        ctx.group_channels = group_channels
        ctx.offset_scale = offset_scale
        ctx.im2col_step = im2col_step
        ctx.remove_center = remove_center
        ctx.backward_d_stride = backward_d_stride
        ctx.backward_block_thread = backward_block_thread

        args = [
            input, offset_mask, kernel_h,
            kernel_w, stride_h, stride_w, pad_h,
            pad_w, dilation_h, dilation_w, group,
            group_channels, offset_scale,
            ctx.im2col_step,
            remove_center,
            forward_d_stride,
            forward_block_thread,
            False,
        ]

        output = ext.dcnv4_forward(*args)
        ctx.save_for_backward(input, offset_mask)

        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        input, offset_mask = ctx.saved_tensors

        args = [
            input, offset_mask, ctx.kernel_h,
            ctx.kernel_w, ctx.stride_h, ctx.stride_w, ctx.pad_h,
            ctx.pad_w, ctx.dilation_h, ctx.dilation_w, ctx.group,
            ctx.group_channels, ctx.offset_scale, ctx.im2col_step,
            grad_output.contiguous(), ctx.remove_center,
            ctx.backward_d_stride, ctx.backward_block_thread,
            False
        ]

        grad_input, grad_offset_mask = \
            ext.dcnv4_backward(*args)

        return grad_input, grad_offset_mask, \
            None, None, None, None, None, None, None,\
            None, None, None, None, None, None

# ---- 原样迁移自 nn/extra_modules/DCNv4_op/DCNv4/modules/dcnv4.py:155-237 ----
class DCNv4_Dyhead(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            dw_kernel_size=None,
            center_feature_scale=False,
            remove_center=False,
            output_bias=True,
            without_pointwise=False,
            **kwargs):
        """
        DCNv4 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group

        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        # assert _d_per_group % 16 == 0

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.dw_kernel_size = dw_kernel_size
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)
        self.without_pointwise = without_pointwise

        if not without_pointwise:
            self.output_proj = nn.Linear(channels, channels, bias=output_bias)
        self._reset_parameters()

    def _reset_parameters(self):
        if not self.without_pointwise:
            xavier_uniform_(self.output_proj.weight.data)
            if self.output_proj.bias is not None:
                constant_(self.output_proj.bias.data, 0.)

    def forward(self, x, offset_mask):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        N, C, H, W = x.size()
        x = DCNv4Function.apply(
            x.permute(0, 2, 3, 1).contiguous(), offset_mask.contiguous(),
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256,
            self.remove_center
            )

        x = x.view(N, H * W // self.stride // self.stride, -1)
        if not self.without_pointwise:
            x = self.output_proj(x)
        return x.transpose(2, 1).reshape((N, C, H // self.stride, W // self.stride))

# ---- 原样迁移自 nn/extra_modules/block.py:373-456 ----
class DyHeadBlockWithDCNV3(nn.Module):
    """DyHead Block with three types of attention.
    HSigmoid arguments in default act_cfg follow official code, not paper.
    https://github.com/microsoft/DynamicHead/blob/master/dyhead/dyrelu.py
    """

    def __init__(self,
                 in_channels,
                 norm_type='GN',
                 zero_init_offset=True,
                 act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)):
        super().__init__()
        self.zero_init_offset = zero_init_offset
        # (offset_x, offset_y, mask) * kernel_size_y * kernel_size_x
        self.offset_and_mask_dim = 3 * 4 * 3 * 3
        self.offset_dim = 2 * 4 * 3 * 3
        
        self.dw_conv_high = Conv(in_channels, in_channels, 3, g=in_channels)
        self.dw_conv_mid = Conv(in_channels, in_channels, 3, g=in_channels)
        self.dw_conv_low = Conv(in_channels, in_channels, 3, g=in_channels)
        
        self.spatial_conv_high = DCNv3_DyHead(in_channels)
        self.spatial_conv_mid = DCNv3_DyHead(in_channels)
        self.spatial_conv_low = DCNv3_DyHead(in_channels, stride=2)
        self.spatial_conv_offset = nn.Conv2d(
            in_channels, self.offset_and_mask_dim, 3, padding=1, groups=4)
        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, 1, 1),
            nn.ReLU(inplace=True), build_activation_layer(act_cfg))
        self.task_attn_module = DyReLU(in_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)
        if self.zero_init_offset:
            constant_init(self.spatial_conv_offset, 0)

    def forward(self, x):
        """Forward function."""
        outs = []
        for level in range(len(x)):
            # calculate offset and mask of DCNv2 from middle-level feature
            mid_feat_ = self.dw_conv_mid(x[level])
            offset_and_mask = self.spatial_conv_offset(mid_feat_)
            offset = offset_and_mask[:, :self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()

            mid_feat = self.spatial_conv_mid(x[level], offset, mask)
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if level > 0:
                low_feat_ = self.dw_conv_low(x[level - 1])
                offset, mask = self.get_offset_mask(low_feat_)
                low_feat = self.spatial_conv_low(x[level - 1], offset, mask)
                sum_feat += low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(x) - 1:
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                high_feat_ = self.dw_conv_high(x[level + 1])
                offset, mask = self.get_offset_mask(high_feat_)
                high_feat = F.interpolate(
                    self.spatial_conv_high(x[level + 1], offset, mask),
                    size=x[level].shape[-2:],
                    mode='bilinear',
                    align_corners=True)
                sum_feat += high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1
            outs.append(self.task_attn_module(sum_feat / summed_levels))

        return outs
    
    def get_offset_mask(self, x):
        N, _, H, W = x.size()
        dtype = x.dtype
        
        offset_and_mask = self.spatial_conv_offset(x).permute(0, 2, 3, 1)
        offset = offset_and_mask[..., :self.offset_dim]
        mask = offset_and_mask[..., self.offset_dim:].reshape(N, H, W, 4, -1)
        mask = F.softmax(mask, -1)
        mask = mask.reshape(N, H, W, -1).type(dtype)
        return offset, mask

# ---- 原样迁移自 nn/extra_modules/head.py:88-91 ----
class Detect_DyHeadWithDCNV3(Detect_DyHead):
    def __init__(self, nc=80, hidc=256, block_num=2, ch=()):
        super().__init__(nc, hidc, block_num, ch)
        self.dyhead = nn.Sequential(*[DyHeadBlockWithDCNV3(hidc) for i in range(block_num)])

# ---- 原样迁移自 nn/extra_modules/block.py:463-537 ----
class DyHeadBlockWithDCNV4(nn.Module):
    """DyHead Block with three types of attention.
    HSigmoid arguments in default act_cfg follow official code, not paper.
    https://github.com/microsoft/DynamicHead/blob/master/dyhead/dyrelu.py
    """

    def __init__(self,
                 in_channels,
                 norm_type='GN',
                 zero_init_offset=True,
                 act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)):
        super().__init__()
        self.zero_init_offset = zero_init_offset
        # (offset_x, offset_y, mask) * kernel_size_y * kernel_size_x
        self.offset_and_mask_dim = int(math.ceil((9 * 3)/8)*8)
        
        self.dw_conv_high = Conv(in_channels, in_channels, 3, g=in_channels)
        self.dw_conv_mid = Conv(in_channels, in_channels, 3, g=in_channels)
        self.dw_conv_low = Conv(in_channels, in_channels, 3, g=in_channels)
        
        self.spatial_conv_high = DCNv4_Dyhead(in_channels, group=1)
        self.spatial_conv_mid = DCNv4_Dyhead(in_channels, group=1)
        self.spatial_conv_low = DCNv4_Dyhead(in_channels, group=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.spatial_conv_offset = nn.Conv2d(
            in_channels, self.offset_and_mask_dim, 1, padding=0, groups=1)
        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, 1, 1),
            nn.ReLU(inplace=True), build_activation_layer(act_cfg))
        self.task_attn_module = DyReLU(in_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)
        if self.zero_init_offset:
            constant_init(self.spatial_conv_offset, 0)

    def forward(self, x):
        """Forward function."""
        outs = []
        for level in range(len(x)):
            # calculate offset and mask of DCNv2 from middle-level feature
            mid_feat_ = self.dw_conv_mid(x[level])
            offset_and_mask = self.get_offset_mask(mid_feat_)

            mid_feat = self.spatial_conv_mid(x[level], offset_and_mask)
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if level > 0:
                low_feat_ = self.dw_conv_low(x[level - 1])
                offset_and_mask = self.get_offset_mask(low_feat_)
                low_feat = self.spatial_conv_low(x[level - 1], offset_and_mask)
                low_feat = self.maxpool(low_feat)
                sum_feat += low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(x) - 1:
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                high_feat_ = self.dw_conv_high(x[level + 1])
                offset_and_mask = self.get_offset_mask(high_feat_)
                high_feat = F.interpolate(
                    self.spatial_conv_high(x[level + 1], offset_and_mask),
                    size=x[level].shape[-2:],
                    mode='bilinear',
                    align_corners=True)
                sum_feat += high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1
            outs.append(self.task_attn_module(sum_feat / summed_levels))
        return outs
    
    def get_offset_mask(self, x):
        offset_mask = self.spatial_conv_offset(x).permute(0, 2, 3, 1)
        return offset_mask

# ---- 原样迁移自 nn/extra_modules/head.py:93-96 ----
class Detect_DyHeadWithDCNV4(Detect_DyHead):
    def __init__(self, nc=80, hidc=256, block_num=2, ch=()):
        super().__init__(nc, hidc, block_num, ch)
        self.dyhead = nn.Sequential(*[DyHeadBlockWithDCNV4(hidc) for i in range(block_num)])
