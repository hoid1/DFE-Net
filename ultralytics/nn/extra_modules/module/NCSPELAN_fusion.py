'''
本文件由 0526 版迁移而来（模块代码逐字保留，未作任何修改）
来源：nn/extra_modules/block.py:4659-4759 , nn/extra_modules/block.py:4761-4771 , nn/extra_modules/block.py:4773-4777 , nn/extra_modules/block.py:4779-4783 , nn/extra_modules/block.py:4785-4789 , nn/extra_modules/block.py:4791-4802 , nn/extra_modules/block.py:4804-4808 , nn/extra_modules/block.py:4810-4814 , nn/extra_modules/block.py:4816-4820 , nn/extra_modules/block.py:4842-4846 , nn/extra_modules/block.py:4848-4852 , nn/extra_modules/block.py:4854-4858 , nn/extra_modules/orepa.py:21-305
二次创新(2)：YOLOv9 的 RepNCSPELAN4 分别融合 DBB / OREPA / DilatedReparamBlock（旧库 md #111/#112/#113）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch.nn.functional as F
import torch.nn.init as init
import math
import torch.nn as nn
import numpy as np
import torch
from ultralytics.nn.extra_modules.block.RepHMS import DilatedReparamBlock
from ultralytics.nn.extra_modules.conv_module.dbb import DiverseBranchBlock
from ultralytics.nn.extra_modules.conv_module.wdbb import transI_fusebn, transVI_multiscale
from ultralytics.nn.modules.block import RepNCSPELAN4
from ultralytics.nn.modules.conv import Conv, autopad


# ---- 原样迁移自 nn/extra_modules/block.py:4659-4759 ----
class RepConvN(nn.Module):
    """RepConv is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process"""
        if hasattr(self, 'conv'):
            return self.forward_fuse(x)
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.c1
        groups = self.g
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')

# ---- 原样迁移自 nn/extra_modules/block.py:4761-4771 ----
class RepNBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConvN(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

# ---- 原样迁移自 nn/extra_modules/block.py:4773-4777 ----
class DBBNBottleneck(RepNBottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DiverseBranchBlock(c1, c_, k[0], 1)

# ---- 原样迁移自 nn/extra_modules/block.py:4791-4802 ----
class RepNCSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

# ---- 原样迁移自 nn/extra_modules/block.py:4804-4808 ----
class DBBNCSP(RepNCSP):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DBBNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:4842-4846 ----
class DBBNCSPELAN4(RepNCSPELAN4):
    def __init__(self, c1, c2, c3, c4, c5=1):
        super().__init__(c1, c2, c3, c4, c5)
        self.cv2 = nn.Sequential(DBBNCSP(c3//2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(DBBNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))

# ---- 原样迁移自 nn/extra_modules/block.py:4785-4789 ----
class DRBNBottleneck(RepNBottleneck):
    def __init__(self, c1, c2, kernel_size, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DilatedReparamBlock(c1, kernel_size)

# ---- 原样迁移自 nn/extra_modules/block.py:4816-4820 ----
class DRBNCSP(RepNCSP):
    def __init__(self, c1, c2, n=1, kernel_size=7, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DRBNBottleneck(c_, c_, kernel_size, shortcut, g, e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:4854-4858 ----
class DRBNCSPELAN4(RepNCSPELAN4):
    def __init__(self, c1, c2, c3, c4, c5=1, c6=7):
        super().__init__(c1, c2, c3, c4, c5)
        self.cv2 = nn.Sequential(DRBNCSP(c3//2, c4, c5, c6), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(DRBNCSP(c4, c4, c5, c6), Conv(c4, c4, 3, 1))

# ---- 原样迁移自 nn/extra_modules/orepa.py:21-305 ----
class OREPA(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=None,
                 groups=1,
                 dilation=1,
                 act=True,
                 internal_channels_1x1_3x3=None,
                 deploy=False,
                 single_init=False, 
                 weight_only=False,
                 init_hyper_para=1.0, init_hyper_gamma=1.0):
        super(OREPA, self).__init__()
        self.deploy = deploy

        self.nonlinear = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.weight_only = weight_only
        
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.stride = stride
        padding = autopad(kernel_size, padding, dilation)
        self.padding = padding
        self.dilation = dilation

        if deploy:
            self.orepa_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)

        else:

            self.branch_counter = 0

            self.weight_orepa_origin = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), kernel_size, kernel_size))
            init.kaiming_uniform_(self.weight_orepa_origin, a=math.sqrt(0.0))
            self.branch_counter += 1

            self.weight_orepa_avg_conv = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), 1,
                            1))
            self.weight_orepa_pfir_conv = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), 1,
                            1))
            init.kaiming_uniform_(self.weight_orepa_avg_conv, a=0.0)
            init.kaiming_uniform_(self.weight_orepa_pfir_conv, a=0.0)
            self.register_buffer(
                'weight_orepa_avg_avg',
                torch.ones(kernel_size,
                        kernel_size).mul(1.0 / kernel_size / kernel_size))
            self.branch_counter += 1
            self.branch_counter += 1

            self.weight_orepa_1x1 = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), 1,
                            1))
            init.kaiming_uniform_(self.weight_orepa_1x1, a=0.0)
            self.branch_counter += 1

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups <= 4 else 2 * in_channels

            if internal_channels_1x1_3x3 == in_channels:
                self.weight_orepa_1x1_kxk_idconv1 = nn.Parameter(
                    torch.zeros(in_channels, int(in_channels / self.groups), 1, 1))
                id_value = np.zeros(
                    (in_channels, int(in_channels / self.groups), 1, 1))
                for i in range(in_channels):
                    id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
                id_tensor = torch.from_numpy(id_value).type_as(
                    self.weight_orepa_1x1_kxk_idconv1)
                self.register_buffer('id_tensor', id_tensor)

            else:
                self.weight_orepa_1x1_kxk_idconv1 = nn.Parameter(
                    torch.zeros(internal_channels_1x1_3x3,
                                int(in_channels / self.groups), 1, 1))
                id_value = np.zeros(
                    (internal_channels_1x1_3x3, int(in_channels / self.groups), 1, 1))
                for i in range(internal_channels_1x1_3x3):
                    id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
                id_tensor = torch.from_numpy(id_value).type_as(
                    self.weight_orepa_1x1_kxk_idconv1)
                self.register_buffer('id_tensor', id_tensor)
                #init.kaiming_uniform_(
                    #self.weight_orepa_1x1_kxk_conv1, a=math.sqrt(0.0))
            self.weight_orepa_1x1_kxk_conv2 = nn.Parameter(
                torch.Tensor(out_channels,
                            int(internal_channels_1x1_3x3 / self.groups),
                            kernel_size, kernel_size))
            init.kaiming_uniform_(self.weight_orepa_1x1_kxk_conv2, a=math.sqrt(0.0))
            self.branch_counter += 1

            expand_ratio = 8
            self.weight_orepa_gconv_dw = nn.Parameter(
                torch.Tensor(in_channels * expand_ratio, 1, kernel_size,
                            kernel_size))
            self.weight_orepa_gconv_pw = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels * expand_ratio / self.groups), 1, 1))
            init.kaiming_uniform_(self.weight_orepa_gconv_dw, a=math.sqrt(0.0))
            init.kaiming_uniform_(self.weight_orepa_gconv_pw, a=math.sqrt(0.0))
            self.branch_counter += 1

            self.vector = nn.Parameter(torch.Tensor(self.branch_counter, self.out_channels))
            if weight_only is False:
                self.bn = nn.BatchNorm2d(self.out_channels)

            self.fre_init()

            init.constant_(self.vector[0, :], 0.25 * math.sqrt(init_hyper_gamma))  #origin
            init.constant_(self.vector[1, :], 0.25 * math.sqrt(init_hyper_gamma))  #avg
            init.constant_(self.vector[2, :], 0.0 * math.sqrt(init_hyper_gamma))  #prior
            init.constant_(self.vector[3, :], 0.5 * math.sqrt(init_hyper_gamma))  #1x1_kxk
            init.constant_(self.vector[4, :], 1.0 * math.sqrt(init_hyper_gamma))  #1x1
            init.constant_(self.vector[5, :], 0.5 * math.sqrt(init_hyper_gamma))  #dws_conv

            self.weight_orepa_1x1.data = self.weight_orepa_1x1.mul(init_hyper_para)
            self.weight_orepa_origin.data = self.weight_orepa_origin.mul(init_hyper_para)
            self.weight_orepa_1x1_kxk_conv2.data = self.weight_orepa_1x1_kxk_conv2.mul(init_hyper_para)
            self.weight_orepa_avg_conv.data = self.weight_orepa_avg_conv.mul(init_hyper_para)
            self.weight_orepa_pfir_conv.data = self.weight_orepa_pfir_conv.mul(init_hyper_para)

            self.weight_orepa_gconv_dw.data = self.weight_orepa_gconv_dw.mul(math.sqrt(init_hyper_para))
            self.weight_orepa_gconv_pw.data = self.weight_orepa_gconv_pw.mul(math.sqrt(init_hyper_para))

            if single_init:
                #   Initialize the vector.weight of origin as 1 and others as 0. This is not the default setting.
                self.single_init()  

    def fre_init(self):
        prior_tensor = torch.Tensor(self.out_channels, self.kernel_size,
                                    self.kernel_size)
        half_fg = self.out_channels / 2
        for i in range(self.out_channels):
            for h in range(3):
                for w in range(3):
                    if i < half_fg:
                        prior_tensor[i, h, w] = math.cos(math.pi * (h + 0.5) *
                                                         (i + 1) / 3)
                    else:
                        prior_tensor[i, h, w] = math.cos(math.pi * (w + 0.5) *
                                                         (i + 1 - half_fg) / 3)

        self.register_buffer('weight_orepa_prior', prior_tensor)

    def weight_gen(self):
        weight_orepa_origin = torch.einsum('oihw,o->oihw',
                                          self.weight_orepa_origin,
                                          self.vector[0, :])

        weight_orepa_avg = torch.einsum('oihw,hw->oihw', self.weight_orepa_avg_conv, self.weight_orepa_avg_avg)
        weight_orepa_avg = torch.einsum(
             'oihw,o->oihw',
             torch.einsum('oi,hw->oihw', self.weight_orepa_avg_conv.squeeze(3).squeeze(2),
                          self.weight_orepa_avg_avg), self.vector[1, :])


        weight_orepa_pfir = torch.einsum(
            'oihw,o->oihw',
            torch.einsum('oi,ohw->oihw', self.weight_orepa_pfir_conv.squeeze(3).squeeze(2),
                          self.weight_orepa_prior), self.vector[2, :])

        weight_orepa_1x1_kxk_conv1 = None
        if hasattr(self, 'weight_orepa_1x1_kxk_idconv1'):
            weight_orepa_1x1_kxk_conv1 = (self.weight_orepa_1x1_kxk_idconv1 +
                                        self.id_tensor).squeeze(3).squeeze(2)
        elif hasattr(self, 'weight_orepa_1x1_kxk_conv1'):
            weight_orepa_1x1_kxk_conv1 = self.weight_orepa_1x1_kxk_conv1.squeeze(3).squeeze(2)
        else:
            raise NotImplementedError
        weight_orepa_1x1_kxk_conv2 = self.weight_orepa_1x1_kxk_conv2

        if self.groups > 1:
            g = self.groups
            t, ig = weight_orepa_1x1_kxk_conv1.size()
            o, tg, h, w = weight_orepa_1x1_kxk_conv2.size()
            weight_orepa_1x1_kxk_conv1 = weight_orepa_1x1_kxk_conv1.view(
                g, int(t / g), ig)
            weight_orepa_1x1_kxk_conv2 = weight_orepa_1x1_kxk_conv2.view(
                g, int(o / g), tg, h, w)
            weight_orepa_1x1_kxk = torch.einsum('gti,gothw->goihw',
                                              weight_orepa_1x1_kxk_conv1,
                                              weight_orepa_1x1_kxk_conv2).reshape(
                                                  o, ig, h, w)
        else:
            weight_orepa_1x1_kxk = torch.einsum('ti,othw->oihw',
                                              weight_orepa_1x1_kxk_conv1,
                                              weight_orepa_1x1_kxk_conv2)
        weight_orepa_1x1_kxk = torch.einsum('oihw,o->oihw', weight_orepa_1x1_kxk, self.vector[3, :])

        weight_orepa_1x1 = 0
        if hasattr(self, 'weight_orepa_1x1'):
            weight_orepa_1x1 = transVI_multiscale(self.weight_orepa_1x1,
                                                self.kernel_size)
            weight_orepa_1x1 = torch.einsum('oihw,o->oihw', weight_orepa_1x1,
                                           self.vector[4, :])

        weight_orepa_gconv = self.dwsc2full(self.weight_orepa_gconv_dw,
                                          self.weight_orepa_gconv_pw,
                                          self.in_channels, self.groups)
        weight_orepa_gconv = torch.einsum('oihw,o->oihw', weight_orepa_gconv,
                                        self.vector[5, :])

        weight = weight_orepa_origin + weight_orepa_avg + weight_orepa_1x1 + weight_orepa_1x1_kxk + weight_orepa_pfir + weight_orepa_gconv

        return weight

    def dwsc2full(self, weight_dw, weight_pw, groups, groups_conv=1):

        t, ig, h, w = weight_dw.size()
        o, _, _, _ = weight_pw.size()
        tg = int(t / groups)
        i = int(ig * groups)
        ogc = int(o / groups_conv)
        groups_gc = int(groups / groups_conv)
        weight_dw = weight_dw.view(groups_conv, groups_gc, tg, ig, h, w)
        weight_pw = weight_pw.squeeze().view(ogc, groups_conv, groups_gc, tg)

        weight_dsc = torch.einsum('cgtihw,ocgt->cogihw', weight_dw, weight_pw)
        return weight_dsc.reshape(o, int(i/groups_conv), h, w)

    def forward(self, inputs=None):
        if hasattr(self, 'orepa_reparam'):
            return self.nonlinear(self.orepa_reparam(inputs))
        
        weight = self.weight_gen()

        if self.weight_only is True:
            return weight

        out = F.conv2d(
            inputs,
            weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)
        return self.nonlinear(self.bn(out))

    def get_equivalent_kernel_bias(self):
        return transI_fusebn(self.weight_gen(), self.bn)

    def switch_to_deploy(self):
        if hasattr(self, 'or1x1_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.orepa_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                     kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding, dilation=self.dilation, groups=self.groups, bias=True)
        self.orepa_reparam.weight.data = kernel
        self.orepa_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('weight_orepa_origin')
        self.__delattr__('weight_orepa_1x1')
        self.__delattr__('weight_orepa_1x1_kxk_conv2')
        if hasattr(self, 'weight_orepa_1x1_kxk_idconv1'):
            self.__delattr__('id_tensor')
            self.__delattr__('weight_orepa_1x1_kxk_idconv1')
        elif hasattr(self, 'weight_orepa_1x1_kxk_conv1'):
            self.__delattr__('weight_orepa_1x1_kxk_conv1')
        else:
            raise NotImplementedError
        self.__delattr__('weight_orepa_avg_avg')  
        self.__delattr__('weight_orepa_avg_conv')
        self.__delattr__('weight_orepa_pfir_conv')
        self.__delattr__('weight_orepa_prior')
        self.__delattr__('weight_orepa_gconv_dw')
        self.__delattr__('weight_orepa_gconv_pw')

        self.__delattr__('bn')
        self.__delattr__('vector')

    def init_gamma(self, gamma_value):
        init.constant_(self.vector, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        init.constant_(self.vector[0, :], 1.0)

# ---- 原样迁移自 nn/extra_modules/block.py:4779-4783 ----
class OREPANBottleneck(RepNBottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = OREPA(c1, c_, k[0], 1)

# ---- 原样迁移自 nn/extra_modules/block.py:4810-4814 ----
class OREPANCSP(RepNCSP):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(OREPANBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

# ---- 原样迁移自 nn/extra_modules/block.py:4848-4852 ----
class OREPANCSPELAN4(RepNCSPELAN4):
    def __init__(self, c1, c2, c3, c4, c5=1):
        super().__init__(c1, c2, c3, c4, c5)
        self.cv2 = nn.Sequential(OREPANCSP(c3//2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(OREPANCSP(c4, c4, c5), Conv(c4, c4, 3, 1))

if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 64, 128, 32, 32
    inputs = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- OREPANBottleneck ----
    try:
        module = OREPANBottleneck(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'OREPANBottleneck  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'OREPANBottleneck  自测跳过: {e}' + RESET)
    # ---- OREPANCSP ----
    try:
        module = OREPANCSP(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'OREPANCSP  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'OREPANCSP  自测跳过: {e}' + RESET)
    # ---- OREPANCSPELAN4 ----
    try:
        module = OREPANCSPELAN4(in_channel, out_channel).to(device)
        outputs = module(inputs)
        print(GREEN + f'OREPANCSPELAN4  inputs:{tuple(inputs.shape)} -> outputs:{tuple(outputs.shape) if hasattr(outputs, "shape") else type(outputs)}' + RESET)
        try:
            from calflops import calculate_flops
            print(ORANGE, end='')
            calculate_flops(model=module, input_shape=(batch_size, in_channel, height, width),
                            output_as_string=True, output_precision=4, print_detailed=False)
            print(RESET, end='')
        except Exception:
            pass
    except Exception as e:
        print(RED + f'OREPANCSPELAN4  自测跳过: {e}' + RESET)

