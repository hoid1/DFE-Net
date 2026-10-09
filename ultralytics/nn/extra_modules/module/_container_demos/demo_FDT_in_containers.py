'''
本文件演示如何把迁移进来的 0526 版模块 FDT 塞进新版三大通用容器
（ResidualBlock / C3k2_Block 系 / MANet）做二次创新组合。

FDT 签名 (inp_channels, num_heads=4, window_sizes=4, shifts=0, shared_depth=1, ffn_expansion_factor=2.66)，
forward(x) 单张量输入、输出同通道 —— 天然满足容器对"注意力槽位"的调用契约
(dim) -> Module，不需要改一行 FDT 自身的代码，只需要 partial() 固定超参。

来源：nn/extra_modules/block.py:11270（旧库 md，二次创新(2) 引用模块，出自 CVPR 频域 Transformer 论文）
'''

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from functools import partial

from ultralytics.nn.extra_modules.module.legacy_block_kernels import FDT
from ultralytics.nn.extra_modules.block.ResBlock import ResidualBlock
from ultralytics.nn.modules.conv import Conv


if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size, in_channel, out_channel, height, width = 1, 128, 128, 32, 32
    inputs_tensor = torch.randn((batch_size, in_channel, height, width)).to(device)

    # ---- 用法一：FDT 直接作为 ResidualBlock 的 attention 槽位 ----
    module1_instance = partial(Conv, k=3, g=1)
    module2_instance = partial(Conv, k=3, g=1)
    # FDT 要求 in_channel == out_channel（它是同通道模块），attention 槽位天然满足这个约束
    attention_instance = partial(FDT, num_heads=4, window_sizes=4)

    resblock = ResidualBlock(in_channel, out_channel, module1_instance, module2_instance,
                             attention_instance, shortcut=True, e=0.5).to(device)
    resblock.eval()
    outputs = resblock(inputs_tensor)
    print(GREEN + f'ResidualBlock+FDT  inputs:{inputs_tensor.size()} outputs:{outputs.size()}' + RESET)

    print(ORANGE)
    try:
        from calflops import calculate_flops
        calculate_flops(model=resblock, input_shape=(batch_size, in_channel, height, width),
                        output_as_string=True, output_precision=4, print_detailed=False)
    except Exception as e:
        print('calflops 不可用:', e)
    print(RESET)

    def _module_cfg_to_str(module_cfg):
        if module_cfg is None:
            return "None"
        if isinstance(module_cfg, partial):
            module, kwargs = module_cfg.func, module_cfg.keywords or {}
        else:
            module, kwargs = module_cfg, {}
        name = getattr(module, "__name__", str(module))
        kw = ", ".join(f"{k}: {repr(v)}" for k, v in kwargs.items())
        return f"{{'module': {name}, 'param': {{{kw}}}}}"

    yaml_copy_template = (
        "{'module': ResidualBlock, 'param': {"
        f"'module1': {_module_cfg_to_str(module1_instance)}, "
        f"'module2': {_module_cfg_to_str(module2_instance)}, "
        f"'attention': {_module_cfg_to_str(attention_instance)}, "
        f"'shortcut': True, 'e': 0.5"
        "}}"
    )
    print(BLUE + "YAML copy template for current ResidualBlock+FDT combination:" + RESET)
    print(yaml_copy_template)

    # ---- 用法二：FDT 直接作为 C3k2_Block 的 selfatt 内核（单参数分支）----
    from ultralytics.nn.extra_modules.block.CSPBlock import C3k2_Block

    fdt_kernel = partial(FDT, num_heads=4, window_sizes=4)
    c3k2_fdt = C3k2_Block(in_channel, out_channel, module=fdt_kernel, n=2, c3k=False, selfatt=True).to(device)
    c3k2_fdt.eval()
    out2 = c3k2_fdt(inputs_tensor)
    print(GREEN + f'\nC3k2_Block(selfatt)+FDT  inputs:{inputs_tensor.size()} outputs:{out2.size()}' + RESET)

    yaml_copy_template_2 = (
        f"[-1, 2, C3k2_Block, [{out_channel}, "
        "{'module': FDT, 'param': {'num_heads': 4, 'window_sizes': 4}}, "
        "{'selfatt': True}]]"
    )
    print(BLUE + "\nYAML copy template for current C3k2_Block+FDT combination:" + RESET)
    print(yaml_copy_template_2)
