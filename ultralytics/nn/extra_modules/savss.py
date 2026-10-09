'''
补全文件：原 0526 版 nn/extra_modules/savss.py 中的 get_norm_layer 辅助函数。

背景：ultralytics/nn/extra_modules/mamba/legacy_mamba_kernels.py 第18行有
`from ultralytics.nn.extra_modules.savss import get_norm_layer`，但迁移压缩包
里只搬运了 savss.py 里的几个类（GBC/PAF/...，见该文件里的"原样迁移自"注释），
没有把 get_norm_layer 这个小工具函数一起带过来，导致 import 链在
extra_modules/__init__.py 加载到 mamba 分支时直接报
`ModuleNotFoundError: No module named 'ultralytics.nn.extra_modules.savss'`，
从而整个 ultralytics 包都 import 不了（不管你用不用 mamba 系列模块）。

这里按 legacy_mamba_kernels.py 里唯一的调用方式：
    get_norm_layer(norm_type, in_channels, in_channels // 16)  # 以及 .., 16
还原一个标准写法的 norm layer 工厂（GroupNorm / BatchNorm / InstanceNorm），
和业界常见的同名实现（如 MedNeXt / nnU-Net 风格）语义一致。项目里目前只用到了
norm_type='GN' 这一种，其余分支按通用约定补全，供以后其他 norm_type 调用时也不报错。
'''

import torch.nn as nn


def get_norm_layer(norm_type: str, channels: int, num_groups: int = 1) -> nn.Module:
    """Build a normalization layer by name.

    Args:
        norm_type: one of {'GN', 'BN', 'IN', 'LN'} (case-insensitive).
        channels: number of channels the norm layer is applied to.
        num_groups: number of groups for GroupNorm (ignored by other norm types).
                    Falls back to 1 if `channels` isn't divisible by it.
    """
    norm_type = (norm_type or 'GN').upper()

    if norm_type == 'GN':
        g = max(1, int(num_groups))
        if channels % g != 0:
            # fall back to the largest divisor of channels that is <= g
            for cand in range(g, 0, -1):
                if channels % cand == 0:
                    g = cand
                    break
        return nn.GroupNorm(g, channels)
    elif norm_type == 'BN':
        return nn.BatchNorm2d(channels)
    elif norm_type == 'IN':
        return nn.InstanceNorm2d(channels, affine=True)
    elif norm_type == 'LN':
        return nn.GroupNorm(1, channels)  # LayerNorm-equivalent for NCHW tensors
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type!r} (expected GN/BN/IN/LN)")
