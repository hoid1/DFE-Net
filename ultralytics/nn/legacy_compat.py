"""0526 版模块与 2107 版基类的签名兼容层。

背景
----
2107 版对若干基类的 ``__init__`` 插入了新形参（``C3k2.attn`` / ``A2C2f.version`` /
``Detect_DyHead.reg_max,end2end`` / ``Star_Block.ouc`` 等），或整体重排了形参
（``MANet``）。而从 0526 版逐字迁入的子类仍按旧顺序做位置调用，例如::

    class C3k2_XXX(C3k2):
        def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
            super().__init__(c1, c2, n, c3k, e, g, shortcut)   # 第 6 个位置是 g，不是 attn

直接运行会发生参数错位（轻则多构造一批随后被丢弃的子模块并扰乱随机数序列，
重则通道数算错、直接报错）。

设计原则
--------
**迁移进来的模块文件一个字节都不改**，兼容逻辑全部放在 2107 版基类这一侧。

判定方式不依赖参数值，而是看 ``type(self)`` 自己的 ``__init__`` 形参表：
若其中不含任何"新增形参"（sentinel），说明这次位置调用是照着 0526 版签名写的，
此时按旧顺序重新绑定；否则原样放行。因此：

* 直接实例化 2107 版基类（``type(self)`` 就是基类本身）  -> 原样放行
* 2107 版自己的 yaml 位置传参（``C3k2 [1024, True, 0.5, True]``）-> 原样放行
* 0526 版迁入的子类                                        -> 按旧签名重绑定

绑定结果与 0526 版逐位等价，已对照两版实现逐行核对。
"""

from __future__ import annotations

import functools
import inspect

__all__ = ("legacy_positional", "kwargs_adapter")

_PARAM_CACHE: dict[type, frozenset] = {}


def _own_init_params(cls: type) -> frozenset:
    """返回 cls 自身 __init__ 的形参名集合（不含 self），失败时返回空集。"""
    cached = _PARAM_CACHE.get(cls)
    if cached is not None:
        return cached
    try:
        names = frozenset(list(inspect.signature(cls.__init__).parameters)[1:])
    except (TypeError, ValueError):  # C 实现的 __init__ 等
        names = frozenset()
    _PARAM_CACHE[cls] = names
    return names


def kwargs_adapter(legacy_order, extra=None):
    """生成"按旧顺序把位置参数改成关键字"的适配函数。

    Args:
        legacy_order (Sequence[str]): 0526 版的位置参数名顺序，元素须为新签名中存在的形参名。
        extra (Callable | None): 可选，接收已绑定的 kwargs 字典并就地补充/改写。
    """

    def adapt(args, kwargs):
        bound = {legacy_order[i]: v for i, v in enumerate(args) if i < len(legacy_order)}
        bound.update(kwargs)
        if extra is not None:
            extra(bound)
        return (), bound

    return adapt


def legacy_positional(sentinels, adapter):
    """装饰 2107 版基类的 ``__init__``，为 0526 版子类的位置调用做参数重绑定。

    Args:
        sentinels (Sequence[str]): 2107 版相对 0526 版新增/改名的形参名。
            只要 ``type(self)`` 自己的 ``__init__`` 一个都不含，就判定为旧式调用。
        adapter (Callable): ``(args, kwargs) -> (new_args, new_kwargs)``。
    """
    sentinel_set = frozenset(sentinels)

    def decorator(init):
        @functools.wraps(init)
        def wrapper(self, *args, **kwargs):
            if args:
                cls = type(self)
                # 直接实例化基类本身时 cls.__init__ 就是 wrapper，不做任何处理
                if getattr(cls, "__init__", None) is not wrapper and not (sentinel_set & _own_init_params(cls)):
                    args, kwargs = adapter(args, kwargs)
            return init(self, *args, **kwargs)

        wrapper.__wrapped_legacy__ = True
        return wrapper

    return decorator
