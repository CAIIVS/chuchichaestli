# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Various utility functions for chuchichaestli."""

import sys
import torch
from functools import partialmethod, wraps
from collections.abc import Sequence, Iterable, Callable


def partialclass(name: str, cls: type[object], *args, **kwargs):
    """Partial for __init__ class constructors."""
    docstring = kwargs.pop("__doc__", None)
    part_cls = type(
        name, (cls,), {"__init__": partialmethod(cls.__init__, *args, **kwargs)}
    )
    try:
        part_cls.__module__ = sys._getframe(1).f_globals.get("__name__", "__main__")
    except (AttributeError, ValueError):
        pass
    if docstring is not None:
        part_cls.__doc__ = docstring
    return part_cls


def alias_kwargs(key: str | dict[str, str], alias: str | None = None) -> Callable:
    """Decorator for aliasing keyword arguments in a function.

    Args:
        key: Name of keyword argument in function to alias, or a dictionary mapping keywords to aliases.
        alias: Alias that can be used for the specified keyword argument.
    """
    if alias is None and isinstance(key, dict):
        alias = key.values()
        key = key.keys()
    if isinstance(key, str):
        key = (key,)
    if isinstance(alias, str):
        alias = (alias,)

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for k, a in zip(key, alias):
                alias_value = kwargs.get(a, None)
                if alias_value is not None:
                    kwargs[k] = alias_value
                if a in kwargs:
                    del kwargs[a]
            result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator


def prod(num_list: Iterable[int] | torch.Size) -> int:
    """Calculate the product of all elements in an iterable (analogous to built-in sum).

    Args:
        num_list: List with numerical values to be multiplied.
    """
    result = 1
    if isinstance(num_list, Iterable):
        for item in num_list:
            result *= prod(item) if isinstance(item, Iterable) else item
    return result
