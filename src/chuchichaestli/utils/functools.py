# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Various higher-order functions and operations for chuchichaestli."""

import sys
from functools import partialmethod, wraps
import torch
import numpy as np
from typing import Any
from collections.abc import Sequence, Iterable, Callable

__all__ = [
    "partialclass",
    "alias_kwargs",
    "nested_list_size",
    "prod",
    "map_nested",
    "per_position",
    "per_position_args",
]


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


def nested_list_size(inputs: Sequence[Any] | torch.Tensor) -> tuple[list[int], int]:
    """Flattens nested list size.

    Args:
        inputs: Any form of nested list or tensor(s).

    Returns:
        - List of size(s).
        - Element byte size.
    """
    if hasattr(inputs, "tensors"):
        size, elem_bytes = nested_list_size(inputs.tensors)
    elif isinstance(inputs, torch.Tensor):
        size, elem_bytes = list(inputs.size()), inputs.element_size()
    elif isinstance(inputs, np.ndarray):  # type: ignore[unreachable]
        # preserves dtype
        inputs_torch = torch.from_numpy(inputs)  # type: ignore[unreachable]
        size, elem_bytes = list(inputs_torch.size()), inputs_torch.element_size()
    elif not hasattr(inputs, "__getitem__") or not inputs:
        size, elem_bytes = [], 0
    elif isinstance(inputs, dict):
        size, elem_bytes = nested_list_size(list(inputs.values()))
    elif (
        hasattr(inputs, "size")
        and callable(inputs.size)
        and hasattr(inputs, "element_size")
        and callable(inputs.element_size)
    ):
        size, elem_bytes = list(inputs.size()), inputs.element_size()
    elif isinstance(inputs, (list, tuple)):
        size, elem_bytes = nested_list_size(inputs[0])
    else:
        size, elem_bytes = [], 0

    return size, elem_bytes


def map_nested(
    data: Any,
    action_fn: Callable[..., Any],
    aggregate_fn: Callable[..., Any] = type,
) -> Any:
    """Traverse any type of nested data.

    Maps `action_fn` onto members, and aggregates the results using `aggregate_fn`.

    Args:
        data: Arbitrarily nested data structure.
        action_fn: Function to be mapped onto data.
        aggregate_fn: Given a container, returns the callable used to rebuild it
            from the mapped members; defaults to `type`, which reconstructs
            lists, tuples, dicts, and named tuples in place.
    """
    if isinstance(data, torch.Tensor):
        result = action_fn(data)
    elif isinstance(data, np.ndarray):
        result = action_fn(torch.from_numpy(data))
        # if result is a tensor, then action_fn was meant for tensors only.
        if isinstance(result, torch.Tensor):
            result = data
    elif isinstance(data, dict):
        aggregate = aggregate_fn(data)
        result = aggregate(
            {k: map_nested(v, action_fn, aggregate_fn) for k, v in data.items()}
        )
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # Named tuple
        aggregate = aggregate_fn(data)
        result = aggregate(*(map_nested(d, action_fn, aggregate_fn) for d in data))
    elif isinstance(data, Iterable) and not isinstance(data, str):
        aggregate = aggregate_fn(data)
        result = aggregate([map_nested(d, action_fn, aggregate_fn) for d in data])
    else:
        # data not a collection
        result = data
    return result


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


def per_position(
    value: Any,
    n: int,
    name: str = "value",
    mask: Sequence[bool] | None = None,
    context: str = "",
) -> tuple:
    """Broadcast a single value, or validate a sequence with one entry per position.

    Only lists and tuples are read as per-position; strings, `None`, and numbers
    are single values and are broadcast to every position.

    Args:
        value: A single value, or a list/tuple with one entry per position.
        n: Number of positions.
        name: Argument name, used in the error message.
        mask: Marks the positions the value has an effect on. A sequence as long
            as the marked positions is scattered onto them, and the remaining
            entries are filled with its first element.
        context: Description of the positions, used in the error message.

    Raises:
        ValueError: If `value` is a sequence whose length matches no accepted count.
    """
    if not isinstance(value, (list, tuple)):
        return (value,) * n
    if len(value) == n:
        return tuple(value)
    n_masked = sum(mask) if mask is not None else 0
    if n_masked and len(value) == n_masked:
        out, it = [value[0]] * n, iter(value)
        for p, marked in enumerate(mask):
            if marked:
                out[p] = next(it)
        return tuple(out)
    where = f" for {context}" if context else ""
    alt = f", or {n_masked} for the blocks it applies to" if n_masked else ""
    raise ValueError(
        f"{name}: expected a single value or {n} values{where}{alt}; got {len(value)}."
    )


def per_position_args(
    spec: dict[str, Any],
    n: int,
    mask: Sequence[bool] | None = None,
    opaque: Sequence[str] = (),
    context: str = "",
) -> list[dict[str, Any]]:
    """Expand a mapping of single-or-per-position values into one dict per position.

    Args:
        spec: Mapping of argument name to a single value or a per-position sequence.
        n: Number of positions.
        mask: Marks the positions the values have an effect on (see `per_position`).
        opaque: Keys whose value is passed through untouched, however it is shaped.
        context: Description of the positions, used in error messages.
    """
    expanded = {
        key: (value,) * n
        if key in opaque
        else per_position(value, n, key, mask, context)
        for key, value in spec.items()
    }
    return [{key: values[p] for key, values in expanded.items()} for p in range(n)]
