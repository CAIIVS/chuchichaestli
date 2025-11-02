# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Various higher-order functions and operations for chuchichaestli."""

import sys
from functools import partialmethod
import torch
import numpy as np
from typing import Any
from collections.abc import Sequence, Iterable, Callable

__all__ = ["partialclass", "nested_list_size", "prod", "map_nested"]


def partialclass(name: str, cls: type[object], *args, **kwargs):
    """Partial for __init__ class constructors."""
    part_cls = type(
        name, (cls,), {"__init__": partialmethod(cls.__init__, *args, **kwargs)}
    )
    try:
        part_cls.__module__ = sys._getframe(1).f_globals.get("__name__", "__main__")
    except (AttributeError, ValueError):
        pass
    return part_cls


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
    aggregate_fn: Callable[..., Any] = lambda result: result,
) -> Any:
    """Traverse any type of nested data.

    Maps `action_fn` onto members, and aggregates the results using `aggregate_fn`.

    Args:
        data: Arbitrarily nested data structure.
        action_fn: Function to be mapped onto data.
        aggregate_fn: Aggregate function for map results.
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
