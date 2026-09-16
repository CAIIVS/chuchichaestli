# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dividing one dataset into disjoint parts, the same way every time."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
import torch
from torch.utils.data import Subset
from chuchichaestli.utils.rng import rng_generator


__all__ = ["split_dataset", "split_sizes"]


def split_sizes(total: int, fractions: Sequence[float]) -> list[int]:
    """Return how many samples each fraction gets, summing to `total`.

    The largest remainders take the leftovers, so no sample is lost or
    counted twice.

    Args:
        total: Samples to divide.
        fractions: Positive shares, summing to one.

    Raises:
        ValueError: If the fractions are unusable, or leave a part empty.
    """
    if not fractions:
        raise ValueError("A split needs at least one fraction.")
    if any(fraction <= 0 for fraction in fractions):
        raise ValueError(f"Every fraction must be positive, got {list(fractions)}.")
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to one, got {sum(fractions)}.")
    exact = [fraction * total for fraction in fractions]
    sizes = [int(value) for value in exact]
    by_remainder = sorted(
        range(len(exact)), key=lambda i: exact[i] - sizes[i], reverse=True
    )
    for index in by_remainder[: total - sum(sizes)]:
        sizes[index] += 1
    if any(size < 1 for size in sizes):
        raise ValueError(
            f"Splitting {total} samples {list(fractions)} leaves an empty part: "
            f"{sizes}. Use fewer parts or more samples."
        )
    return sizes


def split_dataset(
    dataset: Any,
    fractions: Sequence[float] | Mapping[str, float],
    seed: int = 0,
    shuffle: bool = True,
) -> list[Subset] | dict[str, Subset]:
    """Divide a dataset into disjoint parts.

    Returns a mapping when given one, and a list otherwise. The same seed and
    fractions always give the same parts, in this process and the next.

    Args:
        dataset: What to divide.
        fractions: Positive shares summing to one, named or not.
        seed: Seed the permutation derives from.
        shuffle: Whether to permute before dividing; `False` keeps the
            dataset's own order, so the parts are contiguous.

    Raises:
        ValueError: If the fractions are unusable, or leave a part empty.
    """
    names = list(fractions) if isinstance(fractions, Mapping) else None
    values = (
        list(fractions.values()) if isinstance(fractions, Mapping) else list(fractions)
    )
    total = len(dataset)
    sizes = split_sizes(total, values)

    if shuffle:
        order = torch.randperm(total, generator=rng_generator(seed, "split")).tolist()
    else:
        order = list(range(total))
    parts, start = [], 0
    for size in sizes:
        parts.append(Subset(dataset, order[start : start + size]))
        start += size
    return dict(zip(names, parts)) if names is not None else parts
