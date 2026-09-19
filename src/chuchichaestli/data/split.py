# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dividing one dataset into disjoint parts, the same way every time."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from torch.utils.data import Dataset, Subset, random_split
from chuchichaestli.utils.rng import rng_generator


__all__ = ["split_dataset"]


def split_dataset(
    dataset: Dataset,
    fractions: Sequence[float] | Mapping[str, float],
    seed: int = 0,
    shuffle: bool = True,
) -> list[Subset] | dict[str, Subset]:
    """Divide a dataset into disjoint parts.

    Returns a mapping when given one, and a list otherwise.

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
    shares = list(fractions.values()) if names else list(fractions)
    if any(share <= 0 for share in shares):
        raise ValueError(f"Every fraction must be positive, got {shares}.")

    parts = random_split(dataset, shares, generator=rng_generator(seed, "split"))
    if not shuffle:
        start, contiguous = 0, []
        for part in parts:
            contiguous.append(Subset(dataset, list(range(start, start + len(part)))))
            start += len(part)
        parts = contiguous
    if any(len(part) < 1 for part in parts):
        raise ValueError(
            f"Splitting {len(dataset)} samples {shares} leaves an empty part: "
            f"{[len(part) for part in parts]}. Use fewer parts or more samples."
        )
    return dict(zip(names, parts)) if names else list(parts)
