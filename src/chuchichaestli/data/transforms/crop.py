# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""N-dimensional cropping transforms."""

from typing import Any
from collections.abc import Sequence
import torch
from torch.nn import functional as F
from torchvision.transforms.v2 import Transform

__all__ = ["RandomCropND", "CenterCropND"]


def _as_size(size: int | Sequence[int], ndim: int | None) -> tuple[int, ...]:
    """Normalize a size argument to a per-axis tuple.

    Args:
        size: Per-axis crop size, or an int broadcast to `ndim` axes.
        ndim: Number of trailing spatial dims (required when `size` is an int).
    """
    if isinstance(size, int):
        if ndim is None:
            raise ValueError("`ndim` is required when `size` is an int.")
        return (size,) * ndim
    return tuple(size)


class RandomCropND(Transform):
    """Random crop over the trailing N spatial dims (leading dims pass through).

    Randomness lives in `make_params`, so a `SequenceCollate` draws one crop box
    per batch and applies it identically to every paired field/step.
    """

    _transformed_types = (torch.Tensor,)

    def __init__(self, size: int | Sequence[int], ndim: int | None = None):
        """Constructor.

        Args:
            size: Per-axis crop size (e.g. `(128, 128, 128)`), or an `ndim` int.
            ndim: Number of trailing spatial dims when `size` is an int.
        """
        super().__init__()
        self.size = _as_size(size, ndim)
        self.ndim = len(self.size)

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """Sample one in-bounds crop origin from the first tensor's shape."""
        x = next(i for i in flat_inputs if isinstance(i, torch.Tensor))
        spatial = x.shape[-self.ndim :]
        starts = tuple(
            int(torch.randint(0, max(1, dim - crop + 1), size=()))
            for dim, crop in zip(spatial, self.size)
        )
        return {"starts": starts, "size": self.size}

    def transform(self, x: Any, params: dict[str, Any]) -> Any:
        """Slice the shared crop box on the trailing spatial dims."""
        slices = tuple(
            slice(s, s + c) for s, c in zip(params["starts"], params["size"])
        )
        return x[(..., *slices)]


class CenterCropND(Transform):
    """Deterministic centered crop over the trailing N spatial dims.

    Symmetrically zero-pads any axis whose target size exceeds the input.
    """

    _transformed_types = (torch.Tensor,)

    def __init__(self, size: int | Sequence[int], ndim: int | None = None):
        """Constructor.

        Args:
            size: Per-axis crop size (e.g. `(128, 128, 128)`), or an int with
                `ndim`.
            ndim: Number of trailing spatial dims when `size` is an int.
        """
        super().__init__()
        self.size = _as_size(size, ndim)
        self.ndim = len(self.size)

    def transform(self, x: Any, params: dict[str, Any]) -> Any:
        """Crop (and pad if needed) the trailing spatial dims about the center."""
        spatial = x.shape[-self.ndim :]
        slices, pads = [], []
        for dim, crop in zip(spatial, self.size):
            if crop <= dim:
                start = (dim - crop) // 2
                slices.append(slice(start, start + crop))
                pads.append((0, 0))
            else:
                slices.append(slice(0, dim))
                total = crop - dim
                pads.append((total // 2, total - total // 2))
        out = x[(..., *tuple(slices))]
        if any(lo or hi for lo, hi in pads):
            pad_arg = [p for lohi in reversed(pads) for p in lohi]
            out = F.pad(out, pad_arg)
        return out
