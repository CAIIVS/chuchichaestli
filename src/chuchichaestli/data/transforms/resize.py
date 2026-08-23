# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""N-dimensional resize and pad transforms."""

from typing import Any
from collections.abc import Sequence
import torch
from torch.nn import functional as F
from torchvision.transforms.v2 import Transform

__all__ = ["ResizeND", "PadND"]

_ALIGN_MODES = ("linear", "bilinear", "bicubic", "trilinear")


class ResizeND(Transform):
    """Resize the trailing N spatial dims via `torch.nn.functional.interpolate`.

    Bare `(*spatial)` and `(C, *spatial)` leaves are supported by temporarily
    unsqueezing to the `(N, C, *spatial)` layout `interpolate` expects.
    """

    _transformed_types = (torch.Tensor,)

    def __init__(
        self,
        size: int | Sequence[int],
        ndim: int | None = None,
        mode: str = "nearest",
    ):
        """Constructor.

        Args:
            size: Target per-axis size (e.g. `(64, 64, 64)`), or an int with
                `ndim`.
            ndim: Number of trailing spatial dims when `size` is an int.
            mode: Interpolation mode passed to `interpolate` (e.g. `nearest`,
                `trilinear`).
        """
        super().__init__()
        if isinstance(size, int):
            if ndim is None:
                raise ValueError("`ndim` is required when `size` is an int.")
            size = (size,) * ndim
        self.size = tuple(size)
        self.ndim = len(self.size)
        self.mode = mode

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Interpolate to `size`, restoring the input's rank afterwards."""
        x = inpt
        squeezed = 0
        while x.ndim < self.ndim + 2:
            x = x.unsqueeze(0)
            squeezed += 1
        if self.mode in _ALIGN_MODES:
            x = F.interpolate(x, size=self.size, mode=self.mode, align_corners=False)
        else:
            x = F.interpolate(x, size=self.size, mode=self.mode)
        for _ in range(squeezed):
            x = x.squeeze(0)
        return x


class PadND(Transform):
    """Pad the trailing N spatial dims via `torch.nn.functional.pad`."""

    _transformed_types = (torch.Tensor,)

    def __init__(
        self,
        padding: Sequence[Sequence[int]],
        mode: str = "constant",
        value: float = 0.0,
    ):
        """Constructor.

        Args:
            padding: One `(left, right)` pair per trailing spatial axis, in axis
                order (outermost first).
            mode: Padding mode (`constant`, `reflect`, `replicate`, `circular`).
            value: Fill value for `constant` padding.
        """
        super().__init__()
        self.padding = tuple(tuple(p) for p in padding)
        self.mode = mode
        self.value = value

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        """Apply the padding to the trailing spatial dims."""
        pad_arg = [p for lohi in reversed(self.padding) for p in lohi]
        if self.mode == "constant":
            return F.pad(inpt, pad_arg, mode="constant", value=self.value)
        return F.pad(inpt, pad_arg, mode=self.mode)
