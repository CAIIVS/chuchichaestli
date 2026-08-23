# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""N-dimensional geometric augmentations (exact, interpolation-free)."""

from typing import Any
from collections.abc import Sequence
import torch
from torchvision.transforms.v2 import Transform

__all__ = ["RandomFlipND", "RandomRot90ND"]


def _neg_axes(axes: Sequence[int] | None, ndim: int) -> tuple[int, ...]:
    """Return the target spatial axes as negative (trailing) indices.

    Args:
        axes: Spatial axes as `0..ndim-1` (or negatives); `None` selects all.
        ndim: Number of trailing spatial dims.
    """
    if axes is None:
        return tuple(range(-ndim, 0))
    return tuple(a if a < 0 else a - ndim for a in axes)


class RandomFlipND(Transform):
    """Randomly flip each eligible trailing spatial axis with probability `p`.

    The per-axis draws happen in `make_params`, so a `SequenceCollate` flips
    every paired field/step identically.
    """

    _transformed_types = (torch.Tensor,)

    def __init__(
        self, ndim: int, p: float = 0.5, axes: Sequence[int] | None = None
    ):
        """Constructor.

        Args:
            ndim: Number of trailing spatial dims.
            p: Per-axis flip probability.
            axes: Spatial axes eligible for flipping (default: all `ndim`).
        """
        super().__init__()
        self.ndim = ndim
        self.p = p
        self.axes = _neg_axes(axes, ndim)

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """Sample which axes to flip."""
        dims = tuple(a for a in self.axes if float(torch.rand(())) < self.p)
        return {"dims": dims}

    def transform(self, x: Any, params: dict[str, Any]) -> Any:
        """Flip the chosen axes (identity when none were sampled)."""
        return torch.flip(x, dims=params["dims"]) if params["dims"] else x


class RandomRot90ND(Transform):
    """Random k*90 degree rotation in a random trailing spatial axis-plane.

    Exact (no interpolation). `k` and the plane are drawn in `make_params`, so a
    `SequenceCollate` rotates every paired field/step identically.
    """

    _transformed_types = (torch.Tensor,)

    def __init__(self, ndim: int, axes: Sequence[int] | None = None):
        """Constructor.

        Args:
            ndim: Number of trailing spatial dims (must be >= 2).
            axes: Spatial axes eligible to form rotation planes (default: all).
        """
        super().__init__()
        if ndim < 2:
            raise ValueError("RandomRot90ND requires ndim >= 2.")
        self.ndim = ndim
        spatial = _neg_axes(axes, ndim)
        self.planes = [
            (spatial[i], spatial[j])
            for i in range(len(spatial))
            for j in range(i + 1, len(spatial))
        ]

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """Sample the rotation count `k` and the axis-plane."""
        k = int(torch.randint(0, 4, size=()))
        plane = self.planes[int(torch.randint(0, len(self.planes), size=()))]
        return {"k": k, "plane": plane}

    def transform(self, x: Any, params: dict[str, Any]) -> Any:
        """Rotate by `k*90` degrees in the chosen plane (identity when k%4==0)."""
        if params["k"] % 4 == 0:
            return x
        return torch.rot90(x, params["k"], dims=params["plane"])
