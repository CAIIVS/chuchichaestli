# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pixel shuffling modules for 1, 2, and 3D inputs (unlike `torch.nn.PixelShuffle` and `torch.nn.PixelUnshuffle`)."""

import torch
from torch import nn


__all__ = ["PixelShuffleND", "PixelUnshuffleND"]


def _check_rank(x: torch.Tensor, dimensions: int) -> None:
    """Check that a tensor carries a batch and channel axis plus the spatial ones.

    Args:
        x: Input tensor.
        dimensions: Number of spatial dimensions.

    Raises:
        ValueError: If the tensor rank does not match `dimensions`.
    """
    if x.dim() != dimensions + 2:
        raise ValueError(
            f"Expected a {dimensions + 2}-dimensional input for {dimensions} spatial"
            f" dimension(s); got an input of shape {tuple(x.shape)}."
        )


class PixelUnshuffleND(nn.Module):
    """Move a factor of every spatial axis into the channel axis.

    Rearranges `(N, C, s₁·r, …, sₙ·r)` into `(N, C·rⁿ, s₁, …, sₙ)`.
    """

    def __init__(self, dimensions: int, factor: int = 2):
        """Initialize the unshuffling layer.

        Args:
            dimensions: Number of spatial dimensions.
            factor: Downscaling factor, applied to every spatial axis.
        """
        super().__init__()
        self.dimensions = dimensions
        self.factor = factor

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the unshuffling layer.

        Raises:
            ValueError: If the input rank or its spatial sizes do not fit the layer.
        """
        d, r = self.dimensions, self.factor
        _check_rank(x, d)
        if any(s % r for s in x.shape[2:]):
            raise ValueError(
                f"Every spatial axis must be divisible by the factor {r};"
                f" got an input of shape {tuple(x.shape)}."
            )
        n, c = x.shape[:2]
        spatial = [s // r for s in x.shape[2:]]
        x = x.reshape(n, c, *[axis for s in spatial for axis in (s, r)])
        factor_axes = [2 + 2 * i + 1 for i in range(d)]
        spatial_axes = [2 + 2 * i for i in range(d)]
        x = x.permute(0, 1, *factor_axes, *spatial_axes)
        return x.reshape(n, c * r**d, *spatial)


class PixelShuffleND(nn.Module):
    """Move a factor out of the channel axis into every spatial axis.

    Rearranges `(N, C·rⁿ, s₁, …, sₙ)` into `(N, C, s₁·r, …, sₙ·r)`; the
    inverse of `PixelUnshuffleND`.
    """

    def __init__(self, dimensions: int, factor: int = 2):
        """Initialize the shuffling layer.

        Args:
            dimensions: Number of spatial dimensions.
            factor: Upscaling factor, applied to every spatial axis.
        """
        super().__init__()
        self.dimensions = dimensions
        self.factor = factor

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the shuffling layer.

        Raises:
            ValueError: If the input rank or its channel count do not fit the layer.
        """
        d, r = self.dimensions, self.factor
        _check_rank(x, d)
        if x.shape[1] % r**d:
            raise ValueError(
                f"The channel dimension must be divisible by {r}**{d} = {r**d};"
                f" got an input of shape {tuple(x.shape)}."
            )
        n = x.shape[0]
        channels = x.shape[1] // r**d
        spatial = list(x.shape[2:])
        x = x.reshape(n, channels, *([r] * d), *spatial)
        interleaved = [axis for i in range(d) for axis in (2 + d + i, 2 + i)]
        x = x.permute(0, 1, *interleaved)
        return x.reshape(n, channels, *[s * r for s in spatial])
