# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Blurring modules for 1, 2, and 3D inputs."""

import torch
from torch import nn

from chuchichaestli.dwt.modes import ExtensionModeTypes, pad_signal
from chuchichaestli.models.maps import DIM_TO_CONV_FN_MAP


__all__ = ["GaussianBlurND", "gaussian_kernel_1d"]


def gaussian_kernel_1d(kernel_size: int, sigma: float) -> torch.Tensor:
    """Build a normalized one-dimensional Gaussian kernel.

    Args:
        kernel_size: Number of taps; an even count is centred between samples.
        sigma: Standard deviation of the Gaussian, in samples.

    Raises:
        ValueError: If `kernel_size` or `sigma` is not positive.
    """
    if kernel_size < 1:
        raise ValueError(f"A kernel needs at least one tap; got {kernel_size}.")
    if sigma <= 0:
        raise ValueError(f"The standard deviation must be positive; got {sigma}.")
    offsets = torch.arange(kernel_size, dtype=torch.float64)
    offsets = offsets - (kernel_size - 1) / 2
    kernel = torch.exp(-offsets.pow(2) / (2 * sigma**2))
    return kernel / kernel.sum()


class GaussianBlurND(nn.Module):
    """Separable Gaussian blur for 1D, 2D, and 3D inputs.

    The blur runs as one depthwise convolution per spatial axis, which costs
    `dimensions * kernel_size` taps per output instead of `kernel_size**dimensions`.
    """

    def __init__(
        self,
        dimensions: int,
        kernel_size: int = 5,
        sigma: float = 1.0,
        mode: ExtensionModeTypes = "reflect",
    ):
        """Constructor.

        Args:
            dimensions: Number of spatial dimensions.
            kernel_size: Number of taps along each axis.
            sigma: Standard deviation of the Gaussian, in samples.
            mode: Signal extension mode used at the boundaries.
        """
        super().__init__()
        self.dimensions = dimensions
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.mode = mode
        self.register_buffer(
            "kernel", gaussian_kernel_1d(kernel_size, sigma), persistent=False
        )

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the blur."""
        channels = x.shape[1]
        pad_lo = (self.kernel_size - 1) // 2
        pad_hi = self.kernel_size - 1 - pad_lo
        kernel = self.kernel.to(x.dtype)
        for axis in range(self.dimensions):
            shape = [1] * self.dimensions
            shape[axis] = self.kernel_size
            weight = kernel.reshape(1, 1, *shape).repeat(
                channels, 1, *([1] * self.dimensions)
            )
            x = pad_signal(x, 2 + axis, pad_lo, pad_hi, self.mode)
            x = DIM_TO_CONV_FN_MAP[self.dimensions](x, weight, groups=channels)
        return x

    def extra_repr(self) -> str:
        """Report the kernel size, standard deviation and extension mode."""
        return (
            f"dimensions={self.dimensions}, kernel_size={self.kernel_size},"
            f" sigma={self.sigma}, mode={self.mode!r}"
        )
