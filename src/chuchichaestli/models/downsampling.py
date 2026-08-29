# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Downsampling modules for 1, 2, and 3D inputs."""

import torch
from torch import nn
from torch.nn import functional as F
from chuchichaestli.models.maps import DIM_TO_CONV_MAP, DIM_TO_POOL_MAP, DOWNSAMPLE_MODE
from chuchichaestli.models.shuffle import PixelUnshuffleND
from chuchichaestli.utils import partialclass
from collections.abc import Sequence
from typing import Literal


__all__ = [
    "Downsample",
    "DownsampleInterpolate",
    "DownsampleUnshuffle",
    "Pool",
    "MaxPool",
    "AdaptiveMaxPool",
    "AvgPool",
    "AdaptiveAvgPool",
    "DOWNSAMPLE_FUNCTIONS",
    "DOWNSAMPLE_BLOCKS",
]


DownsampleTypes = Literal[
    "Downsample",
    "DownsampleInterpolate",
    "DownsampleUnshuffle",
    "MaxPool",
    "AdaptiveMaxPool",
    "AvgPool",
    "AdaptiveAvgPool",
]


def _stride_factor(stride: int | Sequence[int]) -> int:
    """Reduce a stride to the factor by which it scales every spatial axis.

    Args:
        stride: Stride, either shared by every axis or one entry per axis.

    Raises:
        ValueError: If the axes are not scaled by the same factor.
    """
    if isinstance(stride, int):
        return stride
    factors = set(stride)
    if len(factors) != 1:
        raise ValueError(
            f"Sampling blocks scale every spatial axis by the same factor;"
            f" got stride={tuple(stride)}."
        )
    return factors.pop()


class Downsample(nn.Module):
    """Downsampling layer for 1D, 2D, and 3D inputs."""

    changes_channels = False

    def __init__(self, dimensions: int, num_channels: int, **kwargs):
        """Initialize the downsampling layer."""
        super().__init__()
        conv_cls = DIM_TO_CONV_MAP[dimensions]
        kwargs.setdefault("kernel_size", 3)
        kwargs.setdefault("stride", 2)
        kwargs.setdefault("padding", 1)
        self.factor = _stride_factor(kwargs["stride"])
        self.conv = conv_cls(num_channels, num_channels, **kwargs)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the downsampling layer."""
        return self.conv(x)


class DownsampleInterpolate(nn.Module):
    """Downsampling layer for 1D, 2D, and 3D inputs implemented with interpolation.

    Note: In the U-Net architecture, downsampling by interpolation is not commonly used.
    """

    changes_channels = False

    def __init__(
        self,
        dimensions: int,
        num_channels: int | None = None,
        factor: int | None = None,
        antialias: bool = False,
        with_conv: bool = False,
        **kwargs,
    ):
        """Initialize the downsampling layer."""
        super().__init__()
        conv_cls = DIM_TO_CONV_MAP[dimensions]
        self.dimensions = dimensions
        self.num_channels = num_channels
        self.factor = factor if factor is not None else 2
        self.align_corners = False
        self.antialias = antialias
        kwargs.setdefault("kernel_size", 3)
        kwargs.setdefault("stride", 1)
        kwargs.setdefault("padding", "same")
        if with_conv:
            self.conv = conv_cls(num_channels, num_channels, **kwargs)

    @property
    def mode(self) -> Literal["linear", "bilinear", "trilinear", "nearest"]:
        """Interpolation mode."""
        return DOWNSAMPLE_MODE.get(self.dimensions, "nearest")

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the downsampling layer."""
        spatial_dims = x.shape[2:]
        output_dims = [s // self.factor for s in spatial_dims]
        x = F.interpolate(
            x,
            size=output_dims,
            mode=self.mode,
            align_corners=self.align_corners,
            antialias=self.antialias,
        )
        if hasattr(self, "conv"):
            x = self.conv(x)
        return x


class DownsampleUnshuffle(nn.Module):
    """Downsampling layer for 1D, 2D, and 3D inputs implemented with pixel shuffling."""

    changes_channels = True

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        factor: int | None = None,
        **kwargs,
    ):
        """Initialize the downsampling layer."""
        super().__init__()
        conv_cls = DIM_TO_CONV_MAP[dimensions]
        self.dimensions = dimensions
        self.factor = factor if factor is not None else 2
        rd = self.factor**dimensions
        if out_channels % rd or in_channels * rd % out_channels:
            raise ValueError(
                f"Cannot unshuffle {in_channels} into {out_channels} channels by a factor"
                f" of {self.factor} over {dimensions} dimension(s): out_channels must be"
                f" divisible by {rd}, and {rd} * in_channels by out_channels."
            )
        self.group_size = in_channels * rd // out_channels
        kwargs.setdefault("kernel_size", 3)
        kwargs.setdefault("stride", 1)
        kwargs.setdefault("padding", "same")
        self.conv = conv_cls(in_channels, out_channels // rd, **kwargs)
        self.pixel_unshuffle = PixelUnshuffleND(dimensions, self.factor)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the downsampling layer."""
        h = self.pixel_unshuffle(self.conv(x))
        shortcut = self.pixel_unshuffle(x)
        shortcut = shortcut.unflatten(1, (-1, self.group_size)).mean(dim=2)
        return h + shortcut


ADAPTIVE_POOL_FUNCTIONS = {
    (1, False): F.adaptive_max_pool1d,
    (2, False): F.adaptive_max_pool2d,
    (3, False): F.adaptive_max_pool3d,
    (1, True): F.adaptive_avg_pool1d,
    (2, True): F.adaptive_avg_pool2d,
    (3, True): F.adaptive_avg_pool3d,
}


class Pool(nn.Module):
    """Max/avg (optionally adaptive) pooling layer for 1D, 2D, and 3D inputs."""

    changes_channels = False

    def __init__(
        self,
        dimensions: int,
        num_channels: int | None = None,
        average: bool = False,
        adaptive: bool = False,
        **kwargs,
    ):
        """Initialize the pooling layer (default: max pooling).

        An adaptive layer without an explicit `output_size` pools down by `stride`,
        computing the output size from each input; give `output_size` to pool to a
        fixed size instead.
        """
        super().__init__()
        self.dimensions = dimensions
        self.average = average
        self.adaptive = adaptive
        self.output_size = kwargs.pop("output_size", None)
        if adaptive and (ignored := {"kernel_size", "padding"} & kwargs.keys()):
            raise ValueError(
                f"Adaptive pooling has no {', '.join(sorted(ignored))};"
                f" size it with output_size or stride instead."
            )
        kwargs.setdefault("kernel_size", 3)
        kwargs.setdefault("stride", 2)
        kwargs.setdefault("padding", 1)
        self.factor = (
            None if self.output_size is not None else _stride_factor(kwargs["stride"])
        )
        if adaptive:
            self.pool = None
            return
        pool_cls = DIM_TO_POOL_MAP[dimensions]["AvgPool" if average else "MaxPool"]
        self.pool = pool_cls(**kwargs)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the pooling layer."""
        if not self.adaptive:
            return self.pool(x)
        size = self.output_size or [s // self.factor for s in x.shape[2:]]
        return ADAPTIVE_POOL_FUNCTIONS[(self.dimensions, self.average)](x, size)


MaxPool = partialclass("MaxPool", Pool, average=False, adaptive=False)
AdaptiveMaxPool = partialclass("AdaptiveMaxPool", Pool, adaptive=True)
AvgPool = partialclass("AvgPool", Pool, average=True)
AdaptiveAvgPool = partialclass("AdaptiveAvgPool", Pool, average=True, adaptive=True)


DOWNSAMPLE_FUNCTIONS = {
    "Downsample": Downsample,
    "DownsampleInterpolate": DownsampleInterpolate,
    "DownsampleUnshuffle": DownsampleUnshuffle,
    "MaxPool": MaxPool,
    "AdaptiveMaxPool": AdaptiveMaxPool,
    "AvgPool": AvgPool,
    "AdaptiveAvgPool": AdaptiveAvgPool,
}

DOWNSAMPLE_BLOCKS: tuple[type, ...] = tuple(DOWNSAMPLE_FUNCTIONS.values())
