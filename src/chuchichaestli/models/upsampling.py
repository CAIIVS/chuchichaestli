# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Upsampling modules for 1, 2, and 3D inputs."""

import torch
from torch import nn
from torch.nn import functional as F
from chuchichaestli.models.dwt.layers import InverseWaveletTransformND
from chuchichaestli.models.maps import DIM_TO_CONV_MAP, DIM_TO_CONVT_MAP, UPSAMPLE_MODE
from chuchichaestli.models.resample import ChannelResample
from chuchichaestli.models.shuffle import PixelShuffleND
from typing import Literal


__all__ = [
    "Upsample",
    "UpsampleWavelet",
    "UpsampleInterpolate",
    "UpsampleShuffle",
    "UPSAMPLE_FUNCTIONS",
    "UPSAMPLE_BLOCKS",
]

UpsampleTypes = Literal[
    "Upsample",
    "ChannelResample",
    "UpsampleWavelet",
    "UpsampleInterpolate",
    "UpsampleShuffle",
]


class Upsample(nn.Module):
    """Upsampling layer for 1D, 2D, and 3D inputs."""

    changes_channels = False

    def __init__(self, dimensions: int, num_channels: int):
        """Initialize the upsampling layer."""
        super().__init__()
        conv_cls = DIM_TO_CONVT_MAP[dimensions]
        self.factor = 2
        self.conv = conv_cls(
            num_channels, num_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor, _t: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the upsampling layer."""
        return self.conv(x)


class UpsampleInterpolate(nn.Module):
    """Upsampling layer for 1D, 2D, and 3D inputs implemented with interpolation."""

    changes_channels = False

    def __init__(
        self,
        dimensions: int,
        num_channels: int | None = None,
        factor: int | None = None,
        antialias: bool = False,
        with_conv: bool = True,
        **kwargs,
    ):
        """Initialize the upsampling layer."""
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
        return UPSAMPLE_MODE.get(self.dimensions, "nearest")

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the upsampling layer."""
        spatial_dims = x.shape[2:]
        output_dims = [s * self.factor for s in spatial_dims]
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


class UpsampleShuffle(nn.Module):
    """Upsampling layer for 1D, 2D, and 3D inputs implemented with pixel shuffling."""

    changes_channels = True

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        factor: int | None = None,
        **kwargs,
    ):
        """Initialize the upsampling layer."""
        super().__init__()
        conv_cls = DIM_TO_CONV_MAP[dimensions]
        self.dimensions = dimensions
        self.factor = factor if factor is not None else 2
        rd = self.factor**dimensions
        if out_channels * rd % in_channels:
            raise ValueError(
                f"Cannot shuffle {in_channels} into {out_channels} channels by a factor of"
                f" {self.factor} over {dimensions} dimension(s):"
                f" {rd} * out_channels must be divisible by in_channels."
            )
        self.repeats = out_channels * rd // in_channels
        kwargs.setdefault("kernel_size", 3)
        kwargs.setdefault("stride", 1)
        kwargs.setdefault("padding", "same")
        self.conv = conv_cls(in_channels, out_channels * rd, **kwargs)
        self.pixel_shuffle = PixelShuffleND(dimensions, self.factor)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the upsampling layer."""
        h = self.pixel_shuffle(self.conv(x))
        shortcut = x.repeat_interleave(
            self.repeats, dim=1, output_size=x.shape[1] * self.repeats
        )
        shortcut = self.pixel_shuffle(shortcut)
        return h + shortcut


class UpsampleWavelet(nn.Module):
    """Upsampling layer for 1D, 2D, and 3D inputs implemented with a wavelet transform.

    Reads the detail bands out of the channel axis and synthesizes them back
    into twice the spatial resolution. The counterpart of `DownsampleWavelet`
    and the wavelet analogue of `UpsampleShuffle`.
    """

    changes_channels = True

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        wavelet: str = "haar",
        mode: str = "periodization",
        factor: int | None = None,
        **kwargs,
    ):
        """Initialize the upsampling layer.

        Args:
            dimensions: Number of spatial dimensions.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            wavelet: Wavelet to synthesize with.
            mode: Signal extension mode; the default is the only critically
                sampled one, and hence the only one that doubles exactly.
            factor: Upscaling factor; only a factor of two is supported.
            kwargs: Additional keyword arguments for the convolution.

        Raises:
            ValueError: If the channel counts do not fit the transform, or if a
                factor other than two is requested.
        """
        super().__init__()
        conv_cls = DIM_TO_CONV_MAP[dimensions]
        self.dimensions = dimensions
        self.factor = factor if factor is not None else 2
        if self.factor != 2:
            raise ValueError(
                f"A wavelet transform doubles each axis, so only a factor of two is"
                f" supported; got {self.factor}."
            )
        rd = self.factor**dimensions
        if out_channels * rd % in_channels:
            raise ValueError(
                f"Cannot synthesize {in_channels} into {out_channels} channels over"
                f" {dimensions} dimension(s):"
                f" {rd} * out_channels must be divisible by in_channels."
            )
        self.repeats = out_channels * rd // in_channels
        kwargs.setdefault("kernel_size", 3)
        kwargs.setdefault("stride", 1)
        kwargs.setdefault("padding", "same")
        self.conv = conv_cls(in_channels, out_channels * rd, **kwargs)
        self.idwt = InverseWaveletTransformND(dimensions, wavelet, mode, "channel")

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the upsampling layer."""
        h = self.idwt(self.conv(x))
        shortcut = x.repeat_interleave(
            self.repeats, dim=1, output_size=x.shape[1] * self.repeats
        )
        return h + self.idwt(shortcut)


UPSAMPLE_FUNCTIONS = {
    "Upsample": Upsample,
    "ChannelResample": ChannelResample,
    "UpsampleWavelet": UpsampleWavelet,
    "UpsampleInterpolate": UpsampleInterpolate,
    "UpsampleShuffle": UpsampleShuffle,
}

UPSAMPLE_BLOCKS: tuple[type, ...] = tuple(UPSAMPLE_FUNCTIONS.values())
