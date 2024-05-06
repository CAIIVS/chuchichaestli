"""Upsampling layers for 1D and 2D data."""

import torch
from torch import nn
import torch.nn.functional as F

from functools import partial

from chuchichaestli.models.normalization import RMSNorm


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 1D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: int = None,
        name: str = "conv",
    ):
        """Initialize the upsampling layer."""
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the upsampling layer."""
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)

        outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")

        if self.use_conv:
            outputs = self.conv(outputs)

        return outputs


class Upsample(nn.Module):
    """A 2D and 3D upsampling layer with an optional convolution and normalization.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 2D layer.
        kernel_size (`int`, optional):
            kernel size for the convolution. Defaults to `None`.
        padding (`int`, optional):
            padding for the convolution. Defaults to `1`.
        norm_type (`str`, optional):
            type of normalization to use. Defaults to `None`.
        eps (`float`, optional):
            epsilon value for normalization. Defaults to `None`.
        bias (`bool`, optional):
            option to use bias in the convolution. Defaults to `True`.
        interpolate (`bool`, optional):
            option to interpolate the input. Defaults to `True`.
        dimension (`int`, optional):
            dimension of the convolution. Defaults to `2`.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: int = None,
        name: str = "conv",
        kernel_size: int = None,
        padding=1,
        norm_type=None,
        eps=None,
        bias=True,
        interpolate=True,
        dimension=2,
    ):
        """Initialize the upsampling layer."""
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate

        if dimension == 2:
            conv_transpose_cls = nn.ConvTranspose2d
            conv_cls = nn.Conv2d
            self.in_permutation = (0, 2, 3, 1)
            self.out_permutation = (0, 3, 1, 2)
        elif dimension == 3:
            conv_transpose_cls = nn.ConvTranspose3d
            conv_cls = nn.Conv3d
            self.in_permutation = (0, 2, 3, 4, 1)
            self.out_permutation = (0, 4, 1, 2, 3)

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            conv = conv_transpose_cls(
                channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                bias=bias,
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = conv_cls(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            )
        self.conv = conv

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_size: int = None,
    ) -> torch.FloatTensor:
        """Forward pass of the upsampling layer."""
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(*self.permutation)).permute(
                *self.out_permutation
            )

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if self.interpolate:
            if output_size is None:
                hidden_states = F.interpolate(
                    hidden_states, scale_factor=2.0, mode="nearest"
                )
            else:
                hidden_states = F.interpolate(
                    hidden_states, size=output_size, mode="nearest"
                )

        if self.use_conv:
            hidden_states = self.conv(hidden_states)

        return hidden_states


Upsample2D = partial(Upsample, dimension=2)
Upsample3D = partial(Upsample, dimension=3)
