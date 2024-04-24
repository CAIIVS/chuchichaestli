"""Downsampling layers for 1D and 2D inputs."""

import torch
from torch import nn
import torch.nn.functional as F

from chuchichaestli.models.normalization import RMSNorm


class Downsample1D(nn.Module):
    """A 1D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 1D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: int = None,
        padding: int = 1,
        name: str = "conv",
    ):
        """Initialize the downsampling layer."""
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            self.conv = nn.Conv1d(
                self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.conv = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the downsampling layer."""
        assert inputs.shape[1] == self.channels
        return self.conv(inputs)


class Downsample2D(nn.Module):
    """A 2D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: int = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
    ):
        """Initialize the downsampling layer."""
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        if use_conv:
            conv = nn.Conv2d(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass of the downsampling layer."""
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(
                0, 3, 1, 2
            )

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states
