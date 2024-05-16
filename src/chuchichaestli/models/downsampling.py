"""Downsampling layers for 1D and 2D inputs.

Copyright 2024 The HuggingFace Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Additional modifications made by the Intelligent Vision Systems Group at ZHAW under the
GNU General Public License v3.0 which extends the conditions of the License for further
redistribution and use. See the GPLv3 license at

    http://www.gnu.org/licenses/gpl-3.0.html

This file is part of Chuchichaestli and has been modified for use in this project.
"""

import torch
from torch import nn
import torch.nn.functional as F

from functools import partial

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


class Downsample(nn.Module):
    """A 2D and 3D downsampling layer with an optional convolution and normalization.

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
        kernel_size: int = 3,
        norm_type: str = None,
        eps: float = None,
        elementwise_affine: bool = True,
        bias: bool = True,
        dimension: int = 2,
    ):
        """Initialize the downsampling layer."""
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if dimension == 2:
            conv_cls = nn.Conv2d
            avg_pool_cls = nn.AvgPool2d
            self.in_permutation = (0, 2, 3, 1)
            self.out_permutation = (0, 3, 1, 2)
        elif dimension == 3:
            conv_cls = nn.Conv3d
            avg_pool_cls = nn.AvgPool3d
            self.in_permutation = (0, 2, 3, 4, 1)
            self.out_permutation = (0, 4, 1, 2, 3)

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        if use_conv:
            conv = conv_cls(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        else:
            assert self.channels == self.out_channels
            conv = avg_pool_cls(kernel_size=stride, stride=stride)

        self.conv = conv

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass of the downsampling layer."""
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(
                hidden_states.permute(*self.in_permutation)
            ).permute(*self.out_permutation)

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states


Downsample2D = partial(Downsample, dimension=2)
Downsample3D = partial(Downsample, dimension=3)
