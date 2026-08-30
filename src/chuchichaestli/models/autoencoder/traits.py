# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Traits the two components of an autoencoder implementation."""

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import torch
from torch import nn


__all__ = ["EncoderLike", "DecoderLike"]


@runtime_checkable
class EncoderLike(Protocol):
    """Interface an autoencoder expects from its encoding component.

    Any module exposing these members can be passed to `Autoencoder`; inheriting
    from `Encoder` is not required. Attributes absent from a custom component simply
    skip the consistency checks that read them.

    Attributes:
        dimensions: Number of spatial dimensions.
        in_channels: Number of input channels.
        n_channels: Number of channels after the input convolution.
        out_channels: Number of channels emitted.
        latent_channels: Number of latent channels `out_channels` stands for;
            smaller than `out_channels` when several channels describe one
            latent channel (a mean and a variance, say).
        bottleneck_channels: Number of channels entering the output block.
        channel_mults: Total channel multiplication across all levels.
        levels: Number of spatially hierarchical levels.
    """

    dimensions: int
    in_channels: int
    n_channels: int
    out_channels: int
    latent_channels: int
    bottleneck_channels: int
    channel_mults: int
    levels: int

    @property
    def f(self) -> int:
        """Spatial compression factor."""
        ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an input tensor.

        Args:
            x: Input tensor.
        """
        ...

    def parameters(self) -> Iterator[nn.Parameter]:
        """Iterate over the component's parameters."""
        ...

    def buffers(self) -> Iterator[torch.Tensor]:
        """Iterate over the component's buffers."""
        ...


@runtime_checkable
class DecoderLike(Protocol):
    """Interface an autoencoder expects from its decoding component.

    Any module exposing these members can be passed to `Autoencoder`; inheriting
    from `Decoder` is not required. Attributes absent from a custom component simply
    skip the consistency checks that read them.

    Attributes:
        dimensions: Number of spatial dimensions.
        in_channels: Number of latent channels consumed.
        n_channels: Number of channels after the input block.
        out_channels: Number of output channels.
        channel_mults: Total channel division across all levels.
        levels: Number of spatially hierarchical levels.
    """

    dimensions: int
    in_channels: int
    n_channels: int
    out_channels: int
    channel_mults: int
    levels: int

    @property
    def f(self) -> int:
        """Spatial expansion factor."""
        ...

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent tensor.

        Args:
            z: Input latent tensor.
        """
        ...

    def parameters(self) -> Iterator[nn.Parameter]:
        """Iterate over the component's parameters."""
        ...

    def buffers(self) -> Iterator[torch.Tensor]:
        """Iterate over the component's buffers."""
        ...
