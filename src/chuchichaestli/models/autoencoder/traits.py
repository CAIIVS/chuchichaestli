# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Traits the two components of an autoencoder implement."""

from typing import Protocol, runtime_checkable

import torch


__all__ = ["EncoderLike", "DecoderLike"]


@runtime_checkable
class EncoderLike(Protocol):
    """Interface an autoencoder expects from its encoding component.

    Any module exposing these members can be passed to `Autoencoder`; inheriting
    from `Encoder` is not required. Attributes absent from a custom component simply
    skip the consistency checks that read them.

    Rendering a model with `chuchichaestli.utils.visualization` additionally
    requires a `down_blocks` sequence of level stages and downsampling blocks.
    Submodules are excluded from the trait itself because `isinstance` resolves
    protocol members with `inspect.getattr_static`, which does not see the
    children an `nn.Module` exposes through `__getattr__`.

    Attributes:
        dimensions: Number of spatial dimensions.
        in_channels: Number of input channels.
        n_channels: Number of channels after the input convolution.
        out_channels: Number of latent channels produced (doubled if `double_z`).
        bottleneck_channels: Number of channels entering the output block.
        double_z: Whether the latent channels hold both mean and variance.
        channel_mults: Total channel multiplication across all levels.
        levels: Number of spatially hierarchical levels.
    """

    dimensions: int
    in_channels: int
    n_channels: int
    out_channels: int
    bottleneck_channels: int
    double_z: bool
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


@runtime_checkable
class DecoderLike(Protocol):
    """Interface an autoencoder expects from its decoding component.

    Any module exposing these members can be passed to `Autoencoder`; inheriting
    from `Decoder` is not required. Attributes absent from a custom component simply
    skip the consistency checks that read them.

    Rendering a model with `chuchichaestli.utils.visualization` additionally
    requires an `up_blocks` sequence of level stages and upsampling blocks (see
    `EncoderLike` for why submodules are not trait members).

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
