# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""A highly-customizable autoencoder implementation."""

import torch
from torch import nn
from chuchichaestli.models.activations import ActivationTypes
from chuchichaestli.models.autoencoder.decoder import Decoder
from chuchichaestli.models.autoencoder.encoder import Encoder
from chuchichaestli.models.blocks import (
    AutoencoderDownBlockTypes,
    AutoencoderMidBlockTypes,
    AutoencoderUpBlockTypes,
    EncoderOutBlockTypes,
    DecoderInBlockTypes,
)
from chuchichaestli.models.downsampling import DownsampleTypes
from chuchichaestli.models.maps import DIM_TO_CONV_MAP
from chuchichaestli.models.norm import NormTypes
from chuchichaestli.models.upsampling import UpsampleTypes
from chuchichaestli.utils import prod
from collections.abc import Sequence


__all__ = ["Autoencoder"]


class Autoencoder(nn.Module):
    """Flexible autoencoder implementation.

    The architecture consists of an encoder-decoder structure.
    The encoder chains several residual and downsampling blocks.
    Each downsampling block separates the encoder into spatially hierarchical levels.
    The encoder ends in bottleneck blocks (optionally including attention blocks
    and a convolutional layer) and projects the input into latent space.
    The decoder is built with residual convolutional and upsampling blocks, and
    expands from the latent space to the image domain.
    """

    def __init__(
        self,
        dimensions: int = 2,
        in_channels: int = 1,
        n_channels: int = 64,
        latent_dim: int = 4,
        out_channels: int = 1,
        down_block_types: Sequence[AutoencoderDownBlockTypes] = (
            "AutoencoderDownBlock",
            "AutoencoderDownBlock",
            "AutoencoderDownBlock",
            "AutoencoderDownBlock",
        ),
        down_layers_per_block: int | Sequence[int] = 2,
        downsample_type: DownsampleTypes = "Downsample",
        encoder_mid_block_types: Sequence[AutoencoderMidBlockTypes] = (
            "AutoencoderMidBlock",
            "AttnAutoencoderMidBlock",
        ),
        encoder_out_block_type: EncoderOutBlockTypes = "EncoderOutBlock",
        decoder_in_block_type: DecoderInBlockTypes = "DecoderInBlock",
        decoder_mid_block_types: Sequence[AutoencoderMidBlockTypes] = (
            "AutoencoderMidBlock",
            "AttnAutoencoderMidBlock",
        ),
        up_block_types: AutoencoderUpBlockTypes = (
            "AutoencoderUpBlock",
            "AutoencoderUpBlock",
            "AutoencoderUpBlock",
            "AutoencoderUpBlock",
        ),
        up_layers_per_block: int | Sequence[int] = 3,
        upsample_type: UpsampleTypes = "UpsampleInterpolate",
        block_out_channel_mults: Sequence[int] = (1, 2, 2, 2),
        decoder_block_out_channel_mults: Sequence[int] | None = None,
        use_latent_proj: bool = True,
        use_latent_deproj: bool = True,
        res_act_fn: ActivationTypes = "silu",
        res_dropout: float = 0.0,
        res_norm_type: NormTypes = "group",
        res_groups: int = 8,
        res_kernel_size: int = 3,
        attn_head_dim: int = 32,
        attn_n_heads: int = 1,
        attn_dropout_p: float = 0.0,
        attn_norm_type: NormTypes = "group",
        attn_groups: int = 32,
        attn_kernel_size: int = 1,
        attn_scales: Sequence[int] = (5,),
        context_args: dict = {},
        local_args: dict = {},
        encoder_act_fn: ActivationTypes = "silu",
        encoder_norm_type: NormTypes = "group",
        encoder_groups: int = 8,
        encoder_kernel_size: int = 3,
        encoder_out_shortcut: bool = False,
        decoder_act_fn: ActivationTypes = "silu",
        decoder_norm_type: NormTypes = "group",
        decoder_groups: int = 8,
        decoder_kernel_size: int = 3,
        decoder_in_shortcut: bool = False,
        double_z: bool = False,
    ):
        """Initializes the VAE model with the given parameters.

        Args:
            dimensions: Number of dimensions for the model.
            in_channels: Number of input channels.
            n_channels: Number of channels in the hidden layer.
            latent_dim: Number of channels in the latent space.
            out_channels: Number of output channels.
            down_block_types: Types of down block(s) to use for each level.
            down_layers_per_block: Number of blocks per level in the encoder
                (blocks are repeated if `>1`).
            downsample_type: Type of downsampling block
                (see `chuchichaestli.models.downsampling` for details).
            encoder_mid_block_types: Types of middle block(s) in the encoder.
            encoder_out_block_type: Type of output block in the encoder.
            decoder_in_block_type: Type of input block in the decoder
            decoder_mid_block_types: Types of middle block(s) in the decoder.
            up_block_types: Type of up block(s) to use for each level.
            up_layers_per_block: Number of blocks per level in the decoder
                (blocks are repeated if `>1`).
            upsample_type: Type of upsampling block
                (see `chuchichaestli.models.upsampling` for details).
            block_out_channel_mults: Multiplier for output channels of each level block.
            decoder_block_out_channel_mults: Multiplier for output channels of each decoder level.
            use_latent_proj: Whether to use a linear layer between encoder and latent space.
            use_latent_deproj: Whether to use a linear layer between latent space and decoder.
            res_act_fn: Activation function for the residual blocks
                (see `chuchichaestli.models.activations` for details).
            res_dropout: Dropout rate for the residual blocks.
            res_norm_type: Normalization type for the residual block
                (see `chuchichaestli.models.norm` for details).
            res_groups: Number of groups for the residual block normalization (if group norm).
            res_kernel_size: Kernel size for the residual blocks.
            attn_head_dim: Dimension of the attention heads.
            attn_n_heads: Number of attention heads.
            attn_dropout_p: Dropout probability of the scaled dot product attention.
            attn_norm_type: Normalization type for the convolutional attention block
                (see `chuchichaestli.models.norm` for details).
            attn_groups: Number of groups for the convolutional attention block normalization
                (if `attn_norm_type` is `"group"`).
            attn_kernel_size: Kernel size for the convolutional attention block.
            attn_scales: Scales for the multi-scale attention block.
            context_args: Keyword arguments for the context block in a transformer module.
            local_args: Keyword arguments for the local block in a transformer module.
            encoder_act_fn: Activation function for the output layers in the encoder
                (see `chuchichaestli.models.activations` for details).
            encoder_norm_type: Normalization type for the encoder's output block
                (see `chuchichaestli.models.norm` for details).
            encoder_groups: Number of groups for normalization in the output layer of the encoder.
            encoder_kernel_size: Kernel size for the output convolution in the encoder.
            encoder_out_shortcut: Whether to use an encoder shortcut.
            decoder_act_fn: Activation function for the input/output layers in the decoder
                (see `chuchichaestli.models.activations` for details).
            decoder_norm_type: Normalization type for the decoder's output block
                (see `chuchichaestli.models.norm` for details).
            decoder_groups: Number of groups for normalization in the input/output layer of the decoder.
            decoder_kernel_size: Kernel size for the output convolution in the decoder.
            decoder_in_shortcut: Whether to use a decoder shortcut.
            double_z: Whether to double the latent space.
        """
        super().__init__()

        if encoder_out_block_type == "DCEncoderOutBlock":
            assert (
                dimensions == 2
            ), "Deep-compression autoencoding is only supported for 2D data."

        self.double_z = double_z
        self.channel_mults = prod(block_out_channel_mults)
        if decoder_block_out_channel_mults is None:
            decoder_block_out_channel_mults = block_out_channel_mults

        res_args = {
            "res_act_fn": res_act_fn,
            "res_dropout": res_dropout,
            "res_groups": res_groups,
            "res_norm_type": res_norm_type,
            "res_kernel_size": res_kernel_size,
        }

        attn_args = {
            "n_heads": attn_n_heads,
            "head_dim": attn_head_dim,
            "dropout_p": attn_dropout_p,
            "norm_type": attn_norm_type,
            "groups": attn_groups,
            "num_groups": attn_groups,
            "scales": attn_scales,
            "context_args": context_args,
            "local_args": local_args,
        }

        self.encoder = Encoder(
            dimensions=dimensions,
            in_channels=in_channels,
            n_channels=n_channels,
            out_channels=latent_dim,
            down_block_types=down_block_types,
            block_out_channel_mults=block_out_channel_mults,
            num_layers_per_block=down_layers_per_block,
            mid_block_types=encoder_mid_block_types,
            out_block_type=encoder_out_block_type,
            downsample_type=downsample_type,
            act_fn=encoder_act_fn,
            norm_type=encoder_norm_type,
            num_groups=encoder_groups,
            kernel_size=encoder_kernel_size,
            res_args=res_args,
            attn_args=attn_args,
            double_z=double_z,
            out_shortcut=encoder_out_shortcut,
        )
        self.decoder = Decoder(
            dimensions=dimensions,
            in_channels=latent_dim,
            n_channels=self.channel_mults * n_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            in_block_type=decoder_in_block_type,
            block_out_channel_mults=decoder_block_out_channel_mults,
            num_layers_per_block=up_layers_per_block,
            mid_block_types=decoder_mid_block_types,
            upsample_type=upsample_type,
            act_fn=decoder_act_fn,
            norm_type=decoder_norm_type,
            num_groups=decoder_groups,
            kernel_size=decoder_kernel_size,
            res_args=res_args,
            attn_args=attn_args,
            in_shortcut=decoder_in_shortcut,
        )
        self.latent_proj = (
            DIM_TO_CONV_MAP[dimensions](
                self.latent_dim * (2 if self.double_z else 1),
                self.latent_dim * (2 if self.double_z else 1),
                kernel_size=1,
                stride=1,
                padding="same",
            )
            if use_latent_proj
            else None
        )
        self.latent_deproj = (
            DIM_TO_CONV_MAP[dimensions](
                self.latent_dim,
                self.latent_dim,
                kernel_size=1,
                stride=1,
                padding="same",
            )
            if use_latent_deproj
            else None
        )

    @property
    def latent_dim(self) -> int:
        """Latent channel dimension."""
        if self.double_z:
            return self.encoder.out_channels // 2
        else:
            return self.encoder.out_channels

    @property
    def levels(self) -> tuple[int, int]:
        """Number of stages in the encoder and decoder."""
        return self.encoder.levels, self.decoder.levels

    @property
    def f_comp(self) -> int:
        """Spatial compression factor of the encoder (number of spatial downsampling layers)."""
        return self.encoder.f

    @property
    def f_exp(self) -> int:
        """Spatial expansion factor of the decoder (number of spatial upsampling layers)."""
        return self.decoder.f

    def compute_latent_shape(
        self, input_shape: tuple[int, ...], no_batch_dim: bool = False
    ):
        """Compute the shape of the latent space."""
        batch_dim = input_shape[0] if not no_batch_dim else None
        spatial_dims = tuple(dim // self.f_comp for dim in input_shape[2:])
        if batch_dim is None:
            shape = (self.latent_dim, *spatial_dims)
        else:
            shape = (batch_dim, self.latent_dim, *spatial_dims)
        return shape

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input.

        Args:
            x: Input tensor.
            eps: Small constant value for numerical stability.

        Returns:
            Multivariate normal posterior distribution
        """
        z = self.encoder(x)
        z = self.latent_proj(z) if self.latent_proj is not None else z
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the input.

        Args:
            z: Input latent tensor.

        Returns:
            Image reconstructed from latent code.
        """
        z = self.latent_deproj(z) if self.latent_deproj is not None else z
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model, i.e. encode and decode."""
        code = self.encode(x)
        return self.decode(code)
