# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""A flexible deep-compression autoencoder implementation."""

from chuchichaestli.models.activations import ActivationTypes
from chuchichaestli.models.autoencoder.autoencoder import Autoencoder
from chuchichaestli.models.blocks import (
    AutoencoderDownBlockTypes,
    AutoencoderMidBlockTypes,
    AutoencoderUpBlockTypes,
    EncoderOutBlockTypes,
    DecoderInBlockTypes,
)
from chuchichaestli.models.downsampling import DownsampleTypes
from chuchichaestli.models.norm import NormTypes
from chuchichaestli.models.upsampling import UpsampleTypes
from collections.abc import Sequence


class DCAE(Autoencoder):
    """Flexible deep-compression autoencoder implementation.

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
        n_channels: int = 128,
        latent_dim: int = 32,
        out_channels: int = 1,
        down_block_types: Sequence[AutoencoderDownBlockTypes] = (
            "DCAutoencoderDownBlock",
        )
        * 3
        + ("EfficientViTBlock",) * 3,
        down_layers_per_block: int | Sequence[int] = (2, 2, 2, 3, 3, 3),
        downsample_type: DownsampleTypes = "DownsampleUnshuffle",
        encoder_mid_block_types: Sequence[AutoencoderMidBlockTypes] = (),
        encoder_out_block_type: EncoderOutBlockTypes = "DCEncoderOutBlock",
        decoder_in_block_type: DecoderInBlockTypes = "DCDecoderInBlock",
        decoder_mid_block_types: Sequence[AutoencoderMidBlockTypes] = (),
        up_block_types: Sequence[AutoencoderUpBlockTypes] = ("DCAutoencoderUpBlock",)
        * 3
        + ("EfficientViTBlock",) * 3,
        up_layers_per_block: int | Sequence[int] = 3,
        upsample_type: UpsampleTypes = "UpsampleShuffle",
        block_out_channel_mults: Sequence[int] = (2, 2, 1, 2, 1),
        res_act_fn: ActivationTypes = "silu",
        res_dropout: float = 0.0,
        res_norm_type: NormTypes = "rms",
        res_groups: int = 8,
        res_kernel_size: int = 3,
        attn_head_dim: int = 32,
        attn_n_heads: int = 1,
        attn_dropout_p: float = 0.0,
        attn_norm_type: NormTypes | Sequence[NormTypes] | None = "rms",
        attn_groups: int = 32,
        attn_kernel_size: int = 1,
        attn_scales: Sequence[int] = (5,),
        context_args: dict = dict(norm_type=(None, "rms")),
        local_args: dict = dict(norm_type=((None, None, "rms"))),
        encoder_act_fn: ActivationTypes = "silu",
        encoder_norm_type: NormTypes = "rms",
        encoder_groups: int = 8,
        encoder_kernel_size: int = 3,
        encoder_out_shortcut: bool = True,
        decoder_act_fn: ActivationTypes = "relu",
        decoder_norm_type: NormTypes = "rms",
        decoder_groups: int = 8,
        decoder_kernel_size: int = 3,
        decoder_in_shortcut: bool = True,
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
        """
        super().__init__(
            dimensions=dimensions,
            in_channels=in_channels,
            n_channels=n_channels,
            latent_dim=latent_dim,
            out_channels=out_channels,
            down_block_types=down_block_types,
            down_layers_per_block=down_layers_per_block,
            downsample_type=downsample_type,
            encoder_mid_block_types=encoder_mid_block_types,
            encoder_out_block_type=encoder_out_block_type,
            decoder_in_block_type=decoder_in_block_type,
            decoder_mid_block_types=decoder_mid_block_types,
            up_block_types=up_block_types,
            up_layers_per_block=up_layers_per_block,
            upsample_type=upsample_type,
            block_out_channel_mults=block_out_channel_mults,
            decoder_block_out_channel_mults=tuple(reversed(block_out_channel_mults)),
            use_latent_proj=False,
            use_latent_deproj=False,
            res_act_fn=res_act_fn,
            res_dropout=res_dropout,
            res_norm_type=res_norm_type,
            res_groups=res_groups,
            res_kernel_size=res_kernel_size,
            attn_head_dim=attn_head_dim,
            attn_n_heads=attn_n_heads,
            attn_dropout_p=attn_dropout_p,
            attn_norm_type=attn_norm_type,
            attn_groups=attn_groups,
            attn_kernel_size=attn_kernel_size,
            attn_scales=attn_scales,
            context_args=context_args,
            local_args=local_args,
            encoder_act_fn=encoder_act_fn,
            encoder_norm_type=encoder_norm_type,
            encoder_groups=encoder_groups,
            encoder_kernel_size=encoder_kernel_size,
            encoder_out_shortcut=encoder_out_shortcut,
            decoder_act_fn=decoder_act_fn,
            decoder_norm_type=decoder_norm_type,
            decoder_groups=decoder_groups,
            decoder_kernel_size=decoder_kernel_size,
            decoder_in_shortcut=decoder_in_shortcut,
            double_z=False,
        )
