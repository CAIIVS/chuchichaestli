# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""A flexible deep-compression autoencoder implementation."""

from torch import nn
from chuchichaestli.models.activations import ActivationTypes
from chuchichaestli.models.autoencoder.autoencoder import Autoencoder
from chuchichaestli.models.autoencoder.decoder import Decoder
from chuchichaestli.models.autoencoder.encoder import Encoder
from chuchichaestli.models.autoencoder.traits import DecoderLike, EncoderLike
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


__all__ = ["DCAE", "DCDecoder", "DCEncoder"]


# Block arguments both components share; `DCAE.build` repeats them as its own
# defaults so that building and injecting the components give the same model.
DC_RES_ARGS = {"res_norm_type": "rms"}
DC_ATTN_ARGS = {
    "norm_type": "rms",
    "context_args": {"norm_type": (None, "rms")},
    "local_args": {"norm_type": (None, None, "rms")},
}


class DCEncoder(Encoder):
    """Encoding component of a deep-compression autoencoder.

    Compresses with channel-shuffling downsamplers and efficient attention at
    the deeper levels; see `Encoder` for what each argument does.
    """

    def __init__(
        self,
        dimensions: int = 2,
        in_channels: int = 1,
        n_channels: int = 128,
        out_channels: int = 32,
        down_block_types: Sequence[AutoencoderDownBlockTypes] = (
            "DCAutoencoderDownBlock",
        )
        * 3
        + ("EfficientViTBlock",) * 3,
        block_out_channel_mults: Sequence[int] = (2, 2, 1, 2, 1),
        num_layers_per_block: int | Sequence[int] = (2, 2, 2, 3, 3, 3),
        mid_block_types: Sequence[AutoencoderMidBlockTypes] = (),
        out_block_type: EncoderOutBlockTypes = "DCEncoderOutBlock",
        downsample_type: DownsampleTypes = "DownsampleUnshuffle",
        act_fn: ActivationTypes = "silu",
        norm_type: NormTypes = "rms",
        num_groups: int = 8,
        kernel_size: int = 3,
        res_args: dict = {},
        attn_args: dict = {},
        out_shortcut: bool = True,
    ):
        """Constructor.

        Args:
            dimensions: Number of dimensions.
            in_channels: Number of input channels.
            n_channels: Number of channels in the hidden layer.
            out_channels: Number of output channels (latent space).
            down_block_types: Type of down blocks to use for each level.
            block_out_channel_mults: Multiplier for output channels of each block.
            num_layers_per_block: Number of blocks per level.
            mid_block_types: Type of blocks to use before the output.
            out_block_type: Type of block for output (latent space).
            downsample_type: Type of downsampling block, per level.
            act_fn: Activation function for the output layers.
            norm_type: Normalization type for the output layer.
            num_groups: Number of groups for normalization in the output layer.
            kernel_size: Kernel size for the output convolution.
            res_args: Arguments for residual blocks, overriding `DC_RES_ARGS`.
            attn_args: Arguments for attention blocks, overriding `DC_ATTN_ARGS`.
            out_shortcut: Whether to use a shortcut for the output block.
        """
        super().__init__(
            dimensions=dimensions,
            in_channels=in_channels,
            n_channels=n_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            block_out_channel_mults=block_out_channel_mults,
            num_layers_per_block=num_layers_per_block,
            mid_block_types=mid_block_types,
            out_block_type=out_block_type,
            downsample_type=downsample_type,
            act_fn=act_fn,
            norm_type=norm_type,
            num_groups=num_groups,
            kernel_size=kernel_size,
            res_args={**DC_RES_ARGS, **res_args},
            attn_args={**DC_ATTN_ARGS, **attn_args},
            out_shortcut=out_shortcut,
        )


class DCDecoder(Decoder):
    """Decoding component of a deep-compression autoencoder.

    Expands with channel-shuffling upsamplers, mirroring `DCEncoder`; see
    `Decoder` for what each argument does.
    """

    def __init__(
        self,
        dimensions: int = 2,
        in_channels: int = 32,
        n_channels: int = 1024,
        out_channels: int = 1,
        in_block_type: DecoderInBlockTypes = "DCDecoderInBlock",
        mid_block_types: Sequence[AutoencoderMidBlockTypes] = (),
        up_block_types: Sequence[AutoencoderUpBlockTypes] = ("DCAutoencoderUpBlock",)
        * 3
        + ("EfficientViTBlock",) * 3,
        block_out_channel_mults: Sequence[int] = (1, 2, 1, 2, 2),
        num_layers_per_block: int | Sequence[int] = 3,
        upsample_type: UpsampleTypes = "UpsampleShuffle",
        act_fn: ActivationTypes = "relu",
        norm_type: NormTypes = "rms",
        num_groups: int = 8,
        kernel_size: int = 3,
        res_args: dict = {},
        attn_args: dict = {},
        in_shortcut: bool = True,
    ):
        """Constructor.

        Args:
            dimensions: Number of dimensions.
            in_channels: Number of input channels (latent space).
            n_channels: Number of channels for the first block.
            out_channels: Number of output channels.
            in_block_type: Type of block for input (latent space).
            mid_block_types: Type of blocks to use after the input.
            up_block_types: Type of up blocks to use for each level.
            block_out_channel_mults: Divisor for output channels of each block.
            num_layers_per_block: Number of blocks per level.
            upsample_type: Type of upsampling block, per level.
            act_fn: Activation function for the output layers.
            norm_type: Normalization type for the output layer.
            num_groups: Number of groups for normalization in the output layer.
            kernel_size: Kernel size for the output convolution.
            res_args: Arguments for residual blocks, overriding `DC_RES_ARGS`.
            attn_args: Arguments for attention blocks, overriding `DC_ATTN_ARGS`.
            in_shortcut: Whether to use a shortcut for the input block.
        """
        super().__init__(
            dimensions=dimensions,
            in_channels=in_channels,
            n_channels=n_channels,
            out_channels=out_channels,
            in_block_type=in_block_type,
            mid_block_types=mid_block_types,
            up_block_types=up_block_types,
            block_out_channel_mults=block_out_channel_mults,
            num_layers_per_block=num_layers_per_block,
            upsample_type=upsample_type,
            act_fn=act_fn,
            norm_type=norm_type,
            num_groups=num_groups,
            kernel_size=kernel_size,
            res_args={**DC_RES_ARGS, **res_args},
            attn_args={**DC_ATTN_ARGS, **attn_args},
            in_shortcut=in_shortcut,
        )


class DCAE(Autoencoder):
    """Flexible deep-compression autoencoder implementation.

    The architecture consists of an encoder-decoder structure.
    The encoder chains several residual and downsampling blocks.
    Each downsampling block separates the encoder into spatially hierarchical levels.
    The encoder ends in bottleneck blocks (optionally including attention blocks
    and a convolutional layer) and projects the input into latent space.
    The decoder is built with residual convolutional and upsampling blocks, and
    expands from the latent space to the image domain.

    The compression happens in the blocks themselves, so no latent projection
    is used by default.

    Attributes:
        encoder_cls: Encoder class that `build` instantiates.
        decoder_cls: Decoder class that `build` instantiates.
    """

    encoder_cls: type = DCEncoder
    decoder_cls: type = DCDecoder

    def __init__(
        self,
        encoder: EncoderLike,
        decoder: DecoderLike,
        latent_proj: nn.Module | bool = False,
        latent_deproj: nn.Module | bool = False,
    ):
        """Assemble a deep-compression autoencoder from its two components.

        Args:
            encoder: Encoding component, mapping the input to latent space.
            decoder: Decoding component, expanding the latent space to the output.
            latent_proj: Projection between encoder and latent space (see `Autoencoder`).
            latent_deproj: Projection between latent space and decoder.
        """
        super().__init__(encoder, decoder, latent_proj, latent_deproj)

    @classmethod
    def build(
        cls,
        latent_dim: int = 32,
        res_norm_type: NormTypes = "rms",
        attn_norm_type: NormTypes | Sequence[NormTypes] | None = "rms",
        context_args: dict = DC_ATTN_ARGS["context_args"],
        local_args: dict = DC_ATTN_ARGS["local_args"],
        **kwargs,
    ) -> "DCAE":
        """Build a deep-compression autoencoder from architecture arguments.

        Only the arguments whose defaults differ from `Autoencoder.build` are
        named here; the components supply every structural default.

        Args:
            latent_dim: Number of channels in the latent space.
            res_norm_type: Normalization type for the residual blocks.
            attn_norm_type: Normalization type for the attention blocks.
            context_args: Keyword arguments for the context block in a transformer module.
            local_args: Keyword arguments for the local block in a transformer module.
            kwargs: Further architecture arguments, see `Autoencoder.build`.

        Returns:
            An assembled model of the class this was called on.
        """
        return super().build(
            latent_dim=latent_dim,
            res_norm_type=res_norm_type,
            attn_norm_type=attn_norm_type,
            context_args=context_args,
            local_args=local_args,
            **kwargs,
        )
