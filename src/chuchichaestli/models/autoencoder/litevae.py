# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""A lightweight wavelet-domain variational autoencoder."""

import torch
from torch import nn

from chuchichaestli.dwt.modes import ExtensionModeTypes
from chuchichaestli.dwt.wavelet import Wavelet
from chuchichaestli.models.activations import ActivationTypes
from chuchichaestli.models.autoencoder.decoder import Decoder
from chuchichaestli.models.autoencoder.traits import DecoderLike, EncoderLike
from chuchichaestli.models.autoencoder.vae import VAE
from chuchichaestli.models.blocks import SMConvBlock
from chuchichaestli.models.downsampling import AvgPool
from chuchichaestli.models.dwt import MultilevelWaveletTransformND
from chuchichaestli.models.norm import NormTypes
from chuchichaestli.models.unet import UNet
from chuchichaestli.utils import partialclass, prod
from collections.abc import Sequence


__all__ = [
    "LiteVAE",
    "LiteVAEDecoder",
    "LiteVAEEncoder",
    "LiteVAEEncoderB",
    "LiteVAEEncoderL",
    "LiteVAEEncoderM",
    "LiteVAEEncoderS",
    "LiteVAE_B",
    "LiteVAE_L",
    "LiteVAE_M",
    "LiteVAE_S",
]


class LiteVAEEncoder(nn.Module):
    """Wavelet-domain encoding component of a `LiteVAE`.

    The input is decomposed by a multi-level wavelet transform, every level is
    processed by its own constant-resolution U-Net, the results are pooled to a
    common resolution and a second U-Net aggregates them into the latent code.
    All spatial compression comes from the wavelet transform, so the U-Nets keep
    their resolution and only widen between levels.

    Satisfies `EncoderLike`, so it can be passed to `VAE` and `Autoencoder`
    without inheriting from `Encoder`. Like `VAEEncoder` it emits twice its
    latent channels, a mean and a variance.
    """

    def __init__(
        self,
        dimensions: int = 2,
        in_channels: int = 3,
        n_channels: int = 32,
        out_channels: int = 12,
        dwt_levels: int = 3,
        wavelet: str | Wavelet = "haar",
        mode: ExtensionModeTypes = "periodization",
        extractor_channel_mults: Sequence[int] = (3, 1),
        extractor_num_blocks: int = 4,
        aggregator_channels: int | None = None,
        aggregator_channel_mults: Sequence[int] = (3, 1),
        aggregator_num_blocks: int = 4,
        act_fn: ActivationTypes = "silu",
        norm_type: NormTypes = "group",
        num_groups: int = 8,
        res_args: dict = {},
        attn_args: dict = {},
        extractor_args: dict = {},
        aggregator_args: dict = {},
    ):
        """Constructor.

        Args:
            dimensions: Number of dimensions.
            in_channels: Number of input channels.
            n_channels: Number of channels in the feature extractors.
            out_channels: Number of latent channels; twice as many are emitted.
            dwt_levels: Number of wavelet levels, and hence the compression factor.
            wavelet: Wavelet to decompose with, by name or as a `Wavelet`.
            mode: Signal extension mode; the default is critically sampled, and
                hence the only one that halves each axis exactly.
            extractor_channel_mults: Channel multiplier per level of an extractor,
                as the ratio to the level before it.
            extractor_num_blocks: Number of blocks per level of an extractor.
            aggregator_channels: Number of channels in the aggregator; as many as
                the extractors if omitted.
            aggregator_channel_mults: Channel multiplier per level of the
                aggregator, as the ratio to the level before it.
            aggregator_num_blocks: Number of blocks per level of the aggregator.
            act_fn: Activation function.
            norm_type: Normalization type.
            num_groups: Number of groups for the normalization.
            res_args: Arguments for the residual blocks; those a `UNet` does
                not take are ignored.
            attn_args: Arguments for the attention blocks; those a `UNet` does
                not take are ignored.
            extractor_args: Further architecture arguments for the extractors.
            aggregator_args: Further architecture arguments for the aggregator.

        Raises:
            ValueError: If fewer than one wavelet level is requested.
        """
        super().__init__()
        if dwt_levels < 1:
            raise ValueError(f"A wavelet encoder needs at least one level; got {dwt_levels}.")
        aggregator_channels = (
            n_channels if aggregator_channels is None else aggregator_channels
        )
        band_channels = in_channels * 2**dimensions

        self.dimensions = dimensions
        self.in_channels = in_channels
        self.n_channels = n_channels
        self.latent_channels = out_channels
        self.out_channels = 2 * out_channels
        self.levels = dwt_levels
        self.channel_mults = prod(aggregator_channel_mults)
        self.bottleneck_channels = aggregator_channels * self.channel_mults

        self.dwt = MultilevelWaveletTransformND(
            dimensions, wavelet, mode, "subband", levels=dwt_levels
        )
        shared = {
            "act_fn": act_fn,
            "norm_type": norm_type,
            "groups": num_groups,
            "res_groups": num_groups,
            "downsample_type": "ChannelResample",
            "upsample_type": "ChannelResample",
            "time_embedding": None,
            **res_args,
            **{
                f"attn_{name}": attn_args[name]
                for name in (
                        "n_heads",
                        "head_dim",
                        "dropout_p",
                        "norm_type",
                        "groups",
                        "kernel_size",
                )  # UNet attn_args
                if name in attn_args
            },
        }
        extractor_levels = len(extractor_channel_mults)
        aggregator_levels = len(aggregator_channel_mults)

        self.extractors = nn.ModuleList(
            UNet(
                dimensions=dimensions,
                in_channels=band_channels,
                n_channels=n_channels,
                out_channels=band_channels,
                down_block_types=("DownBlock",) * extractor_levels,
                mid_block_type="MidBlock",
                up_block_types=("UpBlock",) * extractor_levels,
                block_out_channel_mults=tuple(extractor_channel_mults),
                num_blocks_per_level=extractor_num_blocks,
                **{**shared, **extractor_args},
            )
            for _ in range(dwt_levels)
        )
        # every level is pooled down to the resolution of the coarsest one
        self.pools = nn.ModuleList(
            AvgPool(dimensions, kernel_size=factor, stride=factor, padding=0)
            if (factor := 2 ** (dwt_levels - 1 - level)) > 1
            else nn.Identity()
            for level in range(dwt_levels)
        )
        self.aggregator = UNet(
            dimensions=dimensions,
            in_channels=dwt_levels * band_channels,
            n_channels=aggregator_channels,
            out_channels=self.out_channels,
            down_block_types=("DownBlock",) * aggregator_levels,
            mid_block_type="MidBlock",
            up_block_types=("UpBlock",) * aggregator_levels,
            block_out_channel_mults=tuple(aggregator_channel_mults),
            num_blocks_per_level=aggregator_num_blocks,
            **{**shared, **aggregator_args},
        )

    @property
    def f(self) -> int:
        """Compression factor of the encoder, which the wavelet levels alone set."""
        return 2**self.levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder."""
        with torch.no_grad():
            bands = self.dwt(x)
        features = [
            pool(extractor(band.detach()))
            for extractor, pool, band in zip(
                self.extractors, self.pools, bands, strict=True
            )
        ]
        return self.aggregator(torch.cat(features, dim=1))


class LiteVAEDecoder(Decoder):
    """Decoding component of a `LiteVAE`.

    Structurally the decoder of a standard latent diffusion autoencoder, with
    self-modulated convolutions in place of the normalizations of its residual
    blocks; see `Decoder` for the architecture arguments. The attention block of
    the bottleneck keeps its own normalization.
    """

    def __init__(
        self,
        dimensions: int = 2,
        in_channels: int = 12,
        n_channels: int = 512,
        out_channels: int = 3,
        up_block_types: Sequence[str] = ("SMConvAutoencoderUpBlock",) * 4,
        mid_block_types: Sequence[str] = (
            "SMConvAutoencoderMidBlock",
            "AttnAutoencoderMidBlock",
        ),
        block_out_channel_mults: Sequence[int] = (1, 1, 2, 2),
        num_layers_per_block: int | Sequence[int] = 3,
        act_fn: ActivationTypes = "silu",
        num_groups: int = 8,
        kernel_size: int = 3,
        res_args: dict = {},
        **kwargs,
    ):
        """Constructor.

        Args:
            dimensions: Number of dimensions.
            in_channels: Number of latent channels consumed.
            n_channels: Number of channels the decoder starts from.
            out_channels: Number of output channels.
            up_block_types: Type of up blocks to use for each level.
            mid_block_types: Type of blocks to use after the input.
            block_out_channel_mults: Divisor for the output channels of each block.
            num_layers_per_block: Number of blocks per level.
            act_fn: Activation function.
            num_groups: Number of groups for the normalization; also the default
                for the residual blocks, which are narrower here than usual.
            kernel_size: Kernel size for the output convolution.
            res_args: Arguments for the residual blocks.
            kwargs: Further architecture arguments for `Decoder`.
        """
        super().__init__(
            dimensions=dimensions,
            in_channels=in_channels,
            n_channels=n_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            mid_block_types=mid_block_types,
            block_out_channel_mults=block_out_channel_mults,
            num_layers_per_block=num_layers_per_block,
            act_fn=act_fn,
            num_groups=num_groups,
            kernel_size=kernel_size,
            res_args={"res_groups": num_groups, **res_args},
            **kwargs,
        )
        # the output block carries a normalization too, so it is replaced in kind
        self.out_block = SMConvBlock(
            dimensions,
            self.out_block.conv.in_channels,
            out_channels,
            act_fn=act_fn,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )


class LiteVAE(VAE):
    """Lightweight wavelet-domain variational autoencoder.

    Replaces the convolutional encoder of a variational autoencoder with a
    multi-level wavelet transform followed by small per-scale feature
    extractors and an aggregation network, which reaches the same
    reconstruction quality with a fraction of the encoder parameters.

    Attributes:
        encoder_cls: Encoder class that `build` instantiates.
        decoder_cls: Decoder class that `build` instantiates.
    """

    encoder_cls: type = LiteVAEEncoder
    decoder_cls: type = LiteVAEDecoder

    def __init__(
        self,
        encoder: EncoderLike,
        decoder: DecoderLike,
        latent_proj: nn.Module | bool = False,
        latent_deproj: nn.Module | bool = False,
    ):
        """Assemble a wavelet-domain variational autoencoder from its components.

        Args:
            encoder: Encoding component; must double its latent channels.
            decoder: Decoding component, expanding the latent space to the output.
            latent_proj: Projection between encoder and latent space; omitted by
                default, which reconstructs better than a pointwise convolution.
            latent_deproj: Projection between latent space and decoder.
        """
        super().__init__(encoder, decoder, latent_proj, latent_deproj)

    @classmethod
    def build(
        cls,
        dimensions: int = 2,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_dim: int = 12,
        decoder_n_channels: int = 512,
        **kwargs,
    ) -> "LiteVAE":
        """Build a model from architecture arguments, components included.

        Every architecture argument is open; the published sizes are available
        as the `LiteVAE_S`, `LiteVAE_B`, `LiteVAE_M` and `LiteVAE_L` subclasses.

        Args:
            dimensions: Number of dimensions for the model.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            latent_dim: Number of channels in the latent space.
            decoder_n_channels: Number of channels the decoder starts from.
            kwargs: Further arguments for `Autoencoder.build`.
        """
        return super().build(
            dimensions=dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim,
            decoder_n_channels=decoder_n_channels,
            **kwargs,
        )


# The published encoder sizes
LiteVAEEncoderS = partialclass(
    "LiteVAEEncoderS",
    LiteVAEEncoder,
    n_channels=16,
    extractor_channel_mults=(2, 1),
    extractor_num_blocks=4,
    aggregator_channels=16,
    aggregator_channel_mults=(2, 1),
    aggregator_num_blocks=4,
    __doc__="""Encoding component sized like the small published variant.""",
)

LiteVAEEncoderB = partialclass(
    "LiteVAEEncoderB",
    LiteVAEEncoder,
    n_channels=32,
    extractor_channel_mults=(3, 1),
    extractor_num_blocks=4,
    aggregator_channels=32,
    aggregator_channel_mults=(3, 1),
    aggregator_num_blocks=4,
    __doc__="""Encoding component sized like the base published variant (LiteVAEEncoder default).""",
)

LiteVAEEncoderM = partialclass(
    "LiteVAEEncoderM",
    LiteVAEEncoder,
    n_channels=64,
    extractor_channel_mults=(2, 2),
    extractor_num_blocks=4,
    aggregator_channels=64,
    aggregator_channel_mults=(2, 2),
    aggregator_num_blocks=4,
    __doc__="""Encoding component sized like the medium published variant.""",
)

LiteVAEEncoderL = partialclass(
    "LiteVAEEncoderL",
    LiteVAEEncoder,
    n_channels=64,
    extractor_channel_mults=(2, 2),
    extractor_num_blocks=4,
    aggregator_channels=96,
    aggregator_channel_mults=(2, 2),
    aggregator_num_blocks=4,
    __doc__="""Encoding component sized like the large published variant.

    Shares the feature extractors of the medium variant and widens only the
    aggregator.
    """,
)


class LiteVAE_S(LiteVAE):
    """`LiteVAE` with the small published encoder, about 1M parameters."""

    encoder_cls: type = LiteVAEEncoderS


class LiteVAE_B(LiteVAE):
    """`LiteVAE` with the base published encoder, about 7M parameters (default)."""

    encoder_cls: type = LiteVAEEncoderB


class LiteVAE_M(LiteVAE):
    """`LiteVAE` with the medium published encoder, about 32M parameters."""

    encoder_cls: type = LiteVAEEncoderM


class LiteVAE_L(LiteVAE):
    """`LiteVAE` with the large published encoder, about 42M parameters."""

    encoder_cls: type = LiteVAEEncoderL
