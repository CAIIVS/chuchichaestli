# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Autoencoder block tests."""

import pytest
import torch
from torch import nn
from chuchichaestli.models.blocks import (
    AutoencoderDownBlock,
    AutoencoderMidBlock,
    AutoencoderUpBlock,
    GLUMBConvBlock,
    GLUMBResBlock,
    LMAResBlock,
    EfficientViTBlock,
)
from chuchichaestli.models.autoencoder.decoder import Decoder
from chuchichaestli.models.autoencoder.encoder import Encoder
from chuchichaestli.models.autoencoder.traits import DecoderLike, EncoderLike


@pytest.mark.parametrize(
    "block_cls,dimensions,in_channels,out_channels",
    [
        (AutoencoderDownBlock, 1, 32, 64),
        (AutoencoderDownBlock, 2, 32, 64),
        (AutoencoderDownBlock, 3, 32, 64),
        (AutoencoderUpBlock, 1, 32, 32),
        (AutoencoderUpBlock, 2, 32, 32),
        (AutoencoderUpBlock, 3, 32, 64),
        (AutoencoderMidBlock, 1, 32, 32),
        (AutoencoderMidBlock, 2, 32, 32),
        (AutoencoderMidBlock, 3, 32, 32),
    ],
)
def test_autoencoder_blocks(
    block_cls,
    dimensions,
    in_channels,
    out_channels,
):
    """Test and inspect autoencoder blocks."""
    if block_cls == AutoencoderMidBlock:
        block = block_cls(
            dimensions,
            in_channels,
        )
    else:
        block = block_cls(
            dimensions,
            in_channels,
            out_channels,
        )
    assert isinstance(block, nn.Module)
    assert (
        isinstance(block.res_block.shortcut, nn.Identity)
        if in_channels == out_channels
        else isinstance(block.res_block.shortcut, nn.Conv1d | nn.Conv2d | nn.Conv3d)
    )


def test_inspect_autoencoder_down_block():
    """Inspect AutoencoderDownBlock."""
    w, h = 512, 512
    block = AutoencoderDownBlock(2, 32, 64)
    try:
        from torchinfo import summary

        summary(
            block,
            (1, 32, w, h),
            col_names=["input_size", "output_size", "kernel_size", "num_params"],
            depth=4,
        )
    except ImportError:
        print(block)
    print()


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,out_channels",
    [
        (1, 1, 64, 4),
        (2, 1, 64, 4),
        (3, 1, 64, 4),
    ],
)
def test_autoencoder_encoder(dimensions, in_channels, n_channels, out_channels):
    """Test Encoder module."""
    encoder = Encoder(dimensions, in_channels, n_channels, out_channels)
    assert encoder.levels == 4  # 4 stages (each w/ 2 res blocks) + 3 downsamplings
    assert len(encoder.mid_blocks) == 2  # 2 default blocks


def test_autoencoder_encoder_inspect():
    """Inspect Encoder module."""
    dimensions, in_channels, n_channels, out_channels = 2, 1, 64, 4
    encoder = Encoder(dimensions, in_channels, n_channels, out_channels)
    try:
        from torchinfo import summary

        summary(
            encoder,
            (1, in_channels, 512, 512),
            col_names=["input_size", "output_size", "kernel_size", "num_params"],
            depth=4,
        )
    except ImportError:
        print(encoder)
    print()


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,out_channels",
    [
        (1, 4, 512, 1),
        (2, 4, 512, 1),
        (3, 4, 512, 1),
    ],
)
def test_autoencoder_decoder(dimensions, in_channels, n_channels, out_channels):
    """Test Decoder module."""
    decoder = Decoder(dimensions, in_channels, n_channels, out_channels)
    assert decoder.levels == 4  # 4 stages (w/ each 3 residual blocks) + 3 downsamplings
    assert len(decoder.mid_blocks) == 2  # 2 default blocks


def test_autoencoder_decoder_inspect():
    """Inspect Decoder module."""
    dimensions, in_channels, n_channels, out_channels = 2, 4, 512, 1
    decoder = Decoder(dimensions, in_channels, n_channels, out_channels)
    try:
        from torchinfo import summary

        summary(
            decoder,
            (1, in_channels, 4, 4),
            col_names=["input_size", "output_size", "kernel_size", "num_params"],
            depth=4,
        )
    except ImportError:
        print(decoder)
    print()


@pytest.mark.parametrize(
    "dimensions,in_channels,out_channels,expansion",
    [
        (1, 64, 64, 4),
        (2, 64, 64, 4),
        (3, 64, 64, 4),
    ],
)
def test_glumbconv_block(dimensions, in_channels, out_channels, expansion):
    """Test GLUMBConvBlock."""
    block = GLUMBConvBlock(
        dimensions,
        in_channels,
        out_channels,
        expansion,
    )
    wh = 16
    sample = torch.randn(1, in_channels, *((wh,) * dimensions))
    out = block(sample)
    print(out.shape)


@pytest.mark.parametrize(
    "dimensions,in_channels,out_channels,expansion",
    [
        (1, 64, 64, 4),
        (2, 64, 64, 4),
        (3, 64, 64, 4),
    ],
)
def test_glumbres_block(dimensions, in_channels, out_channels, expansion):
    """Test GLUMBResBlock."""
    block = GLUMBResBlock(
        dimensions,
        in_channels,
        out_channels,
        expansion,
    )
    wh = 16
    sample = torch.randn(1, in_channels, *((wh,) * dimensions))
    out = block(sample)
    print(out.shape)


@pytest.mark.parametrize(
    "dimensions,in_channels,out_channels,heads",
    [
        (1, 64, 64, 8),
        (2, 64, 64, 8),
        (3, 64, 64, 8),
    ],
)
def test_lmares_block(dimensions, in_channels, out_channels, heads):
    """Test LMAResBlock."""
    block = LMAResBlock(
        dimensions,
        in_channels,
        out_channels,
        heads,
    )
    wh = 16
    sample = torch.randn(1, in_channels, *((wh,) * dimensions))
    out = block(sample)
    print(out.shape)


def test_glumbconv_block_inspect():
    """Test GLUMBConvBlock."""
    dimensions, in_channels, out_channels, expansion = 2, 64, 64, 4
    block = GLUMBConvBlock(
        dimensions,
        in_channels,
        out_channels,
        expansion,
    )
    try:
        from torchinfo import summary

        wh = 16
        summary(
            block,
            (1, in_channels) + (wh,) * dimensions,
            col_names=["input_size", "output_size", "kernel_size", "num_params"],
            depth=4,
        )
    except ImportError:
        print(block)
    print()


def test_efficientvit_block_inspect():
    """Test EfficientViTBlock."""
    dimensions, in_channels, out_channels, expansion = 2, 64, 64, 4
    block = EfficientViTBlock(
        dimensions, in_channels, out_channels, expansion=expansion
    )
    try:
        from torchinfo import summary

        wh = 16
        summary(
            block,
            (1, in_channels) + (wh,) * dimensions,
            col_names=["input_size", "output_size", "kernel_size", "num_params"],
            depth=4,
        )
    except ImportError:
        print(block)
    print()


if __name__ == "__main__":
    pytest.main(["-sv", "test_autoencoder_blocks.py"])


ENC_CONF = {
    "n_channels": 16,
    "block_out_channel_mults": (1, 2, 2),
    "num_layers_per_block": 1,
    "down_block_types": ("AutoencoderDownBlock",) * 3,
    "mid_block_types": ("AutoencoderMidBlock", "AttnAutoencoderMidBlock"),
}
DEC_CONF = {
    "n_channels": 64,
    "block_out_channel_mults": (1, 2, 2),
    "num_layers_per_block": 1,
    "up_block_types": ("AutoencoderUpBlock",) * 3,
    "mid_block_types": ("AutoencoderMidBlock", "AttnAutoencoderMidBlock"),
}


def test_encoder_per_level_args_run_levels_then_mid_blocks():
    """Test that an encoder reads a per-level sequence as levels, then mid blocks."""
    encoder = Encoder(
        **ENC_CONF, res_args={"res_dropout": (0.1, 0.2, 0.3, 0.4, 0.5), "res_groups": 4}
    )
    assert [s[0].res_block.dropout.p for s in encoder.down_blocks[::2]] == [
        0.1,
        0.2,
        0.3,
    ]
    assert [m.res_block.dropout.p for m in encoder.mid_blocks] == [0.4, 0.5]


def test_decoder_per_level_args_run_mid_blocks_then_levels():
    """Test that a decoder reads a per-level sequence as mid blocks, then levels."""
    decoder = Decoder(
        **DEC_CONF, res_args={"res_dropout": (0.5, 0.4, 0.3, 0.2, 0.1), "res_groups": 4}
    )
    assert [m.res_block.dropout.p for m in decoder.mid_blocks] == [0.5, 0.4]
    assert [s[0].res_block.dropout.p for s in decoder.up_blocks[::2]] == [0.3, 0.2, 0.1]


def test_single_values_still_reach_every_position():
    """Test that a single value in res_args is applied to levels and mid blocks alike."""
    encoder = Encoder(**ENC_CONF, res_args={"res_dropout": 0.25, "res_groups": 4})
    stages = [s[0].res_block.dropout.p for s in encoder.down_blocks[::2]]
    assert stages + [m.res_block.dropout.p for m in encoder.mid_blocks] == [0.25] * 5


def test_opaque_attention_keys_are_passed_through():
    """Test that a norm_type sequence keeps its intra-block meaning."""
    encoder = Encoder(
        n_channels=16,
        block_out_channel_mults=(1, 2),
        num_layers_per_block=1,
        down_block_types=("AutoencoderDownBlock",) * 2,
        mid_block_types=(),
        res_args={"res_groups": 4},
        attn_args={"norm_type": ("rms", "group")},
    )
    assert encoder.levels == 2


def test_throws_error_on_wrong_per_level_length():
    """Test that a sequence matching no accepted position count is rejected."""
    with pytest.raises(ValueError):
        Encoder(**ENC_CONF, res_args={"res_dropout": (0.1, 0.2)})


def test_caller_sequences_are_not_mutated():
    """Test that padding the channel multipliers leaves the caller's list alone."""
    mults = [1, 2]
    Encoder(
        n_channels=16,
        down_block_types=("AutoencoderDownBlock",) * 4,
        block_out_channel_mults=mults,
        num_layers_per_block=1,
        mid_block_types=(),
        res_args={"res_groups": 4},
    )
    assert mults == [1, 2]


def test_encoder_per_level_sampling_types():
    """Test that an encoder can mix sampling types and still shrink correctly."""
    encoder = Encoder(
        n_channels=16,
        block_out_channel_mults=(1, 2, 2),
        num_layers_per_block=1,
        down_block_types=("AutoencoderDownBlock",) * 3,
        mid_block_types=(),
        res_args={"res_groups": 4},
        downsample_type=("Downsample", "DownsampleUnshuffle", "Downsample"),
    )
    assert [type(b).__name__ for b in encoder.down_blocks][1::2] == [
        "Downsample",
        "DownsampleUnshuffle",
    ]
    assert encoder(torch.randn(1, 1, 32, 32)).shape[-2:] == (8, 8)


F_CONF = {
    "n_channels": 16,
    "block_out_channel_mults": (1, 2, 2),
    "num_layers_per_block": 1,
    "down_block_types": ("AutoencoderDownBlock",) * 3,
    "mid_block_types": (),
    "res_args": {"res_groups": 4},
}


def test_compression_factor_is_the_product_of_the_sampler_factors():
    """Test that the compression factor comes from the samplers that were built."""
    from chuchichaestli.models.downsampling import Downsample

    encoder = Encoder(**F_CONF)
    assert encoder.f == 4

    encoder.down_blocks[1] = Downsample(2, 16, stride=4)
    assert encoder.f == 8


def test_compression_factor_rejects_a_fixed_size_sampler():
    """Test that a pool with a fixed output size has no constant compression factor."""
    from chuchichaestli.models.downsampling import AdaptiveMaxPool

    encoder = Encoder(**F_CONF)
    encoder.down_blocks[1] = AdaptiveMaxPool(2, 16, output_size=(8, 8))
    with pytest.raises(ValueError, match="fixed size"):
        encoder.f


def test_components_expose_the_metadata_an_autoencoder_reads():
    """Test that both components satisfy their structural interface."""
    encoder = Encoder(**F_CONF)
    decoder = Decoder(2, 4, 64, 1, mid_block_types=(), res_args={"res_groups": 4})
    assert isinstance(encoder, EncoderLike)
    assert isinstance(decoder, DecoderLike)
    assert not isinstance(nn.Identity(), EncoderLike)


def test_encoder_reports_the_width_entering_its_out_block():
    """Test that the bottleneck width follows the levels that were built."""
    encoder = Encoder(**F_CONF)
    assert encoder.bottleneck_channels == F_CONF["n_channels"] * encoder.channel_mults
    assert encoder.out_block.conv.in_channels == encoder.bottleneck_channels


def test_encoder_keeps_latent_channels_undoubled_by_default():
    """Test that an encoder emits one latent channel per requested channel."""
    encoder = Encoder(**F_CONF, out_channels=4)
    assert encoder.double_z is False
    assert encoder.out_channels == 4
    assert Encoder(**F_CONF, out_channels=4, double_z=True).out_channels == 8
