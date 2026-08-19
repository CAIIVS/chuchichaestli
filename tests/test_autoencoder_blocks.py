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
