# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the attention module."""

import pytest
import torch
from torch import nn
from chuchichaestli.models.attention.multiscale_attention import LiteMultiscaleAttention


@pytest.mark.parametrize(
    "dimensions,in_channels,out_channels",
    [
        (1, 64, 64),
        (2, 64, 64),
        (3, 64, 64),
        (1, 32, 64),
        (2, 32, 64),
        (3, 32, 64),
        (1, 32, 16),
        (2, 32, 16),
        (3, 32, 16),
    ],
)
def test_lma_init(dimensions, in_channels, out_channels):
    """Test LiteMultiscaleAttention block init."""
    block = LiteMultiscaleAttention(
        dimensions,
        in_channels,
        out_channels,
        act_fn=("silu", "silu"),
        norm_type=("batch", "batch"),
    )
    assert isinstance(block.scale_aggregation, nn.ModuleList)
    assert isinstance(block.proj_out, nn.Module)


@pytest.mark.parametrize(
    "dimensions,in_channels,out_channels",
    [
        (1, 32, 32),
        (2, 32, 32),
        (3, 32, 32),
        (1, 16, 32),
        (2, 16, 32),
        (3, 16, 32),
        (1, 32, 16),
        (2, 32, 16),
        (3, 32, 16),
    ],
)
def test_lma_forward(dimensions, in_channels, out_channels):
    """Test LiteMultiscaleAttention block forward pass."""
    block = LiteMultiscaleAttention(
        dimensions,
        in_channels,
        out_channels,
        act_fn=("silu", "silu"),
        norm_type=("batch", "batch"),
    )
    wh = 16
    shape = (1, in_channels) + (wh,) * dimensions
    sample = torch.randn(shape)
    out = block(sample)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, out_channels) + (wh,) * dimensions
    print(out.shape)


@pytest.mark.parametrize(
    "dimensions,in_channels,out_channels",
    [
        (1, 32, 32),
        (2, 32, 32),
        (3, 32, 32),
        (1, 16, 32),
        (2, 16, 32),
        (3, 16, 32),
        (1, 32, 16),
        (2, 32, 16),
        (3, 32, 16),
    ],
)
def test_lma_backward(dimensions, in_channels, out_channels):
    """Test LiteMultiscaleAttention block backward pass."""
    block = LiteMultiscaleAttention(
        dimensions,
        in_channels,
        out_channels,
        act_fn=("silu", "silu"),
        norm_type=("batch", "batch"),
    )
    wh = 16
    shape = (1, in_channels) + (wh,) * dimensions
    sample = torch.randn(shape)
    out = block(sample)
    gt_virt = torch.randn_like(out)
    loss = (out - gt_virt).pow(2).mean()
    loss.backward()


def test_lma_inspect():
    """Test LiteMultiscaleAttention block inspection."""
    dimensions = 2
    in_channels, out_channels = 64, 128
    block = LiteMultiscaleAttention(
        dimensions,
        in_channels,
        out_channels,
        act_fn=("silu", "silu"),
        norm_type=("batch", "batch"),
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
    pytest.main(["-sv", "test_multiscale_attention.py"])
