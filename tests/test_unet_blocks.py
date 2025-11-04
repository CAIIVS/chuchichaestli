# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the U-Net blocks module."""

import pytest
import torch
from chuchichaestli.models.unet import UNet
from chuchichaestli.models.blocks import (
    GaussianNoiseBlock,
    DownBlock,
    MidBlock,
    UpBlock,
)


@pytest.fixture
def net_conf():
    """Fixture to provide a configuration for the UNet model."""
    return {
        "dimensions": 3,
        "in_channels": 1,
        "n_channels": 8,
        "out_channels": 1,
        "down_block_types": ["DownBlock", "DownBlock", "DownBlock"],
        "mid_block_type": "MidBlock",
        "up_block_types": ["UpBlock", "UpBlock", "UpBlock"],
        "block_out_channel_mults": [2, 2, 2],
        "num_blocks_per_level": 1,
        "skip_connection_action": "concat",
        "skip_connection_to_all_blocks": True,
    }


def count_res_blocks(blocks):
    """Count the number of residual blocks in a list of blocks."""
    return sum(1 for block in blocks if hasattr(block, "res_block"))


@pytest.mark.filterwarnings("ignore:Number of channels")
@pytest.mark.parametrize("num_blocks_per_level, expected_res_blocks", [(1, 3), (2, 6)])
def test_res_blocks(net_conf, num_blocks_per_level, expected_res_blocks):
    """Test the number of residual blocks in the UNet model."""
    net_conf["num_blocks_per_level"] = num_blocks_per_level
    model = UNet(**net_conf)

    encoder_res_blocks = count_res_blocks(model.down_blocks)
    decoder_res_blocks = count_res_blocks(model.up_blocks)

    print(f"Number of res_blocks in encoder: {encoder_res_blocks}")
    print(f"Number of res_blocks in decoder: {decoder_res_blocks}")

    assert encoder_res_blocks == expected_res_blocks
    assert decoder_res_blocks == expected_res_blocks


def test_gaussian_noise_block_forward_training():
    """Test the GaussianNoiseBlock in training mode."""
    torch.manual_seed(0)
    block = GaussianNoiseBlock(sigma=0.1, mu=0.0, detached=True)
    block.train()
    x = torch.ones(2, 3)
    y = block(x)
    assert y.shape == x.shape
    assert not torch.equal(x, y)


def test_gaussian_noise_block_forward_eval_no_noise():
    """Test the GaussianNoiseBlock in evaluation mode with no noise."""
    block = GaussianNoiseBlock(sigma=0.0)
    block.eval()
    x = torch.ones(2, 3)
    y = block(x)
    assert torch.equal(x, y)


def test_down_block_forward_default():
    """Test the DownBlock (default)."""
    block = DownBlock(2, 16, 32, res_args={"res_groups": 8})
    x = torch.randn(3, 16, 64, 64)
    y = block(x)
    assert y.shape == (3, 32, 64, 64)


def test_down_block_forward_with_time_embedding():
    """Test the DownBlock with time embedding."""
    block = DownBlock(2, 16, 32, time_embedding=True, res_args={"res_groups": 8})
    x = torch.randn(3, 16, 64, 64)
    t = torch.randn(3, 32)
    y = block(x, t)
    assert y.shape == (3, 32, 64, 64)


@pytest.mark.parametrize("attention", ["self_attention", "conv_attention"])
def test_down_block_forward_with_attention(attention):
    """Test the DownBlock (default)."""
    block = DownBlock(
        2,
        16,
        32,
        attention=attention,
        res_args={"res_groups": 8},
        attn_args={"groups": 8},
    )
    x = torch.randn(3, 16, 64, 64)
    y = block(x)
    assert y.shape == (3, 32, 64, 64)


@pytest.mark.parametrize("attention", [None, "self_attention", "conv_attention"])
def test_mid_block_forward(attention):
    """Test the MidBlock."""
    block = MidBlock(
        2, 16, attention=attention, res_args={"res_groups": 8}, attn_args={"groups": 8}
    )
    x = torch.randn(3, 16, 64, 64)
    y = block(x)
    assert y.shape == (3, 16, 64, 64)


def test_up_block_forward_default():
    """Test the UpBlock (default)."""
    block = UpBlock(2, 32, 16, res_args={"res_groups": 8})
    x = torch.randn(3, 32, 64, 64)
    h = torch.randn(3, 32, 64, 64)
    y = block(x, h)
    assert y.shape == (3, 16, 64, 64)


def test_up_block_forward_with_time_embedding():
    """Test the UpBlock (default)."""
    block = UpBlock(2, 32, 16, time_embedding=True, res_args={"res_groups": 8})
    x = torch.randn(3, 32, 64, 64)
    h = torch.randn(3, 32, 64, 64)
    t = torch.randn(3, 32)
    y = block(x, h, t)
    assert y.shape == (3, 16, 64, 64)


def test_up_block_init_error():
    """Test the UpBlock constructor for invalid skip_connection_action."""
    with pytest.raises(ValueError):
        UpBlock(
            2,
            32,
            16,
            res_args={"res_groups": 8},
            skip_connection_action="invalid_action",
        )


@pytest.mark.parametrize("skip_connection_action", [None, "concat", "add", "avg"])
def test_up_block_forward_with_skip_connection_action(skip_connection_action):
    """Test the UpBlock with various skip_connection_actions."""
    block = UpBlock(
        2,
        32,
        16,
        res_args={"res_groups": 8},
        skip_connection_action=skip_connection_action,
    )
    x = torch.randn(3, 32, 64, 64)
    h = torch.randn(3, 32, 64, 64)
    y = block(x, h)
    assert y.shape == (3, 16, 64, 64)


@pytest.mark.parametrize(
    "attention", ["self_attention", "conv_attention", "attention_gate"]
)
def test_up_block_forward_with_attention(attention):
    """Test the UpBlock (default)."""
    block = UpBlock(
        2,
        32,
        16,
        attention=attention,
        res_args={"res_groups": 8},
        attn_args={"groups": 8},
    )
    x = torch.randn(3, 32, 64, 64)
    h = torch.randn(3, 32, 64, 64)
    y = block(x, h)
    assert y.shape == (3, 16, 64, 64)
