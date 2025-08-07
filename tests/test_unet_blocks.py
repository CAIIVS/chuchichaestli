# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the U-Net blocks module."""

import pytest
from chuchichaestli.models.unet import UNet


@pytest.fixture
def net_conf():
    return {
        "dimensions": 3,
        "in_channels": 1,
        "n_channels": 8,
        "out_channels": 1,
        "down_block_types": ["DownBlock", "DownBlock", "DownBlock"],
        "mid_block_type": "MidBlock",
        "up_block_types": ["UpBlock", "UpBlock", "UpBlock"],
        "block_out_channel_mults": [2, 2, 2],
        "num_layers_per_block": 1,
        "skip_connection_action": "concat",
        "skip_connection_between_levels": True,
    }


def count_res_blocks(blocks):
    return sum(1 for block in blocks if hasattr(block, "res_block"))


@pytest.mark.filterwarnings("ignore:Number of channels")
@pytest.mark.parametrize("num_layers_per_block, expected_res_blocks", [(1, 3), (2, 6)])
def test_res_blocks(net_conf, num_layers_per_block, expected_res_blocks):
    """Test the number of residual blocks in the UNet model."""
    net_conf["num_layers_per_block"] = num_layers_per_block
    model = UNet(**net_conf)

    encoder_res_blocks = count_res_blocks(model.down_blocks)
    decoder_res_blocks = count_res_blocks(model.up_blocks)

    print(f"Number of res_blocks in encoder: {encoder_res_blocks}")
    print(f"Number of res_blocks in decoder: {decoder_res_blocks}")

    # Add your assertions here
    assert (
        encoder_res_blocks == expected_res_blocks
    )  # Adjust the expected value as needed
    assert (
        decoder_res_blocks == expected_res_blocks
    )  # Adjust the expected value as needed
