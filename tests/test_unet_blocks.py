"""Tests for the unet_blocks module.

This file is part of Chuchichaestli.

Chuchichaestli is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Chuchichaestli is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Chuchichaestli.  If not, see <http://www.gnu.org/licenses/>.

Developed by the Intelligent Vision Systems Group at ZHAW.
"""

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
    return sum(1 for block in blocks if hasattr(block, 'res_block'))

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
    assert encoder_res_blocks == expected_res_blocks  # Adjust the expected value as needed
    assert decoder_res_blocks == expected_res_blocks  # Adjust the expected value as needed
