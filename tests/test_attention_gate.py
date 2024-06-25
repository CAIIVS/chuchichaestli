"""Tests for the attention gate module.

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

import torch
import pytest

from chuchichaestli.models.attention.attention_gate import AttentionGate
from chuchichaestli.models.unet import UNet


@pytest.mark.parametrize(
    "dimension, feats", [(2, 64), (3, 64), (2, 32), (3, 16), (2, 128)]
)
def test_attention_gate_forward(dimension: int, feats: int):
    """Test the forward pass of the attention gate module."""
    # Create input tensors
    x_shape = (1, 32) + (64,) * dimension
    g_shape = (1, 64) + (feats,) * dimension
    x = torch.randn(x_shape)  # Example input tensor
    g = torch.randn(g_shape)  # Example guidance tensor

    # Create attention gate module
    attention_gate = AttentionGate(
        dimension=dimension, num_channels_g=64, num_channels_x=32, num_channels_inter=3
    )

    # Perform forward pass
    output = attention_gate.forward(x, g)

    # Check output shape
    assert output.shape == x.shape


def test_attention_gate_unet():
    """Test that the attention gate can be used in a UNet model."""
    dimensions = 2
    down_block_types = ["DownBlock"] * 4
    up_block_types = ["AttnGateUpBlock"] * 4
    block_out_channels = [64, 128, 256, 512]
    model = UNet(
        dimensions=dimensions,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channels=block_out_channels,
        resnet_groups=16,
    )
    input_dims = (1, 1) + (64,) * dimensions
    sample = torch.randn(*input_dims)  # Example input

    timestep = 0.5  # Example timestep
    output = model(sample, timestep)
    assert output.shape == input_dims  # Check output shape
