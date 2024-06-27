"""Tests for the ResnetBlock1D, ResnetBlock2D, and ResnetBlock3D modules.

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
import torch
from chuchichaestli.models.resnet import ResidualBlock


@pytest.mark.parametrize(
    "dimensions, in_channels, out_channels, time_channels, res_groups, res_act_fn, res_droupout",
    [
        (1, 32, 32, 32, 16, "silu", 0.1),
        (2, 32, 32, 32, 16, "silu", 0.1),
        (3, 32, 32, 32, 16, "silu", 0.1),
        (1, 48, 32, 32, 16, "silu", 0.1),
        (2, 48, 32, 32, 16, "silu", 0.1),
        (3, 48, 32, 32, 16, "silu", 0.1),
        (1, 8, 16, 16, 4, "relu", 0.1),
        (2, 8, 16, 16, 4, "relu", 0.1),
        (3, 8, 16, 16, 4, "relu", 0.1),
        (1, 32, 64, 16, 16, "mish", 0.1),
        (2, 32, 64, 16, 16, "mish", 0.1),
        (3, 32, 64, 16, 16, "mish", 0.1),
    ],
)
def test_forward_resnet(
    dimensions,
    in_channels,
    out_channels,
    time_channels,
    res_groups,
    res_act_fn,
    res_droupout,
):
    """Test the forward method of the ResnetBlock1D module."""
    # Create dummy input tensor
    input_shape = (1, in_channels) + (32,) * dimensions
    input_tensor = torch.randn(input_shape)

    t_embedding = torch.randn((1, time_channels))

    resnet = ResidualBlock(
        dimensions,
        in_channels,
        out_channels,
        True,
        time_channels,
        res_groups,
        res_act_fn,
        res_droupout,
    )

    # Call the forward method
    output_tensor = resnet.forward(input_tensor, t_embedding)

    # Check the output tensor shape
    assert output_tensor.shape == (1, out_channels) + (32,) * dimensions


@pytest.mark.parametrize(
    "dimensions,input_channels, res_groups",
    [
        (1, 16, 17),
        (2, 16, 17),
        (3, 16, 17),
    ],
)
def test_forward_groups_not_divisible(dimensions, input_channels, res_groups):
    """Test the forward method of the ResnetBlock3D module."""
    # Call the forward method
    with pytest.raises(ValueError):
        ResidualBlock(dimensions, input_channels, 32, True, 32, res_groups=res_groups)
