"""Test the upsampling module.

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
from chuchichaestli.models.upsampling import Upsample


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_forward(dimensions):
    """Test the forward method of the upsample module."""
    # Create dummy input tensor
    input_shape = (1, 16) + (32,) * dimensions
    output_shape = (1, 16) + (64,) * dimensions
    input_tensor = torch.randn(input_shape)

    upsample = Upsample(dimensions=dimensions, num_channels=16)

    # Call the forward method
    output_tensor = upsample.forward(input_tensor, None)

    # Check the output tensor shape
    assert output_tensor.shape == output_shape


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_forward_with_large_batch_3d(dimensions):
    """Test the forward method of the upsample module with a large batch size."""
    # Create dummy input tensor
    input_shape = (128, 16) + (32,) * dimensions
    output_shape = (128, 16) + (64,) * dimensions
    input_tensor = torch.randn(input_shape)

    upsample = Upsample(dimensions=dimensions, num_channels=16)

    # Call the forward method
    output_tensor = upsample.forward(input_tensor, None)

    # Check the output tensor shape
    assert output_tensor.shape == output_shape
