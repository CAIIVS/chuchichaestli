"""Tests for the InDI class.

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
from chuchichaestli.diffusion.indi import InDI


@pytest.mark.parametrize(
    "dimensions, batchsize",
    [
        (1, 1),
        (2, 1),
        (3, 1),
        (1, 4),
        (2, 4),
        (3, 4),
    ],
)
def test_noise_step(dimensions, batchsize):
    """Test the noise_step method of the InDI class."""
    # Create an instance of the InDI class
    indi = InDI(num_timesteps=10)

    # Create dummy input tensors
    input_shape = (batchsize, 16) + (32,) * dimensions
    x = torch.randn(input_shape)
    y = torch.randn(input_shape)

    # Call the noise_step method
    x_t, noise, timesteps = indi.noise_step(x, y)

    # Check the output shapes
    assert x_t.shape == input_shape
    assert noise.shape == input_shape
    assert timesteps.shape == (batchsize,)


@pytest.mark.parametrize(
    "dimensions, batchsize",
    [
        (1, 1),
        (2, 1),
        (3, 1),
        (1, 4),
        (2, 4),
        (3, 4),
    ],
)
def test_denoise_step(dimensions, batchsize):
    """Test the denoise_step method of the InDI class."""
    # Create an instance of the InDI class
    indi = InDI(num_timesteps=10)

    # Create dummy input tensors
    input_shape = (batchsize, 16) + (32,) * dimensions
    x_t = torch.randn(input_shape)
    t = 5
    model_output = torch.randn(input_shape)

    # Call the denoise_step method
    x_tmdelta = indi.denoise_step(x_t, t, model_output)

    # Check the output shape
    assert x_tmdelta.shape == input_shape
