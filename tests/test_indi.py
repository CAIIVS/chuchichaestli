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
from chuchichaestli.diffusion.ddpm.indi import InDI


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
    "dimensions, batchsize, yield_intermediate",
    [
        (1, 1, False),
        (2, 1, False),
        (3, 1, False),
        (1, 4, False),
        (2, 4, False),
        (3, 4, False),
        (1, 1, True),
        (2, 1, True),
        (3, 1, True),
        (1, 4, True),
        (2, 4, True),
        (3, 4, True),
    ],
)
def test_denoise_step(dimensions, batchsize, yield_intermediate):
    """Test the denoise_step method of the InDI class."""
    # Create an instance of the InDI class
    indi = InDI(num_timesteps=10)

    # Create dummy input tensors
    input_shape = (batchsize, 16) + (32,) * dimensions
    c = torch.randn(input_shape)

    model = lambda x, t: x[:, :16, ...]  # noqa: E731

    # Call the denoise_step method
    output_generator = indi.generate(
        model, c, n=2, yield_intermediate=yield_intermediate
    )
    output = None
    for o in output_generator:
        output = o

    # Check the output shape
    assert output.shape == (2 * batchsize, 16) + (32,) * dimensions
