"""Tests for DDPM.

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

from functools import partial

import pytest
import torch
from chuchichaestli.diffusion import ILVR
from chuchichaestli.diffusion.ilvr import low_pass_filter


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
    """Test the noise_step method of the DDPM class."""
    # Create dummy input tensor
    match dimensions:
        case 1:
            lpf = partial(low_pass_filter, N=4, mode="linear")
        case 2:
            lpf = partial(low_pass_filter, N=4, mode="bilinear")
        case 3:
            lpf = partial(low_pass_filter, N=4, mode="trilinear")
    ilvr = ILVR(num_timesteps=10, low_pass_filter=lpf)
    input_shape = (batchsize, 16) + (32,) * dimensions
    x_t = torch.randn(input_shape)

    # Call the noise_step method
    output = ilvr.noise_step(x_t)

    # Check the output shape
    assert output[0].shape == input_shape
    assert output[1].shape == input_shape
    assert output[2].shape[0] == batchsize


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
    """Test the denoise_step method of the DDPM class."""
    # Create dummy input tensors
    match dimensions:
        case 1:
            lpf = partial(low_pass_filter, N=4, mode="linear")
        case 2:
            lpf = partial(low_pass_filter, N=4, mode="bilinear")
        case 3:
            lpf = partial(low_pass_filter, N=4, mode="trilinear")
    ilvr = ILVR(num_timesteps=10, low_pass_filter=lpf)
    input_shape = (batchsize, 16) + (32,) * dimensions
    x_t = torch.randn(input_shape)
    y = torch.randn(input_shape)
    t = 0
    model_output = torch.randn(input_shape)

    # Call the denoise_step method
    output = ilvr.denoise_step(x_t, y, t, model_output)

    # Check the output shape
    assert output.shape == input_shape
