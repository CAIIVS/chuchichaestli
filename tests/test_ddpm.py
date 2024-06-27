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

import pytest
import torch
from chuchichaestli.diffusion import DDPM


@pytest.fixture
def ddpm():
    """Create an instance of the DDPM class."""
    return DDPM(num_timesteps=10)


def test_noise_step(ddpm):
    """Test the noise_step method of the DDPM class."""
    # Create dummy input tensor
    x_t = torch.randn(1, 16, 32)

    # Call the noise_step method
    output = ddpm.noise_step(x_t)

    # Check the output shape
    assert output[0].shape == (1, 16, 32)
    assert output[1].shape == (1, 16, 32)
    assert output[2].shape == (1,)


def test_denoise_step(ddpm):
    """Test the denoise_step method of the DDPM class."""
    # Create dummy input tensors
    x_t = torch.randn(1, 16, 32)
    t = 0
    model_output = torch.randn(1, 16, 32)

    # Call the denoise_step method
    output = ddpm.denoise_step(x_t, t, model_output)

    # Check the output shape
    assert output.shape == (1, 16, 32)
