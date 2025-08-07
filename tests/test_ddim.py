# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the DDIM module."""

import pytest
import torch
from chuchichaestli.diffusion.ddpm import DDIM


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
def test_generation(dimensions, batchsize, yield_intermediate):
    """Test the denoise_step method of the DDIM class."""
    # Create dummy input tensors
    ddpm = DDIM(num_timesteps=50, num_sample_steps=10)
    input_shape = (batchsize, 16) + (32,) * dimensions
    c = torch.randn(input_shape)

    model = lambda x, t: x[:, :16, ...]  # noqa: E731

    # Call the denoise_step method
    output_generator = ddpm.generate(
        model, c, n=2, yield_intermediate=yield_intermediate
    )

    output = None
    for o in output_generator:
        output = o

    # Check the output shape
    assert output.shape == (2 * batchsize, 16) + (32,) * dimensions


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
def test_generation_unconditional(dimensions, batchsize, yield_intermediate):
    """Test the denoise_step method of the DDIM class."""
    # Create dummy input tensors
    ddpm = DDIM(num_timesteps=50, num_sample_steps=10)
    input_shape = (batchsize, 16) + (32,) * dimensions

    def model(x, t):
        assert x.size(1) == 16
        return x[:, :16, ...]

    # Call the denoise_step method
    output_generator = ddpm.generate(
        model,
        condition=None,
        shape=input_shape[1:],
        n=2,
        yield_intermediate=yield_intermediate,
    )

    output = None
    for o in output_generator:
        output = o

    # Check the output shape
    assert output.shape == (2, 16) + (32,) * dimensions
