# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the BBDM module."""

import pytest
import torch
from chuchichaestli.diffusion.processes import BBDM


@pytest.mark.parametrize(
    "dimensions, batchsize, schedule",
    [
        (1, 1, "linear"),
        (2, 1, "linear"),
        (3, 1, "linear"),
        (1, 4, "linear"),
        (2, 4, "linear"),
        (3, 4, "linear"),
        (1, 1, "linear_scaled"),
        (1, 2, "squared"),
        (1, 2, "cosine"),
        (1, 2, "exponential"),
    ],
)
def test_noise_step(dimensions, batchsize, schedule):
    """Test the noise_step method of the DDPM class."""
    # Create dummy input tensor
    input_shape = (batchsize, 16) + (32,) * dimensions
    x_t = torch.randn(input_shape)
    c = torch.randn(input_shape)

    # Call the noise_step method
    bbdm = BBDM(num_timesteps=10, schedule=schedule)
    output = bbdm.noise_step(x_t, c)

    # Check the output shape
    assert output[0].shape == input_shape
    assert output[1].shape == input_shape
    assert output[2].shape[0] == batchsize


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
    """Test the denoise_step method of the DDPM class."""
    # Create dummy input tensors
    bbdm = BBDM(num_timesteps=10)
    input_shape = (batchsize, 16) + (32,) * dimensions
    c = torch.randn(input_shape)

    model = lambda x, t: x[:, :16, ...]  # noqa: E731

    # Call the denoise_step method
    output_generator = bbdm.generate(
        model, c, n=2, yield_intermediate=yield_intermediate
    )

    output = None
    for o in output_generator:
        output = o

    # Check the output shape
    assert output.shape == (2 * batchsize, 16) + (32,) * dimensions


def _rows_marked(batchsize, shape):
    """Return a batch whose every row is the constant of its index.

    Args:
        batchsize: Number of rows.
        shape: Shape of one row.
    """
    index = torch.arange(float(batchsize)).view(batchsize, *([1] * len(shape)))
    return index.expand(batchsize, *shape).contiguous()


def test_a_condition_is_expanded_with_the_sample_over_timesteps():
    """Its condition is required, so a batch mismatch is an outright error."""
    c = _rows_marked(4, (16, 32))
    x_0 = torch.randn(4, 16, 32)
    x_t, noise, timesteps = BBDM(num_timesteps=10).noise_step(
        x_0, c, timesteps=torch.tensor([0, 1, 2, 3])
    )
    assert x_t.shape == (4 * 4, 16, 32)
    assert noise.shape == (4 * 4, 16, 32)
    assert timesteps.tolist() == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
