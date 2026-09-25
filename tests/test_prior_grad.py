# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the PriorGrad module."""

import pytest
import torch
from chuchichaestli.diffusion import PriorGrad


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
    ddpm = PriorGrad(
        num_timesteps=10,
        schedule=schedule,
        mean=torch.randn((16,) + (32,) * dimensions),
        scale=torch.randn((16,) + (32,) * dimensions),
    )
    output = ddpm.noise_step(x_t, c)

    # Check the output shape
    assert output[0].shape == (batchsize, 32) + (32,) * dimensions  # x_t and c
    assert output[1].shape == (batchsize, 16) + (32,) * dimensions  # noise
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
    input_shape = (batchsize, 16) + (32,) * dimensions
    c = torch.randn(input_shape)

    model = lambda x, t: x[:, :16, ...]  # noqa: E731

    prior_grad = PriorGrad(
        num_timesteps=10,
        schedule="linear",
        mean=torch.randn((1, 16) + (32,) * dimensions),
        scale=torch.randn((1, 16) + (32,) * dimensions),
    )

    # Call the denoise_step method
    output_generator = prior_grad.generate(
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
    """It inherits DDPM's expansion and cats the condition the same way."""
    c = _rows_marked(4, (3, 32, 32))
    prior_grad = PriorGrad(
        torch.zeros(3, 32, 32), torch.ones(3, 32, 32), num_timesteps=10
    )
    out, _, timesteps = prior_grad.noise_step(
        torch.randn(4, 3, 32, 32), condition=c, timesteps=torch.tensor([0, 1, 2, 3])
    )
    assert out.shape[0] == 4 * 4
    assert torch.equal(out[:, :3], torch.cat([c] * 4, dim=0))
    assert timesteps.tolist() == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
