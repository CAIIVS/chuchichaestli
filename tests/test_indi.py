# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the InDI module."""

import pytest
import torch
from chuchichaestli.diffusion.processes.indi import InDI


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


def test_noise_step_takes_the_condition_alias():
    """`condition=` names what `generate` and the other processes call it."""
    indi = InDI(num_timesteps=10)
    x, y = torch.randn(4, 16, 32), torch.randn(4, 16, 32)
    torch.manual_seed(0)
    by_name = indi.noise_step(x, condition=y)
    torch.manual_seed(0)
    positionally = indi.noise_step(x, y)
    for one, other in zip(by_name, positionally):
        assert torch.equal(one, other)


def test_noise_step_refuses_the_sample_under_both_names():
    """Two low quality samples are a caller mistake, not a merge."""
    indi = InDI(num_timesteps=10)
    x, y = torch.randn(4, 16, 32), torch.randn(4, 16, 32)
    with pytest.raises(ValueError, match="twice, as y= and as condition="):
        indi.noise_step(x, y, condition=y)


def test_noise_step_needs_the_sample_under_one_name():
    """Interpolating towards nothing would silently return the input."""
    indi = InDI(num_timesteps=10)
    with pytest.raises(ValueError, match="needs the low quality sample"):
        indi.noise_step(torch.randn(4, 16, 32))


def _rows_marked(batchsize, shape):
    """Return a batch whose every row is the constant of its index.

    Args:
        batchsize: Number of rows.
        shape: Shape of one row.
    """
    index = torch.arange(float(batchsize)).view(batchsize, *([1] * len(shape)))
    return index.expand(batchsize, *shape).contiguous()


def test_the_low_quality_sample_is_expanded_with_the_sample_over_timesteps():
    """At t=0 the interpolation returns x, at the last step it reaches y."""
    y = _rows_marked(4, (16, 32))
    x = torch.zeros(4, 16, 32)
    indi = InDI(num_timesteps=4)
    x_t, _, timesteps = indi.noise_step(x, y, timesteps=torch.tensor([0, 1, 2, 3]))
    assert x_t.shape == (4 * 4, 16, 32)
    assert torch.equal(x_t[:4], x)
    # x is zero, so each later row is its own t scaled by its own y
    for row in range(4, 16):
        assert torch.allclose(x_t[row], timesteps[row] * y[row % 4])
