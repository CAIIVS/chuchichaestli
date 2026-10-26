# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests that ShiftDDPM noises and samples as the ShiftDDPMs paper defines."""

import pytest
import torch
from torch import nn

from chuchichaestli.diffusion.processes.shift_ddpm import ShiftDDPM

SHAPE = (4, 3, 8, 8)
STEPS = 10


def shifted(seed: int = 0, **kwargs) -> ShiftDDPM:
    """Build a process drawing from its own generator.

    Args:
        seed: Seed for the process's generator.
        kwargs: Passed on to `ShiftDDPM`.
    """
    return ShiftDDPM(STEPS, generator=torch.Generator().manual_seed(seed), **kwargs)


def test_the_forward_marginal_is_the_papers():
    """The noised sample follows eq. 6; the target is the noise with the shift."""
    x_0, condition = torch.randn(SHAPE), torch.randn(SHAPE)
    process = shifted(3)
    x_t, target, timesteps = process.noise_step(x_0, condition)
    replay = torch.Generator().manual_seed(3)
    t = torch.randint(0, STEPS, (SHAPE[0],), generator=replay)
    eps = torch.randn(SHAPE, generator=replay)
    a = process.sqrt_alpha_cumprod[t].reshape(-1, 1, 1, 1)
    b = process.sqrt_1m_alpha_cumprod[t].reshape(-1, 1, 1, 1)
    k = process.k[t].reshape(-1, 1, 1, 1)
    assert torch.equal(timesteps, t)
    assert torch.allclose(x_t, a * x_0 + k * condition + b * eps)
    assert torch.allclose(target, (x_t - a * x_0) / b)


def test_an_oracle_network_recovers_the_sample():
    """With the exact target predicted at every step, the last step lands on x_0."""
    x_0, condition = torch.randn(SHAPE), torch.randn(SHAPE)
    process = shifted()

    def oracle(x_t, t):
        a, b = process.sqrt_alpha_cumprod[t], process.sqrt_1m_alpha_cumprod[t]
        return (x_t - a * x_0) / b

    *_, sample = process.generate(oracle, condition=condition)
    assert torch.allclose(sample, x_0, atol=1e-4)


def test_every_step_lands_on_the_papers_posterior():
    """Each step is the posterior of eq. 12 written through x_0, plus the same noise."""
    x_0, condition = torch.randn(SHAPE), torch.randn(SHAPE)
    process = shifted(4)

    def oracle(x_t, t):
        a, b = process.sqrt_alpha_cumprod[t], process.sqrt_1m_alpha_cumprod[t]
        return (x_t - a * x_0) / b

    steps = list(process.generate(oracle, condition=condition, yield_intermediate=True))
    replay = torch.Generator().manual_seed(4)
    x_t = process.k[-1] * condition + torch.randn(SHAPE, generator=replay)
    expected = []
    for t in reversed(range(STEPS)):
        ab, beta = process.alpha_cumprod[t], process.beta[t]
        ab_prev = process.alpha_cumprod[t - 1] if t > 0 else torch.tensor(1.0)
        k_prev = process.k[t - 1] if t > 0 else torch.tensor(0.0)
        to_x_0 = ab_prev.sqrt() * beta / (1 - ab)
        to_x_t = process.alpha[t].sqrt() * (1 - ab_prev) / (1 - ab)
        x_t = (
            to_x_0 * x_0
            + to_x_t * (x_t - process.k[t] * condition)
            + k_prev * condition
        )
        if t > 0:
            sigma = ((1 - ab_prev) / (1 - ab) * beta).sqrt()
            x_t = x_t + sigma * torch.randn(SHAPE, generator=replay)
        expected.append(x_t)
    assert len(steps) == len(expected)
    # the last step divides by √β_0, which magnifies float32 rounding a hundredfold
    for one, other in zip(steps, expected):
        assert torch.allclose(one, other, atol=2e-3)


def test_the_network_sees_the_noised_sample_only():
    """Training feeds it x_t alone, so sampling must too."""
    seen = []

    def model(x_t, t):
        seen.append(x_t.shape)
        return torch.zeros_like(x_t)

    x_t, _, _ = shifted().noise_step(torch.randn(SHAPE), torch.randn(SHAPE))
    list(shifted().generate(model, condition=torch.randn(SHAPE)))
    assert x_t.shape == SHAPE
    assert set(seen) == {torch.Size(SHAPE)}


def test_n_samples_per_condition_and_every_step_yielded():
    """The last yield holds n samples per condition; asked to, every step is yielded."""

    def zero(x_t, t):
        return torch.zeros_like(x_t)

    steps = list(
        shifted().generate(
            zero, condition=torch.randn(SHAPE), n=2, yield_intermediate=True
        )
    )
    assert len(steps) == STEPS
    assert steps[-1].shape == (2 * SHAPE[0], *SHAPE[1:])


def test_a_condition_is_required():
    """There is nothing to shift by without one, in training or in sampling."""
    with pytest.raises(ValueError, match="needs one"):
        shifted().noise_step(torch.randn(SHAPE))
    with pytest.raises(ValueError, match="needs one"):
        next(shifted().generate(lambda x_t, t: x_t, shape=SHAPE[1:]))


def test_without_a_predictor_the_condition_is_the_shift():
    """E defaults to the identity."""
    x_0, condition = torch.randn(SHAPE), torch.randn(SHAPE)
    plain = shifted(5).noise_step(x_0, condition)
    identity = shifted(5, E=nn.Identity()).noise_step(x_0, condition)
    for one, other in zip(plain, identity):
        assert torch.equal(one, other)


def test_sampling_repeats_for_the_same_generator_seed():
    """Every draw comes from the process's generator."""
    condition = torch.randn(SHAPE)

    def zero(x_t, t):
        return torch.zeros_like(x_t)

    torch.manual_seed(0)
    *_, first = shifted(7).generate(zero, condition=condition)
    torch.manual_seed(1)
    *_, again = shifted(7).generate(zero, condition=condition)
    *_, other = shifted(8).generate(zero, condition=condition)
    assert torch.equal(first, again)
    assert not torch.equal(first, other)
