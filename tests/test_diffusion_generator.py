# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests that a diffusion process draws only from the generator it was given."""

import pytest
import torch

from chuchichaestli.diffusion.ddpm import BBDM, CFGDDPM, DDPM, InDI, PriorGrad
from chuchichaestli.diffusion.distributions import (
    HalfNormalDistribution,
    NormalDistribution,
)

SHAPE = (4, 3, 8, 8)


def a_process(name: str, seed: int):
    """Build a process seeded through its own generator, and its extra arguments.

    Args:
        name: Which process to build.
        seed: Seed for the process's generator.
    """
    gen = torch.Generator().manual_seed(seed)
    condition = torch.full(SHAPE, 0.5)
    builders = {
        "ddpm": lambda: (DDPM(10, generator=gen), {}),
        "cfg_ddpm": lambda: (CFGDDPM(10, generator=gen), {"condition": condition}),
        "bbdm": lambda: (BBDM(10, generator=gen), {"condition": condition}),
        "indi": lambda: (InDI(num_timesteps=10, generator=gen), {"y": condition}),
        "prior_grad": lambda: (
            PriorGrad(torch.zeros(SHAPE[1:]), torch.ones(SHAPE[1:]), 10, generator=gen),
            {"condition": condition},
        ),
        "normal": lambda: (
            DDPM(10, generator=gen, noise_distribution=NormalDistribution(0.0, 1.0)),
            {},
        ),
        "half_normal": lambda: (
            DDPM(
                10,
                generator=gen,
                noise_distribution=HalfNormalDistribution(0.0, 1.0),
            ),
            {},
        ),
    }
    return builders[name]()


PROCESSES = ["ddpm", "cfg_ddpm", "bbdm", "indi", "prior_grad", "normal", "half_normal"]


@pytest.mark.parametrize("name", PROCESSES)
def test_a_generator_makes_the_noise_step_reproducible(name):
    """Training is not reproducible until every draw comes from the generator."""
    x = torch.randn(SHAPE)
    first, kwargs = a_process(name, 7)
    again, _ = a_process(name, 7)
    for one, other in zip(first.noise_step(x, **kwargs), again.noise_step(x, **kwargs)):
        assert torch.equal(one, other)


@pytest.mark.parametrize("name", PROCESSES)
def test_the_seed_of_the_generator_is_what_decides_the_draw(name):
    """Reproducing from any seed would mean the draw is not random at all."""
    x = torch.randn(SHAPE)
    first, kwargs = a_process(name, 7)
    other_seed, _ = a_process(name, 8)
    assert any(
        not torch.equal(one, other)
        for one, other in zip(
            first.noise_step(x, **kwargs), other_seed.noise_step(x, **kwargs)
        )
    )


@pytest.mark.parametrize("name", PROCESSES)
def test_the_global_rng_does_not_reach_a_seeded_process(name):
    """A draw that falls back to the global RNG moves when the global seed does."""
    x = torch.randn(SHAPE)
    torch.manual_seed(0)
    first, kwargs = a_process(name, 7)
    quiet = first.noise_step(x, **kwargs)
    torch.manual_seed(12345)
    again, _ = a_process(name, 7)
    disturbed = again.noise_step(x, **kwargs)
    for one, other in zip(quiet, disturbed):
        assert torch.equal(one, other)


@pytest.mark.parametrize(
    "adapter", [NormalDistribution(0.0, 1.0), HalfNormalDistribution(0.0, 1.0)]
)
def test_a_distribution_draws_from_the_generator_it_is_handed(adapter):
    """The process owns the generator, so the adapter takes it per call."""
    torch.manual_seed(0)
    first = adapter(SHAPE, generator=torch.Generator().manual_seed(3))
    torch.manual_seed(999)
    again = adapter(SHAPE, generator=torch.Generator().manual_seed(3))
    assert torch.equal(first, again)
    assert not torch.equal(
        first, adapter(SHAPE, generator=torch.Generator().manual_seed(4))
    )
