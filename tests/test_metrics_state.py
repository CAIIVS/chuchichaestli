# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for checkpointing what a metric has accumulated."""

import pytest
import torch

from chuchichaestli.metrics import MSE, PSNR, SSIM
from chuchichaestli.runtime import Stateful


def pair(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a prediction and a target.

    Args:
        seed: Seeds the draw.
    """
    generator = torch.Generator().manual_seed(seed)
    shape = (2, 1, 16, 16)
    return (
        torch.rand(shape, generator=generator),
        torch.rand(shape, generator=generator),
    )


@pytest.mark.parametrize("metric_cls", [MSE, PSNR, SSIM])
def test_a_metric_is_stateful(metric_cls):
    """Every metric satisfies the protocol the runtime checkpoints through.

    Args:
        metric_cls: The metric under test.
    """
    assert isinstance(metric_cls(), Stateful)


@pytest.mark.parametrize("metric_cls", [MSE, PSNR, SSIM])
def test_a_partial_accumulation_round_trips(metric_cls):
    """A metric resumed mid-pass reports what the uninterrupted one would.

    Args:
        metric_cls: The metric under test.
    """
    first, second = pair(0), pair(1)

    uninterrupted = metric_cls()
    uninterrupted.update(*first)
    uninterrupted.update(*second)

    interrupted = metric_cls()
    interrupted.update(*first)
    resumed = metric_cls()
    resumed.load_state_dict(interrupted.state_dict())
    resumed.update(*second)

    assert resumed.compute() == pytest.approx(uninterrupted.compute())


def test_state_holds_accumulators_and_not_configuration():
    """Tensor attributes are state; the kernel settings are not."""
    state = SSIM().state_dict()
    assert "aggregate" in state and "n_observations" in state
    assert not any(name.startswith("kernel") for name in state)
    assert "device" not in state


def test_state_from_another_metric_is_refused():
    """A key the metric does not hold means the state came from elsewhere."""
    with pytest.raises(KeyError, match="holds no"):
        PSNR().load_state_dict({"not_a_field": torch.tensor(1.0)})


def test_a_reset_metric_forgets_what_it_accumulated():
    """Reset returns a metric to the state a fresh one has."""
    metric = PSNR()
    metric.update(*pair())
    metric.reset()
    fresh = PSNR()
    assert all(
        torch.equal(value, fresh.state_dict()[name])
        for name, value in metric.state_dict().items()
    )


def test_every_accumulator_follows_the_metric_to_a_device():
    """A state tensor left behind fails the first update on the new device."""
    from chuchichaestli.metrics import FID

    metric = FID()
    metric.to(torch.device("meta"))
    left = [
        name
        for name, value in vars(metric).items()
        if isinstance(value, torch.Tensor) and value.device.type != "meta"
    ]
    assert left == []
