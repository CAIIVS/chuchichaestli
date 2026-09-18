# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for turning gradients into an update, and averaging the result."""

import pytest
import torch
from torch import nn

from chuchichaestli.training import Ema, OptimSpec, Swa, UpdatePolicy


def model(fill: float = 1.0) -> nn.Module:
    """Return a model whose gradients are easy to reason about.

    Args:
        fill: Value every weight starts at.
    """
    net = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        net.weight.fill_(fill)
    return net


def with_gradients(net: nn.Module, scale: float = 1.0) -> nn.Module:
    """Give a model gradients of a known size.

    Args:
        net: Model to fill.
        scale: Value every gradient element takes.
    """
    net.weight.grad = torch.full_like(net.weight, scale)
    return net


def test_clipping_by_norm_bounds_the_whole_gradient():
    """Four elements of 1.0 make a norm of 2.0."""
    net = with_gradients(model())
    assert net.weight.grad.norm().item() == pytest.approx(2.0)
    UpdatePolicy(clip=0.5, clip_mode="norm").clip_grads(net.parameters())
    assert net.weight.grad.norm().item() == pytest.approx(0.5)


def test_clipping_by_value_bounds_each_element():
    """The norm may still exceed the threshold; each element may not."""
    net = with_gradients(model(), scale=3.0)
    UpdatePolicy(clip=0.5, clip_mode="value").clip_grads(net.parameters())
    assert net.weight.grad.abs().max().item() == pytest.approx(0.5)


def test_without_a_threshold_the_gradients_are_left_alone():
    """Clipping is opt-in."""
    net = with_gradients(model(), scale=7.0)
    UpdatePolicy().clip_grads(net.parameters())
    assert net.weight.grad.abs().max().item() == pytest.approx(7.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"clip_mode": "abs"}, "Unsupported clip mode"),
        ({"reduction": "median"}, "Unsupported reduction"),
        ({"clip": 0.0}, "must be positive"),
        ({"clip": -1.0}, "must be positive"),
    ],
)
def test_an_unusable_policy_is_refused(kwargs, match):
    """Each would fail later, or quietly not clip at all."""
    with pytest.raises(ValueError, match=match):
        UpdatePolicy(**kwargs)


def test_a_mean_reduced_loss_is_scaled_by_its_share():
    """Micro-batches are weighted by samples seen, not by 1/k."""
    policy = UpdatePolicy(reduction="mean")
    assert policy.share(torch.tensor(4.0), 0.25).item() == pytest.approx(1.0)


def test_a_sum_reduced_loss_already_carries_its_weight():
    """Scaling it would count the same samples twice."""
    policy = UpdatePolicy(reduction="sum")
    assert policy.share(torch.tensor(4.0), 0.25).item() == pytest.approx(4.0)


def test_shares_of_one_step_sum_to_the_whole_loss():
    """Two micro-batches of 3 and 1 samples must equal one batch of 4."""
    policy = UpdatePolicy(reduction="mean")
    parts = [
        policy.share(torch.tensor(2.0), 3 / 4),
        policy.share(torch.tensor(6.0), 1 / 4),
    ]
    assert sum(p.item() for p in parts) == pytest.approx(3.0)


def test_only_a_re_evaluating_optimizer_needs_a_closure():
    """L-BFGS recomputes the loss; the rest step on what is there."""
    net = model()
    assert OptimSpec.needs_closure(OptimSpec.lbfgs().build(net.parameters()))
    assert not OptimSpec.needs_closure(OptimSpec.adamw().build(net.parameters()))


def test_stepping_without_the_closure_it_needs_says_so():
    """Torch would raise something less helpful much further in."""
    net = with_gradients(model())
    optimizer = OptimSpec.lbfgs().build(net.parameters())
    with pytest.raises(ValueError, match="needs a closure"):
        UpdatePolicy().step(optimizer, net.parameters())


def test_the_closure_zeroes_computes_backwards_and_clips():
    """All four, in that order, every time the optimizer re-evaluates."""
    net = model()
    optimizer = OptimSpec.sgd(lr=0.1).build(net.parameters())
    policy = UpdatePolicy(clip=0.5, clip_mode="norm")
    calls = []

    def compute():
        calls.append(net.weight.grad is None)
        return (net.weight * 3.0).sum()

    closure = policy.closure_for(optimizer, net.parameters(), compute)
    loss = closure()
    assert calls == [True]
    assert loss.item() == pytest.approx(12.0)
    assert net.weight.grad.norm().item() == pytest.approx(0.5)


def test_a_re_evaluating_optimizer_steps_through_the_closure():
    """The whole point of the closure path."""
    net = model()
    optimizer = OptimSpec.lbfgs(lr=0.1).build(net.parameters())
    policy = UpdatePolicy()
    closure = policy.closure_for(
        optimizer, net.parameters(), lambda: (net.weight**2).sum()
    )
    before = net.weight.detach().clone()
    policy.step(optimizer, net.parameters(), closure)
    assert not torch.equal(net.weight, before)


def test_zeroing_drops_the_gradients():
    """Set to none, so the next step starts from nothing."""
    net = with_gradients(model())
    optimizer = OptimSpec.sgd().build(net.parameters())
    UpdatePolicy().zero_grad(optimizer)
    assert net.weight.grad is None


def test_the_average_seeds_from_the_first_update():
    """The random init must not be blended into the average."""
    net = model(fill=1.0)
    ema = Ema(net, decay=0.5)
    with torch.no_grad():
        net.weight.fill_(3.0)
    ema.update_parameters(net)
    assert ema.module.weight[0, 0].item() == pytest.approx(3.0)


def test_later_updates_blend_at_the_decay():
    """Half of the average, half of the live weights."""
    net = model(fill=1.0)
    ema = Ema(net, decay=0.5)
    for value in (3.0, 5.0):
        with torch.no_grad():
            net.weight.fill_(value)
        ema.update_parameters(net)
    assert ema.module.weight[0, 0].item() == pytest.approx(4.0)


def test_an_equal_average_is_available_too():
    """SWA weights every model it has seen the same."""
    net = model(fill=1.0)
    swa = Swa(net)
    for value in (3.0, 5.0):
        with torch.no_grad():
            net.weight.fill_(value)
        swa.update_parameters(net)
    assert swa.module.weight[0, 0].item() == pytest.approx(4.0)
    assert swa.n_averaged.item() == 2


def test_the_average_is_a_module_and_moves_with_one():
    """A plain dict of tensors stayed on the CPU and broke on the first update."""
    net = model()
    ema = Ema(net)
    assert isinstance(ema, nn.Module)
    assert ema.decay == 0.9999
    assert "module.weight" in ema.state_dict()
    assert "n_averaged" in ema.state_dict()


def test_the_average_round_trips_through_its_state():
    """It is checkpointed like any other module."""
    net = model(fill=2.0)
    ema = Ema(net, decay=0.9)
    ema.update_parameters(net)
    revived = Ema(model(fill=0.0), decay=0.9)
    revived.load_state_dict(ema.state_dict())
    assert torch.equal(revived.module.weight, ema.module.weight)


def test_an_unusable_decay_is_refused():
    """A decay outside [0, 1] diverges rather than averages."""
    for decay in (1.5, -0.1):
        with pytest.raises(ValueError, match="lies in"):
            Ema(model(), decay=decay)
