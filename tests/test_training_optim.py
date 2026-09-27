# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for describing an optimizer and its schedule."""

import pytest
import torch
from torch import nn

from chuchichaestli.training import (
    OPTIMIZER_MAP,
    SCHEDULER_MAP,
    OptimSpec,
    disjoint_params,
)


class Mine(torch.optim.SGD):
    """An optimizer no registry knows about."""


def model() -> nn.Module:
    """Return a model small enough to build an optimizer over."""
    return nn.Linear(4, 4)


@pytest.mark.parametrize("name", sorted(OPTIMIZER_MAP))
def test_every_named_constructor_builds_its_optimizer(name):
    """The shorthand must produce the thing it names."""
    spec = getattr(OptimSpec, name)()
    built = spec.build(model().parameters())
    assert isinstance(built, OPTIMIZER_MAP[name])
    assert built.param_groups[0]["lr"] == spec.lr


def test_the_arguments_reach_the_optimizer():
    """`kwargs` is the escape hatch for anything the shorthand omits."""
    built = OptimSpec.adam(lr=3e-4, betas=(0.5, 0.999)).build(model().parameters())
    assert built.param_groups[0]["betas"] == (0.5, 0.999)
    assert built.param_groups[0]["lr"] == 3e-4


def test_an_unknown_optimizer_names_the_alternatives():
    """The message follows the package's usual shape."""
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        OptimSpec(cls="adamax").build(model().parameters())


def test_an_optimizer_of_your_own_is_describable():
    """The old registry-only serializer raised on anything unregistered."""
    spec = OptimSpec(cls=f"{Mine.__module__}:{Mine.__qualname__}", lr=0.1)
    assert isinstance(spec.build(model().parameters()), Mine)


@pytest.mark.parametrize(
    ("builder", "args", "expected"),
    [
        ("with_cosine_schedule", (10,), "cosine"),
        ("with_cosine_warm_restarts_schedule", (5,), "cosine_warm_restarts"),
        ("with_step_schedule", (3,), "step"),
        ("with_multistep_schedule", ([2, 4],), "multistep"),
        ("with_exponential_schedule", (), "exponential"),
        ("with_linear_schedule", (), "linear"),
    ],
)
def test_every_schedule_builder_builds_its_scheduler(builder, args, expected):
    """A schedule rides on the spec rather than beside it."""
    spec = getattr(OptimSpec.adamw(), builder)(*args)
    assert spec.scheduler.cls == expected
    built = spec.scheduler.build(spec.build(model().parameters()))
    assert isinstance(built, SCHEDULER_MAP[expected])


def test_onecycle_defaults_to_stepping_per_step():
    """It spans steps, not epochs, so the default says so."""
    spec = OptimSpec.adamw().with_onecycle_schedule(max_lr=0.1, total_steps=100)
    assert spec.scheduler.interval == "step"
    assert isinstance(
        spec.scheduler.build(spec.build(model().parameters())),
        SCHEDULER_MAP["onecycle"],
    )


def test_a_plateau_schedule_carries_what_it_monitors():
    """The metric is the runtime's to supply, not the scheduler's argument."""
    spec = OptimSpec.adamw().with_plateau_schedule(monitor="probe/psnr", patience=3)
    assert spec.scheduler.monitor == "probe/psnr"
    assert "monitor" not in spec.scheduler.kwargs
    assert spec.scheduler.build(spec.build(model().parameters())).patience == 3


def test_adding_a_schedule_keeps_everything_else():
    """The old builder re-listed 17 fields by hand and dropped the rest."""
    spec = OptimSpec.adam(lr=7e-4, betas=(0.5, 0.9)).with_params(["gen", "gen/head"])
    scheduled = spec.with_cosine_schedule(t_max=10)
    assert scheduled.cls == spec.cls
    assert scheduled.lr == spec.lr
    assert scheduled.kwargs == spec.kwargs
    assert scheduled.params == spec.params


def test_a_spec_is_frozen_so_a_variant_is_a_new_one():
    """Two stages sharing a spec must not edit each other's."""
    spec = OptimSpec.adamw(lr=1e-3)
    assert spec.with_cosine_schedule(t_max=10) is not spec
    assert spec.scheduler is None
    with pytest.raises(AttributeError):
        spec.lr = 1.0


def test_only_lbfgs_asks_for_a_closure():
    """It re-evaluates the loss, so the update has to hand it one."""
    assert OptimSpec.lbfgs().requires_closure
    assert not OptimSpec.adamw().requires_closure


def test_a_parameter_belongs_to_one_group_only():
    """Two optimizers stepping one parameter would apply its update twice."""
    shared, extra = nn.Linear(2, 2), nn.Linear(3, 3)
    groups = disjoint_params(
        {
            "gen": list(shared.parameters()) + list(extra.parameters()),
            "disc": list(shared.parameters()),
        }
    )
    assert len(groups["gen"]) == 4
    assert groups["disc"] == []
    kept = [id(p) for params in groups.values() for p in params]
    assert len(kept) == len(set(kept))


def test_the_first_group_named_keeps_a_shared_parameter():
    """Priority is the caller's, so the first group named wins."""
    shared = nn.Linear(2, 2)
    groups = disjoint_params(
        {"disc": list(shared.parameters()), "gen": list(shared.parameters())}
    )
    assert len(groups["disc"]) == 2
    assert groups["gen"] == []


def test_a_named_constructor_takes_the_parameters_it_owns():
    """The obvious spelling must work; it went into kwargs and broke build()."""
    spec = OptimSpec.adam(lr=2e-4, params="disc")
    assert spec.params == "disc"
    assert "params" not in spec.kwargs
    assert spec.build(model().parameters()).param_groups[0]["lr"] == 2e-4


def test_a_plateau_schedule_must_name_what_it_reads():
    """Without a monitor it can never step, which is worse than refusing it."""
    with pytest.raises(ValueError, match="monitor="):
        OptimSpec.adamw().with_schedule("plateau")
