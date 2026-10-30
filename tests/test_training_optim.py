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
    SchedulerSpec,
    SequentialScheduler,
    SequentialSchedulerSpec,
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
    """The metric is the runner's to supply, not the scheduler's argument."""
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


def rates(spec: OptimSpec, steps: int, value: float = 1.0) -> list[float]:
    """Return the learning rate before each of a number of schedule steps.

    Args:
        spec: Describes the optimizer and its schedule.
        steps: Schedule steps to take.
        value: What a plateau schedule reads at each step.
    """
    optimizer = spec.build(model().parameters())
    scheduler = spec.scheduler.build(optimizer)
    seen = [optimizer.param_groups[0]["lr"]]
    for _ in range(steps):
        optimizer.step()
        reads = getattr(scheduler, "reads_value", False)
        scheduler.step(value) if reads else scheduler.step()
        seen.append(round(optimizer.param_groups[0]["lr"], 6))
    return seen


@pytest.mark.parametrize(
    ("builder", "args"),
    [
        ("with_cosine_schedule", (5,)),
        ("with_exponential_schedule", (0.5,)),
        ("with_step_schedule", (2,)),
        ("with_multistep_schedule", ([1, 3],)),
        ("with_linear_schedule", ()),
        ("with_cosine_warm_restarts_schedule", (2,)),
    ],
)
@pytest.mark.parametrize(
    ("warmup", "kwargs"),
    [
        ("linear", {"start_factor": 0.1}),
        ("constant", {"factor": 0.1}),
        ("exponential", {"gamma": 2.0}),
        ("lambda", {"lr_lambda": lambda step: (step + 1) / 4}),
    ],
    ids=["linear", "constant", "exponential", "lambda"],
)
def test_any_warmup_runs_before_any_schedule_as_torch_would(
    builder, args, warmup, kwargs
):
    """The sequence is torch's `SequentialLR` wherever torch has one."""
    spec = getattr(OptimSpec.sgd(lr=1.0), builder)(*args).with_warmup(
        warmup, 3, **kwargs
    )
    optimizer = spec.build(model().parameters())
    first, second = spec.scheduler.schedulers
    expected = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [first.build(optimizer), second.build(optimizer)],
        milestones=[3],
    )
    seen = [optimizer.param_groups[0]["lr"]]
    for _ in range(10):
        optimizer.step()
        expected.step()
        seen.append(round(optimizer.param_groups[0]["lr"], 6))
    assert rates(spec, 10) == seen


def test_a_linear_warmup_climbs_to_the_rate_then_hands_over():
    """The common case, spelled out: ramp up, then decay from the full rate."""
    spec = (
        OptimSpec.sgd(lr=1.0)
        .with_exponential_schedule(0.5, interval="step")
        .with_warmup("linear", 4, start_factor=0.25)
    )
    assert rates(spec, 6) == [0.25, 0.4375, 0.625, 0.8125, 1.0, 0.5, 0.25]


def test_a_warmup_is_given_its_length_when_it_takes_one():
    """`total_iters` defaults to 5 in torch, which a warmup of 10 must not get."""
    spec = OptimSpec.sgd().with_cosine_schedule(10).with_warmup("linear", 10)
    assert spec.scheduler.schedulers[0].kwargs["total_iters"] == 10
    kept = OptimSpec.sgd().with_warmup("linear", 10, total_iters=4)
    assert kept.scheduler.schedulers[0].kwargs["total_iters"] == 4


def test_a_warmup_without_a_schedule_holds_the_rate_after_it():
    """With nothing after the warmup, the rate it climbed to is the rate."""
    spec = OptimSpec.sgd(lr=1.0).with_warmup("exponential", 2, gamma=2.0)
    assert rates(spec, 4)[0] == 1.0
    assert rates(spec, 4)[2:] == [1.0, 1.0, 1.0]


def test_a_warmup_steps_at_the_interval_of_the_schedule_after_it():
    """The two are one schedule, stepped by the runner at one cadence."""
    spec = OptimSpec.sgd().with_cosine_schedule(10, interval="step")
    assert spec.with_warmup("linear", 5).scheduler.interval == "step"
    mismatched = SchedulerSpec("linear", interval="epoch")
    with pytest.raises(ValueError, match="same interval"):
        spec.with_warmup(mismatched, 5)


def test_warmups_stack_ahead_of_one_another():
    """A second warmup goes first, and the milestones after it move along."""
    spec = (
        OptimSpec.sgd(lr=1.0)
        .with_exponential_schedule(0.5)
        .with_warmup("linear", 3)
        .with_warmup("constant", 2, factor=0.1)
    )
    assert [s.cls for s in spec.scheduler.schedulers] == [
        "constant",
        "linear",
        "exponential",
    ]
    assert spec.scheduler.milestones == (2, 5)


def test_a_warmup_runs_before_a_plateau_schedule():
    """Torch's sequence refuses a plateau schedule; this one hands it its value."""
    spec = (
        OptimSpec.sgd(lr=1.0)
        .with_plateau_schedule("probe/mse", factor=0.5, patience=0)
        .with_warmup("linear", 3, start_factor=0.1)
    )
    assert spec.scheduler.monitor == "probe/mse"
    assert rates(spec, 7) == [0.1, 0.4, 0.7, 1.0, 1.0, 0.5, 0.25, 0.125]


def test_a_plateau_schedule_in_charge_needs_its_value():
    """Stepping it without one would hold the rate for the rest of the run."""
    spec = OptimSpec.sgd().with_plateau_schedule("probe/mse").with_warmup("linear", 1)
    optimizer = spec.build(model().parameters())
    scheduler = spec.scheduler.build(optimizer)
    assert not scheduler.reads_value
    scheduler.step()
    assert scheduler.reads_value
    with pytest.raises(ValueError, match="plateau"):
        scheduler.step()


def test_a_sequence_resumes_where_it_left_off():
    """Its state carries every member's, so a resume lands mid-sequence."""
    spec = OptimSpec.sgd(lr=1.0).with_cosine_schedule(6).with_warmup("linear", 3)
    whole = rates(spec, 8)
    optimizer = spec.build(model().parameters())
    scheduler = spec.scheduler.build(optimizer)
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    saved = (optimizer.state_dict(), scheduler.state_dict())
    optimizer = spec.build(model().parameters())
    scheduler = spec.scheduler.build(optimizer)
    optimizer.load_state_dict(saved[0])
    scheduler.load_state_dict(saved[1])
    resumed = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        resumed.append(round(optimizer.param_groups[0]["lr"], 6))
    assert resumed == whole[5:]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"schedulers": [SchedulerSpec("linear")], "milestones": [3]}, "milestones"),
        (
            {
                "schedulers": [
                    SchedulerSpec("linear", interval="step"),
                    SchedulerSpec("cosine"),
                ],
                "milestones": [3],
            },
            "one interval",
        ),
        (
            {
                "schedulers": [
                    SchedulerSpec("plateau", monitor="a"),
                    SchedulerSpec("plateau", monitor="b"),
                ],
                "milestones": [3],
            },
            "one value",
        ),
    ],
    ids=["milestones", "intervals", "monitors"],
)
def test_a_sequence_that_cannot_run_as_one_is_refused(kwargs, match):
    """Each would fail later, mid-run, or step a schedule at the wrong time."""
    with pytest.raises(ValueError, match=match):
        SequentialSchedulerSpec(**kwargs)


@pytest.mark.parametrize("milestones", [[0], [3, 3], [4, 2]])
def test_milestones_must_be_positive_and_increasing(milestones):
    """A schedule taking over at or before the one ahead of it never runs."""
    optimizer = OptimSpec.sgd().build(model().parameters())
    schedulers = [
        torch.optim.lr_scheduler.ConstantLR(optimizer)
        for _ in range(len(milestones) + 1)
    ]
    with pytest.raises(ValueError, match="increasing"):
        SequentialScheduler(optimizer, schedulers, milestones)


def test_a_warmup_lasts_at_least_one_interval():
    """A warmup of no length is no warmup, and a milestone of 0 never runs."""
    with pytest.raises(ValueError, match="at least one"):
        OptimSpec.sgd().with_warmup("linear", 0)
