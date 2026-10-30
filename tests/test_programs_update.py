# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for applying one optimizer step over one update group or several."""

import pytest
import torch
from torch import nn
from torch.optim import SGD, LBFGS

from chuchichaestli.programs import (
    Alternating,
    C3liRunError,
    SwaWindow,
    CompositeObjective,
    Context,
    Criterion,
    Simultaneous,
    Step,
)
from chuchichaestli.training import Loss, Term, UpdatePolicy


def linear(weight: float = 1.0) -> nn.Module:
    """Build a one-parameter model with a known weight.

    Args:
        weight: Value the single parameter starts at.
    """
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(weight)
    return model


class Adversarial:
    """A two-group objective over the bound generator and discriminator."""

    def compute(self, batch, ctx) -> Loss:
        """Compute the loss for the group being applied.

        Args:
            batch: The batch to run through both models.
            ctx: Execution context, carrying the update group.
        """
        gen, disc = ctx["gen"], ctx["disc"]
        if ctx.group == "disc":
            return Loss(disc(gen(batch).detach()).mean())
        return Loss(disc(gen(batch)).mean())


def a_context(**bindings) -> Context:
    """Build a root context holding the given bindings.

    Args:
        bindings: Artifacts to bind.
    """
    return Context("t", bindings=bindings)


def gan_update(kind, gen, disc, frequencies=None):
    """Build a bound multi-group update over a generator and discriminator.

    Args:
        kind: `Alternating` or `Simultaneous`.
        gen: The generator.
        disc: The discriminator.
        frequencies: How often each group steps.
    """
    update = kind(("disc", "gen"), frequencies)
    update.bind(
        {
            "disc": SGD(disc.parameters(), lr=0.1),
            "gen": SGD(gen.parameters(), lr=0.1),
        }
    )
    return update


def test_a_frequency_must_divide_the_largest():
    """A ratio that cannot be expressed as a cadence is refused."""
    with pytest.raises(ValueError, match="must divide the largest"):
        Alternating(("disc", "gen"), {"disc": 3, "gen": 2})


def test_a_frequency_is_positive():
    """A group stepping zero times is refused."""
    with pytest.raises(ValueError, match="are positive"):
        Alternating(("disc", "gen"), {"disc": 0, "gen": 1})


def test_a_ratio_becomes_a_cadence():
    """`{"disc": 5, "gen": 1}` steps the discriminator five times as often."""
    update = Alternating(("disc", "gen"), {"disc": 5, "gen": 1})
    due = [update.groups(step) for step in range(10)]
    assert sum("disc" in groups for groups in due) == 10
    assert sum("gen" in groups for groups in due) == 2
    assert due[0] == ("disc", "gen")
    assert due[1] == ("disc",)


def test_binding_checks_the_groups_match():
    """An optimizer per group is required, and named when missing."""
    update = Alternating(("disc", "gen"))
    with pytest.raises(ValueError, match="one optimizer per group"):
        update.bind({"disc": SGD(linear().parameters(), lr=0.1)})


def test_a_multi_group_update_refuses_a_closure_optimizer():
    """L-BFGS cannot be driven alongside another group."""
    gen, disc = linear(), linear()
    update = Alternating(("disc", "gen"))
    with pytest.raises(ValueError, match="re-evaluate the loss"):
        update.bind(
            {
                "disc": LBFGS(disc.parameters()),
                "gen": SGD(gen.parameters(), lr=0.1),
            }
        )


def test_micro_batches_are_weighted_by_their_sample_count():
    """Splitting a batch unevenly reproduces the gradient of one pass."""
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = torch.zeros(5, 1)

    split = linear()
    update = Step()
    update.bind({None: SGD(split.parameters(), lr=0.0)})
    objective = Criterion(nn.MSELoss(), model=split)
    update.apply(objective, [(x[:3], y[:3]), (x[3:], y[3:])], a_context())

    whole = linear()
    nn.MSELoss()(whole(x), y).backward()
    assert torch.allclose(split.weight.grad, whole.weight.grad, atol=1e-6)


def test_the_even_share_shortcut_would_fail_that():
    """Weighting each micro-batch by `1/k` gives a different gradient."""
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = torch.zeros(5, 1)

    naive = linear()
    for part in (slice(0, 3), slice(3, 5)):
        (nn.MSELoss()(naive(x[part]), y[part]) / 2).backward()

    whole = linear()
    nn.MSELoss()(whole(x), y).backward()
    assert not torch.allclose(naive.weight.grad, whole.weight.grad, atol=1e-6)


def test_a_summed_objective_is_not_rescaled():
    """Under `reduction="sum"` the micro-batches simply add up."""
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = torch.zeros(5, 1)

    split = linear()
    update = Step(UpdatePolicy(reduction="sum"))
    update.bind({None: SGD(split.parameters(), lr=0.0)})
    objective = Criterion(nn.MSELoss(reduction="sum"), model=split)
    update.apply(objective, [(x[:3], y[:3]), (x[3:], y[3:])], a_context())

    whole = linear()
    nn.MSELoss(reduction="sum")(whole(x), y).backward()
    assert torch.allclose(split.weight.grad, whole.weight.grad, atol=1e-6)


def test_a_step_moves_the_parameters():
    """One apply steps the optimizer it was bound to."""
    model = linear()
    before = model.weight.detach().clone()
    update = Step()
    update.bind({None: SGD(model.parameters(), lr=0.1)})
    update.apply(
        Criterion(nn.MSELoss(), model=model),
        [(torch.ones(4, 1), torch.zeros(4, 1))],
        a_context(),
    )
    assert not torch.equal(model.weight, before)


def test_a_closure_optimizer_steps():
    """L-BFGS re-evaluates the loss through the policy's closure."""
    model = linear()
    before = model.weight.detach().clone()
    update = Step()
    update.bind({None: LBFGS(model.parameters(), lr=0.1)})
    loss = update.apply(
        Criterion(nn.MSELoss(), model=model),
        [(torch.ones(4, 1), torch.zeros(4, 1))],
        a_context(),
    )
    assert not torch.equal(model.weight, before)
    assert torch.isfinite(loss.total)


def test_a_closure_optimizer_cannot_accumulate():
    """More than one micro-batch is refused rather than silently dropped."""
    model = linear()
    update = Step()
    update.bind({None: LBFGS(model.parameters(), lr=0.1)})
    with pytest.raises(ValueError, match="cannot accumulate 2"):
        update.apply(
            Criterion(nn.MSELoss(), model=model),
            [(torch.ones(2, 1), torch.zeros(2, 1))] * 2,
            a_context(),
        )


def test_each_group_touches_only_its_own_parameters():
    """A step the generator sits out leaves its weights untouched."""
    gen, disc = linear(), linear()
    update = gan_update(Alternating, gen, disc, {"disc": 5, "gen": 1})
    ctx = a_context(gen=gen, disc=disc)
    ctx.progress = ctx.progress.next_step()
    before_gen = gen.weight.detach().clone()
    before_disc = disc.weight.detach().clone()

    update.apply(Adversarial(), [torch.ones(4, 1)], ctx)

    assert torch.equal(gen.weight, before_gen)
    assert not torch.equal(disc.weight, before_disc)


def test_the_parts_of_each_group_are_namespaced():
    """Both groups report under their own name."""
    gen, disc = linear(), linear()
    update = gan_update(Alternating, gen, disc)
    objective = CompositeObjective(
        [
            Term("adv", Adversarial(), groups=("gen",)),
            Term("d", Adversarial(), groups=("disc",)),
        ]
    )
    loss = update.apply(objective, [torch.ones(4, 1)], a_context(gen=gen, disc=disc))
    assert sorted(loss.parts) == ["disc/d", "gen/adv"]


def test_simultaneous_takes_its_gradients_before_any_step():
    """The generator's gradient is the one against the pre-step discriminator."""
    gen, disc = linear(2.0), linear(3.0)
    update = gan_update(Simultaneous, gen, disc)
    batch = torch.ones(4, 1)
    update.apply(Adversarial(), [batch], a_context(gen=gen, disc=disc))
    stepped = gen.weight.detach().clone()

    by_hand, pre_disc = linear(2.0), linear(3.0)
    pre_disc(by_hand(batch)).mean().backward()
    with torch.no_grad():
        by_hand.weight -= 0.1 * by_hand.weight.grad
    assert torch.allclose(stepped, by_hand.weight, atol=1e-6)


def test_alternating_and_simultaneous_are_distinguishable():
    """One sees the updated discriminator, the other does not."""
    batch = torch.ones(4, 1)
    outcome = {}
    for kind in (Alternating, Simultaneous):
        gen, disc = linear(2.0), linear(3.0)
        update = gan_update(kind, gen, disc)
        update.apply(Adversarial(), [batch], a_context(gen=gen, disc=disc))
        outcome[kind.__name__] = gen.weight.detach().clone()
    assert not torch.equal(outcome["Alternating"], outcome["Simultaneous"])


def test_the_optimizer_state_round_trips():
    """Momentum survives a save and restore through the update."""
    model = linear()
    update = Step()
    update.bind({None: SGD(model.parameters(), lr=0.1, momentum=0.9)})
    objective = Criterion(nn.MSELoss(), model=model)
    update.apply(objective, [(torch.ones(4, 1), torch.zeros(4, 1))], a_context())
    saved = update.state_dict()

    restored = linear()
    other = Step()
    other.bind({None: SGD(restored.parameters(), lr=0.1, momentum=0.9)})
    other.load_state_dict(saved)
    buffers = [s["momentum_buffer"] for s in other.optimizers[None].state.values()]
    assert buffers and all(b is not None for b in buffers)


def test_group_state_is_keyed_by_group():
    """Each group's optimizer and scheduler get their own key."""
    gen, disc = linear(), linear()
    update = gan_update(Alternating, gen, disc)
    assert sorted(update.state_dict()) == ["optim/disc", "optim/gen"]


def test_a_frequency_must_name_a_real_group():
    """A ratio for a group this update does not have is refused."""
    with pytest.raises(ValueError, match=r"\['enc'\] name no group"):
        Alternating(("disc", "gen"), {"enc": 2})


def test_a_group_cannot_repeat():
    """The same group twice would step twice per turn."""
    with pytest.raises(ValueError, match="own name"):
        Alternating(("gen", "gen"))


def test_groups_left_out_of_frequencies_step_every_time():
    """Only the ratios that differ need stating."""
    update = Alternating(("disc", "gen"), {"disc": 2})
    assert update.frequencies == {"disc": 2, "gen": 1}


def test_frequencies_may_be_given_positionally():
    """A tuple pairs with the groups in order, like a mapping by name."""
    positional = Alternating(("disc", "gen"), (5, 1))
    named = Alternating(("disc", "gen"), {"disc": 5, "gen": 1})
    assert positional.frequencies == named.frequencies


def test_positional_frequencies_must_cover_every_group():
    """One per group, or none; a partial tuple is ambiguous."""
    with pytest.raises(ValueError, match="1 frequencies for 2 groups"):
        Alternating(("disc", "gen"), (5,))
    with pytest.raises(ValueError, match="3 frequencies for 2 groups"):
        Alternating(("disc", "gen"), (5, 1, 1))


def test_a_mapping_alone_names_the_groups_and_their_ratios():
    """The keys order the groups; the values set how often each steps."""
    from_mapping = Alternating({"disc": 5, "gen": 1})
    explicit = Alternating(("disc", "gen"), {"disc": 5})
    assert from_mapping.groups() == explicit.groups() == ("disc", "gen")
    assert from_mapping.frequencies == explicit.frequencies


def test_frequencies_cannot_be_given_twice():
    """A mapping in both positions is a mistake, not a merge."""
    with pytest.raises(ValueError, match="given twice"):
        Alternating({"disc": 5}, {"gen": 1})


def test_positional_frequencies_need_groups_to_pair_with():
    """A bare tuple names nothing, so it cannot stand alone."""
    with pytest.raises(ValueError, match="name no groups"):
        Alternating(frequencies=(5, 1))


def test_clipping_is_skipped_when_no_threshold_is_set():
    """Without a threshold the gradients reach the optimizer untouched."""
    model = linear()
    update = Step()
    update.bind({None: SGD(model.parameters(), lr=0.0)})
    update.apply(
        Criterion(nn.MSELoss(), model=model),
        [(torch.full((4, 1), 3.0), torch.zeros(4, 1))],
        a_context(),
    )
    loose = model.weight.grad.clone()

    clipped = linear()
    tight = Step(UpdatePolicy(clip=1e-3))
    tight.bind({None: SGD(clipped.parameters(), lr=0.0)})
    tight.apply(
        Criterion(nn.MSELoss(), model=clipped),
        [(torch.full((4, 1), 3.0), torch.zeros(4, 1))],
        a_context(),
    )
    assert not torch.allclose(loose, clipped.weight.grad)
    assert clipped.weight.grad.norm() <= 1e-3 + 1e-9


def test_a_scaled_step_recovers_from_overflow_with_and_without_clipping():
    """The float16 path unscales only when clipping, and steps once it settles."""
    for policy in (UpdatePolicy(), UpdatePolicy(clip=1.0)):
        model = linear()
        before = model.weight.detach().clone()
        update = Step(policy, precision=torch.float16)
        update.bind({None: SGD(model.parameters(), lr=0.1)})
        scales = []
        for _ in range(20):
            update.apply(
                Criterion(nn.MSELoss(), model=model),
                [(torch.ones(4, 1), torch.zeros(4, 1))],
                a_context(),
            )
            scales.append(update._scalers[None].get_scale())
        assert scales[-1] < scales[0]
        assert not torch.equal(model.weight, before)


def test_the_window_opens_on_the_tail_it_was_given():
    """The fraction is of the run, so it ports between runs of any length."""
    window = SwaWindow(True, start=0.75)
    window.averages = {"model": None}
    assert [window.is_open(e, 8) for e in range(8)] == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]


def test_an_open_ended_run_has_no_tail_to_average():
    """There is no fraction of a run whose length is unknown."""
    window = SwaWindow(True, start=0.75)
    window.averages = {"model": None}
    assert not window.is_open(100, None)


def test_a_window_averaging_nothing_never_opens():
    """Its fraction has nothing to apply to."""
    assert not SwaWindow(None).is_open(7, 8)


def test_the_window_retires_only_the_groups_it_takes_over():
    """A group SWALR does not drive keeps whatever schedule it had."""
    window = SwaWindow(True, lr=0.01)
    window.schedulers = {"gen": object()}
    sweepwise = {"gen": object(), "disc": object()}
    window.hand_over(sweepwise)
    assert list(sweepwise) == ["disc"]


def test_a_window_without_a_rate_retires_nothing():
    """Nothing took the rate over, so the schedules must go on."""
    window = SwaWindow(True)
    sweepwise = {"gen": object(), "disc": object()}
    window.hand_over(sweepwise)
    assert sorted(sweepwise) == ["disc", "gen"]


def test_a_rate_without_averaging_is_refused():
    """It would schedule a window that never opens."""
    with pytest.raises(ValueError, match="was not asked to average"):
        SwaWindow(None, lr=0.01)


def test_the_start_of_the_window_is_a_fraction():
    """An epoch number would not port between runs of different lengths."""
    with pytest.raises(ValueError, match="fraction of the run"):
        SwaWindow(True, start=3)


def test_the_scaler_state_is_saved_beside_the_optimizer():
    """`load_state_dict` reads it back, so leaving it out breaks a resume."""
    import torch.nn as nn
    from chuchichaestli.programs.update import Step

    model = nn.Linear(2, 1)
    update = Step(precision=torch.float16)
    update.bind({None: torch.optim.SGD(model.parameters(), lr=0.1)})

    state = update.state_dict()
    assert any(key.startswith("scaler") for key in state)
    update.load_state_dict(state)


def test_a_group_cadence_keeps_its_phase_across_a_pass():
    """`step` restarts each pass, which would re-phase every frequency."""
    gen, disc = linear(), linear()
    update = gan_update(Alternating, gen, disc, {"disc": 3, "gen": 1})
    ctx = a_context(gen=gen, disc=disc)
    stepped: list[int] = []
    for _ in range(3):
        for _ in range(4):
            before = gen.weight.detach().clone()
            update.apply(Adversarial(), [torch.ones(4, 1)], ctx)
            if not torch.equal(gen.weight, before):
                stepped.append(ctx.progress.global_step)
            ctx.progress = ctx.progress.next_step()
        ctx.progress = ctx.progress.next_epoch()
    assert stepped == [0, 3, 6, 9]


def test_a_sat_out_group_stays_sat_out_after_a_pass():
    """A generator whose cadence has not come round must not step regardless."""
    gen, disc = linear(), linear()
    update = gan_update(Alternating, gen, disc, {"disc": 2, "gen": 1})
    ctx = a_context(gen=gen, disc=disc)
    ctx.progress = ctx.progress.next_step()
    ctx.progress = ctx.progress.next_epoch()
    assert (ctx.progress.step, ctx.progress.global_step) == (0, 1)
    before = gen.weight.detach().clone()
    update.apply(Adversarial(), [torch.ones(4, 1)], ctx)
    assert torch.equal(gen.weight, before)


def warmup_objective() -> CompositeObjective:
    """Build a GAN objective whose generator term waits two steps."""
    return CompositeObjective(
        [
            Term("gadv", Adversarial(), groups=("gen",), after=2),
            Term("dadv", Adversarial(), groups=("disc",)),
        ]
    )


@pytest.mark.parametrize("kind", [Alternating, Simultaneous])
def test_an_adversarial_warmup_runs_from_the_first_step(kind):
    """A group with no term on yet has no graph, so it must sit the step out."""
    gen, disc = linear(), linear()
    update = gan_update(kind, gen, disc)
    objective = warmup_objective()
    ctx = a_context(gen=gen, disc=disc)
    moved: list[bool] = []
    for _ in range(4):
        before = gen.weight.detach().clone()
        update.apply(objective, [torch.ones(4, 1)], ctx)
        moved.append(not torch.equal(gen.weight, before))
        ctx.progress = ctx.progress.next_step()
    assert moved == [False, False, True, True]


@pytest.mark.parametrize("kind", [Alternating, Simultaneous])
def test_a_sitting_group_reports_no_part_of_its_own(kind):
    """A group that did not run must not report a zero as if it had."""
    gen, disc = linear(), linear()
    loss = gan_update(kind, gen, disc).apply(
        warmup_objective(), [torch.ones(4, 1)], a_context(gen=gen, disc=disc)
    )
    assert sorted(loss.parts) == ["disc/dadv"]


def test_a_single_group_waits_for_its_only_term():
    """The default update has one group, which can be scheduled off too."""
    model = linear()
    update = Step()
    update.bind({None: SGD(model.parameters(), lr=0.1)})
    objective = CompositeObjective([Term("rec", Adversarial(), after=2)])
    ctx = a_context(gen=model, disc=model)
    before = model.weight.detach().clone()
    assert update.apply(objective, [torch.ones(4, 1)], ctx).parts == {}
    assert torch.equal(model.weight, before)


def test_a_closure_rebuilds_what_it_cached_on_each_evaluation():
    """L-BFGS re-evaluates the loss, and the first backward freed the graph."""
    from chuchichaestli.programs import Reconstruction

    model = nn.Linear(3, 3, bias=False)
    update = Step()
    update.bind({None: LBFGS(model.parameters())})
    objective = CompositeObjective([Term("rec", Reconstruction("l1"))])
    loss = update.apply(objective, [{"x": torch.ones(2, 3)}], a_context(model=model))
    assert sorted(loss.parts) == ["rec"]


def scaled_update(scale: float | None = None) -> Step:
    """Build a bound half-precision update, optionally at a given scale.

    Args:
        scale: Scale to leave its scaler holding, or `None` for the default.
    """
    update = Step(precision=torch.float16)
    update.bind({None: SGD(nn.Linear(2, 1).parameters(), lr=0.1)})
    if scale is not None:
        scaler = update._scalers[None]
        scaler.scale(torch.zeros(1))
        scaler.update(new_scale=scale)
    return update


def test_binding_builds_a_scaler_for_every_group():
    """A scaler built on first use would still be missing when a resume loads."""
    gen, disc = linear(), linear()
    update = gan_update(Alternating, gen, disc)
    assert sorted(map(str, update._scalers)) == ["disc", "gen"]


def test_a_resumed_scaler_keeps_the_scale_it_was_saved_at():
    """Resuming with the scale reset re-runs the overflow backoff."""
    saved = scaled_update(scale=1024.0).state_dict()
    resumed = scaled_update()
    resumed.load_state_dict(saved)
    assert resumed._scalers[None].get_scale() == 1024.0


def test_a_full_precision_resume_restores_nothing():
    """A disabled scaler saves nothing, which must not be restored as empty."""
    update = Step()
    update.bind({None: SGD(nn.Linear(2, 1).parameters(), lr=0.1)})
    assert not update._scalers[None].is_enabled()
    update.load_state_dict(update.state_dict())


def discriminator_step(kind, groups=("disc", "gen"), terms=("g", "d")):
    """Return how far one update moved the discriminator.

    Args:
        kind: `Alternating` or `Simultaneous`.
        groups: Update groups to step.
        terms: Which of the generator and discriminator terms to sum.
    """
    from chuchichaestli.programs.objective import DiscriminatorAdv, GeneratorAdv

    torch.manual_seed(0)
    gen, disc = nn.Linear(2, 2), nn.Linear(2, 1)
    made = {
        "g": Term("g", GeneratorAdv(inputs="x"), groups=("gen",)),
        "d": Term("d", DiscriminatorAdv(inputs="x"), groups=("disc",)),
    }
    update = kind(groups)
    update.bind(
        {g: SGD((disc if g == "disc" else gen).parameters(), lr=1.0) for g in groups}
    )
    before = disc.weight.detach().clone()
    update.apply(
        CompositeObjective([made[t] for t in terms]),
        [{"x": torch.randn(4, 2)}],
        a_context(model=gen, disc=disc),
    )
    return disc.weight.detach() - before


@pytest.mark.parametrize("kind", [Alternating, Simultaneous])
def test_the_discriminator_steps_on_its_own_loss_alone(kind):
    """The generator's loss reaches the discriminator, which must not step on it."""
    alone = discriminator_step(Alternating, ("disc",), ("d",))
    assert torch.allclose(discriminator_step(kind), alone)


def broken_step(
    policy: UpdatePolicy,
    model: nn.Module,
    optimizer: type = SGD,
    objective=None,
    **kwargs,
) -> Step:
    """Apply one step whose gradients are NaN.

    Args:
        policy: Says what a non-finite step does.
        model: Model stepped, whose input is NaN.
        optimizer: Class of the optimizer stepped.
        objective: Produces the loss; a criterion fed NaN when absent.
        kwargs: Passed on to `Step`.
    """
    update = Step(policy, **kwargs)
    built = optimizer(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(built, gamma=0.5)
    update.bind({None: built}, {None: scheduler})
    inputs = torch.ones(4, 1) if objective else torch.full((4, 1), float("nan"))
    update.apply(
        objective or Criterion(nn.MSELoss(), model=model),
        [(inputs, torch.zeros(4, 1))],
        a_context(),
    )
    return update


class NanAfterFirst:
    """A loss that turns NaN once the optimizer has evaluated it once."""

    def __init__(self, model: nn.Module):
        """Constructor.

        Args:
            model: Model the loss is computed through.
        """
        self.model = model
        self.calls = 0

    def compute(self, batch, ctx) -> Loss:
        """Return a finite loss on the first evaluation and NaN after.

        Args:
            batch: Inputs and targets.
            ctx: Execution context, unused.
        """
        self.calls += 1
        x, y = batch
        loss = nn.functional.mse_loss(self.model(x), y)
        return Loss(loss if self.calls == 1 else loss * float("nan"))


@pytest.mark.parametrize("optimizer", [SGD, LBFGS])
def test_a_non_finite_step_stops_the_run_by_default(optimizer):
    """Silently stepping would leave every later weight NaN."""
    with pytest.raises(C3liRunError, match="Non-finite gradients in t after 0 steps"):
        broken_step(UpdatePolicy(), linear(), optimizer)


@pytest.mark.parametrize("optimizer", [SGD, LBFGS])
def test_a_skipped_step_leaves_the_weights_and_the_schedule(optimizer):
    """The step did not happen, so neither the weights nor the rate move."""
    model = linear()
    with pytest.warns(UserWarning, match="the step is skipped"):
        update = broken_step(UpdatePolicy(nonfinite="skip"), model, optimizer)
    assert model.weight.item() == 1.0
    assert model.weight.grad is None
    assert update.schedulers[None].last_epoch == 0


def test_a_closure_step_skipped_partway_puts_everything_back():
    """L-BFGS moves the weights between evaluations; a skip undoes all of it."""
    model = linear()
    objective = NanAfterFirst(model)
    with pytest.warns(UserWarning, match="the step is skipped"):
        update = broken_step(
            UpdatePolicy(nonfinite="skip"), model, LBFGS, objective=objective
        )
    assert objective.calls > 1
    assert model.weight.item() == 1.0
    assert update.optimizers[None].state_dict()["state"] == {}


def test_an_ignored_step_steps_anyway():
    """The old behaviour, kept for whoever wants to spare the check its sync."""
    model = linear()
    broken_step(UpdatePolicy(nonfinite="ignore"), model)
    assert torch.isnan(model.weight).all()


@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step\\(\\)`")
def test_the_scaler_handles_a_non_finite_step_in_float16():
    """Overflow is routine while the scale settles, so it must not stop the run."""
    model = linear()
    broken_step(UpdatePolicy(), model, precision=torch.float16)
    assert model.weight.item() == 1.0


def test_a_non_finite_step_names_its_group():
    """With several models, the report says which one broke."""
    gen, disc = linear(), linear()
    update = Alternating(("disc", "gen"))
    update.bind(
        {
            "disc": SGD(disc.parameters(), lr=0.1),
            "gen": SGD(gen.parameters(), lr=0.1),
        }
    )
    with pytest.raises(C3liRunError, match="group 'disc'"):
        update.apply(
            Adversarial(),
            [torch.full((4, 1), float("nan"))],
            a_context(gen=gen, disc=disc),
        )
