# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for applying one optimizer step over one update group or several."""

import pytest
import torch
from torch import nn
from torch.optim import SGD, LBFGS

from chuchichaestli.runtime import (
    Alternating,
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
    from chuchichaestli.runtime.context import Context
    from chuchichaestli.runtime.update import Step

    model = nn.Linear(2, 1)
    update = Step(precision=torch.float16)
    update.bind({None: torch.optim.SGD(model.parameters(), lr=0.1)})
    update._scaler(None, Context("program"))

    state = update.state_dict()
    assert any(key.startswith("scaler") for key in state)
    update.load_state_dict(state)
