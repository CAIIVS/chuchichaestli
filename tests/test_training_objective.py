# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for weighted sums of named loss terms."""

import pytest
import torch
from torch import nn

from chuchichaestli.training import (
    AdaptiveWeight,
    Loss,
    Objective,
    Term,
)


def terms() -> list[Term]:
    """Return the shape a latent-diffusion objective takes."""
    return [
        Term("rec", weight=1.0, groups=("gen",)),
        Term("lpips", weight=0.1, groups=("gen",)),
        Term("gadv", weight=1.0, groups=("gen",), after=2),
        Term("dadv", weight=1.0, groups=("disc",)),
    ]


def test_the_terms_sum_with_their_weights():
    """One scalar to backpropagate, built from the parts."""
    loss = Objective(terms()).combine(
        {"rec": torch.tensor(2.0), "lpips": torch.tensor(10.0)}
    )
    assert loss.total.item() == pytest.approx(3.0)


def test_every_term_is_reported_separately():
    """Each part reaches the step payload without extra wiring."""
    loss = Objective(terms()).combine(
        {"rec": torch.tensor(2.0), "lpips": torch.tensor(10.0)}
    )
    assert sorted(loss.parts) == ["lpips", "rec"]
    assert loss.parts["lpips"].item() == pytest.approx(10.0)


def test_an_unweighted_objective_is_the_plain_sum():
    """The default weight is one."""
    objective = Objective([Term("a"), Term("b")])
    loss = objective.combine({"a": torch.tensor(1.5), "b": torch.tensor(2.5)})
    assert loss.total.item() == pytest.approx(4.0)


def test_a_term_switches_on_at_its_step():
    """`after=` is the usual adversarial warmup."""
    objective = Objective(terms())
    assert "gadv" not in [t.name for t in objective.terms("gen", step=1)]
    assert "gadv" in [t.name for t in objective.terms("gen", step=2)]


def test_a_term_that_is_off_is_absent_from_the_parts_too():
    """It must not appear in the payload as a zero."""
    objective = Objective(terms())
    active = {t.name: torch.tensor(1.0) for t in objective.terms("gen", step=0)}
    loss = objective.combine(active)
    assert "gadv" not in loss.parts


def test_an_interval_evaluates_a_term_every_nth_step():
    """Counted from the step it switched on."""
    term = Term("slow", after=2, every=3)
    assert [step for step in range(10) if term.contributing(step=step)] == [2, 5, 8]


def test_groups_route_terms_to_their_own_update():
    """The generator and discriminator steps see different subsets."""
    objective = Objective(terms())
    assert [t.name for t in objective.terms("gen", step=5)] == ["rec", "lpips", "gadv"]
    assert [t.name for t in objective.terms("disc", step=5)] == ["dadv"]


def test_a_term_without_groups_feeds_every_update():
    """Most stages have one group and should not have to name it."""
    term = Term("everywhere")
    assert (
        term.contributing("gen")
        and term.contributing("disc")
        and term.contributing(None)
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": ""}, "without dots"),
        ({"name": "a.b"}, "without dots"),
        ({"name": "a", "after": -1}, "step 0 or later"),
        ({"name": "a", "every": 0}, "positive interval"),
    ],
)
def test_an_unusable_term_is_refused(kwargs, match):
    """Each would misbehave quietly rather than loudly."""
    with pytest.raises(ValueError, match=match):
        Term(**kwargs)


def test_two_terms_cannot_share_a_name():
    """Their parts would collide in the payload."""
    with pytest.raises(ValueError, match="its own name"):
        Objective([Term("rec"), Term("rec")])


def test_a_value_belonging_to_no_term_is_refused():
    """A typo would otherwise be summed in silently."""
    with pytest.raises(ValueError, match="No term named"):
        Objective([Term("rec")]).combine({"recon": torch.tensor(1.0)})


def test_a_term_carrying_weights_moves_with_the_objective():
    """`LPIPSLoss` holds a VGG that must follow the run to its device."""
    inner = nn.Linear(2, 2)
    objective = Objective([Term("perceptual", criterion=inner), Term("plain")])
    assert list(objective.criteria) == ["perceptual"]
    assert any(p is inner.weight for p in objective.parameters())


def test_an_adaptive_weight_is_the_ratio_of_gradient_norms():
    """Three times the gradient means three times the weight."""
    net = nn.Linear(2, 2, bias=False)
    reference = (net.weight * 3.0).sum()
    balanced = (net.weight * 1.0).sum()
    weight = AdaptiveWeight(ref="rec", layer="weight")(reference, balanced, net.weight)
    assert weight.item() == pytest.approx(3.0, rel=1e-3)


def test_an_adaptive_weight_contributes_no_gradient_of_its_own():
    """It scales a term; it must not be trained through."""
    net = nn.Linear(2, 2, bias=False)
    weight = AdaptiveWeight(ref="a", layer="weight")(
        (net.weight * 2.0).sum(), (net.weight * 1.0).sum(), net.weight
    )
    assert not weight.requires_grad


def test_an_adaptive_weight_stays_within_its_bounds():
    """An unbounded ratio destabilises the very training it balances."""
    net = nn.Linear(2, 2, bias=False)
    weight = AdaptiveWeight(ref="a", layer="weight", bounds=(0.0, 1.5))(
        (net.weight * 9.0).sum(), (net.weight * 1.0).sum(), net.weight
    )
    assert weight.item() == pytest.approx(1.5)


def test_an_adaptive_weight_must_be_computed_before_combining():
    """`combine` cannot take the gradients itself; the caller has the graph."""
    objective = Objective([Term("gadv", weight=AdaptiveWeight(ref="rec", layer="out"))])
    with pytest.raises(ValueError, match="weighs adaptively"):
        objective.combine({"gadv": torch.tensor(1.0)})
    loss = objective.combine({"gadv": torch.tensor(2.0)}, weights={"gadv": 0.5})
    assert loss.total.item() == pytest.approx(1.0)


def test_a_loss_says_what_it_holds():
    """It is read in tracebacks and logs."""
    loss = Loss(torch.tensor(1.25), {"rec": torch.tensor(1.25)})
    assert "total=1.25" in repr(loss)
    assert "'rec'" in repr(loss)


def test_a_scale_aims_the_balance_below_the_reference():
    """LDM's `discriminator_weight`: balanced, then given a share of that."""
    net = nn.Linear(4, 4, bias=False)
    reference = (net.weight * 1.0).sum()
    balanced = (net.weight * 4.0).sum()
    full = AdaptiveWeight("rec", "out")(reference, balanced, net.weight)
    half = AdaptiveWeight("rec", "out", scale=0.5)(reference, balanced, net.weight)
    assert half.item() == pytest.approx(full.item() * 0.5)


def test_the_bounds_cap_the_ratio_and_the_scale_multiplies_it():
    """A ceiling and a target are different knobs."""
    net = nn.Linear(4, 4, bias=False)
    reference = (net.weight * 9.0).sum()
    balanced = (net.weight * 1.0).sum()
    capped = AdaptiveWeight("rec", "out", bounds=(0.0, 2.0))(
        reference, balanced, net.weight
    )
    assert capped.item() == pytest.approx(2.0)
    scaled = AdaptiveWeight("rec", "out", bounds=(0.0, 2.0), scale=0.5)(
        reference, balanced, net.weight
    )
    assert scaled.item() == pytest.approx(1.0)


def test_losses_merge_with_their_parts_kept_apart():
    """Merging namespaces each source's parts and sums the totals."""
    merged = Loss.merge(
        {
            "gen": Loss(torch.tensor(1.0), {"rec": torch.tensor(1.0)}),
            "disc": Loss(torch.tensor(2.0), {"adv": torch.tensor(2.0)}),
        }
    )
    assert float(merged.total) == 3.0
    assert sorted(merged.parts) == ["disc/adv", "gen/rec"]


def test_merging_one_unnamed_loss_leaves_its_parts_alone():
    """A `None` key means the parts keep their own names."""
    merged = Loss.merge({None: Loss(torch.tensor(1.0), {"rec": torch.tensor(1.0)})})
    assert sorted(merged.parts) == ["rec"]


def test_merging_nothing_gives_a_zero_loss():
    """An empty merge is a zero total, not a crash."""
    assert float(Loss.merge({}).total) == 0.0
