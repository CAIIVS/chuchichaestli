# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for objectives that read the execution context."""

import pytest
import torch
from torch import nn

from chuchichaestli.runtime import CompositeObjective, Context, Criterion
from chuchichaestli.training import AdaptiveWeight, Loss, Term


class Tiny(nn.Module):
    """A one-layer model."""

    def __init__(self):
        """Constructor."""
        super().__init__()
        self.head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model.

        Args:
            x: Input batch.
        """
        return self.head(x)


class Constant:
    """An objective returning a fixed multiple of the batch's mean."""

    def __init__(self, factor: float = 1.0, parts: bool = False):
        """Constructor.

        Args:
            factor: Scales the batch mean.
            parts: Whether to report a sub-part of its own.
        """
        self.factor = factor
        self.parts = parts
        self.calls = 0

    def compute(self, batch, ctx) -> Loss:
        """Compute the loss for one batch.

        Args:
            batch: The batch to reduce.
            ctx: Execution context for the stage.
        """
        self.calls += 1
        total = batch.mean() * self.factor
        return Loss(total, {"inner": total.detach()} if self.parts else {})


class Decoding:
    """An objective scoring a cached decode, counting how often it decodes."""

    def __init__(self, counter: list[int]):
        """Constructor.

        Args:
            counter: Incremented on every actual decode.
        """
        self.counter = counter
        self.seen: list[torch.Tensor] = []

    def decode(self, batch, ctx) -> torch.Tensor:
        """Decode the batch through the bound model.

        Args:
            batch: The batch to decode.
            ctx: Execution context for the stage.
        """
        self.counter.append(1)
        out = ctx["model"](batch)
        return out.detach() if ctx.group == "disc" else out

    def compute(self, batch, ctx) -> Loss:
        """Score the cached decode.

        Args:
            batch: The batch to decode.
            ctx: Execution context for the stage.
        """
        decoded = ctx.cache("recon", lambda: self.decode(batch, ctx))
        self.seen.append(decoded)
        return Loss(decoded.square().mean())


class Through:
    """An objective passing the batch through the bound model."""

    def __init__(self, factor: float = 1.0):
        """Constructor.

        Args:
            factor: Scales the model's mean output.
        """
        self.factor = factor

    def compute(self, batch, ctx) -> Loss:
        """Compute the loss for one batch.

        Args:
            batch: The batch to run through the model.
            ctx: Execution context for the stage.
        """
        return Loss(ctx["model"](batch).mean() * self.factor)


def a_context(**bindings) -> Context:
    """Build a root context holding the given bindings.

    Args:
        bindings: Artifacts to bind.
    """
    return Context("t", bindings=bindings)


def test_a_criterion_reads_a_pair_batch():
    """A tuple batch is unpacked as input and target."""
    objective = Criterion(nn.MSELoss(), model=Tiny())
    x = torch.ones(4, 2)
    loss = objective.compute((x, x), a_context())
    assert torch.allclose(loss.total, torch.zeros(()))


def test_a_criterion_reads_a_mapping_batch():
    """A mapping batch is read by the configured keys."""
    objective = Criterion(nn.MSELoss(), model=Tiny(), inputs="a", targets="b")
    batch = {"a": torch.ones(4, 2), "b": torch.zeros(4, 2)}
    loss = objective.compute(batch, a_context())
    assert torch.allclose(loss.total, torch.ones(()))


def test_a_criterion_resolves_its_model_by_name():
    """A string model is looked up in the context."""
    objective = Criterion(nn.MSELoss())
    x = torch.ones(4, 2)
    loss = objective.compute((x, x), a_context(model=Tiny()))
    assert torch.allclose(loss.total, torch.zeros(()))


def test_a_criterion_says_which_keys_a_batch_lacks():
    """A batch missing a configured key names it."""
    objective = Criterion(nn.MSELoss(), model=Tiny())
    with pytest.raises(ValueError, match=r"no \['y'\]"):
        objective.compute({"x": torch.ones(4, 2)}, a_context())


def test_a_criterion_rejects_a_batch_that_is_not_a_pair():
    """A batch of three items is refused."""
    objective = Criterion(nn.MSELoss(), model=Tiny())
    with pytest.raises(ValueError, match="got 3 items"):
        objective.compute((1, 2, 3), a_context())


def test_groups_route_terms_to_their_own_update():
    """Each group sees only the terms that feed it."""
    gen, disc = Constant(1.0), Constant(2.0)
    objective = CompositeObjective(
        [
            Term("rec", gen, groups=("gen",)),
            Term("dadv", disc, groups=("disc",)),
        ]
    )
    ctx = a_context()
    assert sorted(objective.compute(torch.ones(4), ctx.at_group("gen")).parts) == [
        "rec"
    ]
    assert sorted(objective.compute(torch.ones(4), ctx.at_group("disc")).parts) == [
        "dadv"
    ]
    assert (gen.calls, disc.calls) == (1, 1)


def test_an_ungrouped_term_feeds_every_group():
    """A term declaring no group contributes to all of them."""
    objective = CompositeObjective(
        [Term("shared", Constant()), Term("g", Constant(), groups=("gen",))]
    )
    ctx = a_context()
    assert sorted(objective.compute(torch.ones(4), ctx.at_group("disc")).parts) == [
        "shared"
    ]
    assert sorted(objective.compute(torch.ones(4), ctx.at_group("gen")).parts) == [
        "g",
        "shared",
    ]


def test_a_term_is_absent_from_the_payload_before_it_switches_on():
    """`after=` keeps a term out of the total and the parts alike."""
    late = Constant(5.0)
    objective = CompositeObjective(
        [Term("rec", Constant()), Term("gadv", late, after=3)]
    )
    ctx = a_context()
    early = objective.compute(torch.ones(4), ctx)
    assert "gadv" not in early.parts
    assert late.calls == 0
    ctx.progress = ctx.progress.next_step()
    ctx.progress = ctx.progress.next_step()
    ctx.progress = ctx.progress.next_step()
    on = objective.compute(torch.ones(4), ctx)
    assert "gadv" in on.parts
    assert torch.allclose(on.total, torch.tensor(6.0))


def test_a_terms_own_parts_are_namespaced():
    """A term reporting sub-parts keeps them under its own name."""
    objective = CompositeObjective([Term("rec", Constant(parts=True))])
    loss = objective.compute(torch.ones(4), a_context())
    assert sorted(loss.parts) == ["rec", "rec/inner"]


def test_the_cache_decodes_once_per_group_per_step():
    """Terms sharing a cached decode trigger exactly one forward each group."""
    counter: list[int] = []
    first, second = Decoding(counter), Decoding(counter)
    objective = CompositeObjective([Term("a", first), Term("b", second)])
    ctx = a_context(model=Tiny())
    objective.compute(torch.ones(4, 2), ctx.at_group("gen"))
    assert len(counter) == 1
    objective.compute(torch.ones(4, 2), ctx.at_group("disc"))
    assert len(counter) == 2


def test_the_disc_group_is_handed_a_detached_decode():
    """The cached value differs by group, so the graphs stay separate."""
    counter: list[int] = []
    term = Decoding(counter)
    objective = CompositeObjective([Term("a", term)])
    ctx = a_context(model=Tiny())
    objective.compute(torch.ones(4, 2), ctx.at_group("gen"))
    objective.compute(torch.ones(4, 2), ctx.at_group("disc"))
    assert term.seen[0].requires_grad
    assert not term.seen[1].requires_grad


def test_an_adaptive_weight_balances_against_its_reference():
    """The weight is the ratio of gradient norms at the named layer."""
    model = Tiny()
    objective = CompositeObjective(
        [
            Term("rec", Through(1.0)),
            Term(
                "gadv",
                Through(4.0),
                weight=AdaptiveWeight(ref="rec", layer="head.weight"),
            ),
        ]
    )
    loss = objective.compute(torch.ones(4, 2), a_context(model=model))
    assert torch.allclose(loss.total, torch.tensor(2.0), atol=1e-4)


def test_an_adaptive_weight_contributes_no_gradient_of_its_own():
    """Being detached, it scales the term without entering the graph."""
    model = Tiny()
    objective = CompositeObjective(
        [
            Term("rec", Through(1.0)),
            Term(
                "gadv",
                Through(4.0),
                weight=AdaptiveWeight(ref="rec", layer="head.weight"),
            ),
        ]
    )
    objective.compute(torch.ones(4, 2), a_context(model=model)).total.backward()

    plain = Tiny()
    (plain(torch.ones(4, 2)).mean() * 2.0).backward()
    assert torch.allclose(model.head.weight.grad, plain.head.weight.grad, atol=1e-5)


def test_an_adaptive_weight_needs_its_reference_to_contribute():
    """Balancing against a term of another group is refused."""
    objective = CompositeObjective(
        [
            Term("rec", Constant(), groups=("gen",)),
            Term(
                "dadv",
                Constant(),
                groups=("disc",),
                weight=AdaptiveWeight(ref="rec", layer="head.weight"),
            ),
        ]
    )
    with pytest.raises(ValueError, match="balanced against 'rec'"):
        objective.compute(torch.ones(4, 2), a_context(model=Tiny()).at_group("disc"))


def test_an_adaptive_weight_needs_a_real_parameter():
    """A name matching no parameter is reported by torch itself."""
    objective = CompositeObjective(
        [
            Term("rec", Through(1.0)),
            Term(
                "gadv",
                Through(4.0),
                weight=AdaptiveWeight(ref="rec", layer="nope.weight"),
            ),
        ]
    )
    with pytest.raises(AttributeError, match="has no attribute `nope`"):
        objective.compute(torch.ones(4, 2), a_context(model=Tiny()))


def test_a_term_without_compute_says_so():
    """A bare loss module in a term is refused with its type named."""
    objective = CompositeObjective([Term("rec", nn.MSELoss())])
    with pytest.raises(TypeError, match="no compute"):
        objective.compute(torch.ones(4), a_context())


def test_a_direct_objective_carrying_weights_moves_with_it():
    """A loss module registers, so it moves and checkpoints."""
    objective = Criterion(nn.Linear(2, 2))
    assert sorted(objective.state_dict()) == ["loss.bias", "loss.weight"]
