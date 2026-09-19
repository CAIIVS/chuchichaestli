# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Objectives that read the execution context to know what to compute."""

from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
import torch
from torch import nn
from chuchichaestli.data.batch import BatchType, unpack_batch
from chuchichaestli.runtime.context import Context
from chuchichaestli.training.objective import (
    AdaptiveWeight,
    Loss,
    Objective,
    Term,
)


__all__ = ["Criterion", "CompositeObjective"]


class Criterion(nn.Module):
    """Score a model's output against a target taken from the batch.

    Attributes:
        loss: Compares the model's output to the target.
        model: Model, or the name of a binding holding one.
        inputs: Key the model's input is read from.
        targets: Key the target is read from.
    """

    def __init__(
        self,
        loss: Callable[..., torch.Tensor],
        model: nn.Module | str = "model",
        inputs: str = "x",
        targets: str = "y",
    ):
        """Constructor.

        Args:
            loss: Compares the model's output to the target.
            model: Model, or the name of a binding holding one.
            inputs: Key the model's input is read from, for mapping batches.
            targets: Key the target is read from, for mapping batches.
        """
        super().__init__()
        self.loss = loss
        self.model = model
        self.inputs = inputs
        self.targets = targets

    def __repr__(self) -> str:
        """Return a short description of the objective."""
        return f"{type(self).__name__}({type(self.loss).__name__})"

    def compute(self, batch: BatchType, ctx: Context) -> Loss:
        """Compute the loss for one batch.

        Args:
            batch: A mapping, or a pair of input and target.
            ctx: Execution context for the stage.
        """
        inputs, targets = unpack_batch(
            batch, self.inputs, self.targets, reader=type(self).__name__
        )
        return Loss(self.loss(ctx.resolve(self.model)(inputs), targets))


class CompositeObjective(Objective):
    """Terms evaluated for the group being applied, then summed.

    Attributes:
        model: Binding an adaptive weight resolves its layer against.
    """

    def __init__(self, terms: Sequence[Term], model: nn.Module | str = "model"):
        """Constructor.

        Args:
            terms: What the objective sums.
            model: Model, or the name of a binding holding one, that an
                adaptive weight takes its gradients against.
        """
        super().__init__(terms)
        self.model = model

    def compute(self, batch: BatchType, ctx: Context) -> Loss:
        """Sum the terms contributing to the current group and step.

        Args:
            batch: One micro-batch of the stage's data.
            ctx: Execution context, carrying the update group being applied.
        """
        terms = self.terms(ctx.group, ctx.progress.step)
        values: dict[str, torch.Tensor] = {}
        nested: dict[str, torch.Tensor] = {}
        for term in terms:
            loss = self._evaluate(term, batch, ctx)
            values[term.name] = loss.total
            nested.update({f"{term.name}/{k}": v for k, v in loss.parts.items()})
        combined = self.combine(values, self._weights(terms, values, ctx))
        if not nested:
            return combined
        return Loss(total=combined.total, parts={**combined.parts, **nested})

    def _evaluate(self, term: Term, batch: BatchType, ctx: Context) -> Loss:
        """Return what one term evaluates to.

        Args:
            term: The term to evaluate.
            batch: One micro-batch of the stage's data.
            ctx: Execution context for the stage.

        Raises:
            TypeError: If the term's objective cannot compute a loss.
        """
        compute = getattr(term.criterion, "compute", None)
        if compute is None:
            raise TypeError(
                f"Term {term.name!r} holds a {type(term.criterion).__name__}, "
                "which has no compute(batch, ctx)."
            )
        return compute(batch, ctx)

    def _weights(
        self,
        terms: Sequence[Term],
        values: Mapping[str, torch.Tensor],
        ctx: Context,
    ) -> dict[str, torch.Tensor]:
        """Compute the weight of every adaptively weighted term.

        Args:
            terms: The terms contributing to this step.
            values: What each of them evaluated to.
            ctx: Execution context for the stage.

        Raises:
            ValueError: If a term is balanced against one that is not
                contributing to the same group and step.
        """
        weights: dict[str, torch.Tensor] = {}
        model: nn.Module | None = None
        for term in terms:
            if not isinstance(term.weight, AdaptiveWeight):
                continue
            adaptive = term.weight
            if adaptive.ref not in values:
                raise ValueError(
                    f"Term {term.name!r} is balanced against {adaptive.ref!r}, "
                    f"which is not contributing here; {sorted(values)} are."
                )
            if model is None:
                model = ctx.resolve(self.model)
            weights[term.name] = adaptive(
                values[adaptive.ref],
                values[term.name],
                model.get_parameter(adaptive.layer),
            )
        return weights
