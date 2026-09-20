# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Weighted sums of named loss terms."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
import torch
from torch import nn


__all__ = [
    "Loss",
    "Term",
    "AdaptiveWeight",
    "Objective",
]


@dataclass(frozen=True, slots=True)
class Loss:
    """The result of one objective evaluation.

    Attributes:
        total: The scalar that gets backpropagated.
        parts: Named components, for logging and for stopping conditions.
    """

    total: torch.Tensor
    parts: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return a short description of the loss."""
        total, parts = float(self.total), sorted(self.parts)
        return f"Loss({total=:.4g}, {parts=})"

    def as_floats(self) -> dict[str, float]:
        """Return the total and its parts as plain numbers.

        Keyed `"loss"` for the total, and by term name for the rest.
        """
        values = {"loss": float(self.total.detach())}
        values.update({name: float(v.detach()) for name, v in self.parts.items()})
        return values

    @classmethod
    def merge(cls, results: Mapping[str | None, Loss]) -> Loss:
        """Return one loss from several, keeping each source's parts apart.

        Args:
            results: What each source produced, keyed by a name that prefixes
                its parts, or `None` to leave them unprefixed.
        """
        total: torch.Tensor | None = None
        parts: dict[str, torch.Tensor] = {}
        for name, loss in results.items():
            total = loss.total if total is None else total + loss.total
            prefix = "" if name is None else f"{name}/"
            parts.update({f"{prefix}{key}": v for key, v in loss.parts.items()})
        return cls(total=torch.zeros(()) if total is None else total, parts=parts)


@dataclass(frozen=True, slots=True)
class AdaptiveWeight:
    """A weight balancing one term's gradient against another's.

    Attributes:
        ref: Name of the term this one is balanced against.
        layer: Parameter the gradient norms are taken with respect to.
        scale: Ratio the balanced gradient is aimed at; `1.0` matches the
            reference, `0.5` halves it.
        bounds: Lowest and highest ratio this may take, before scaling.
        eps: Added to the denominator.
    """

    ref: str
    layer: str
    scale: float = 1.0
    bounds: tuple[float, float] = (0.0, 1e4)
    eps: float = 1e-4

    def __call__(
        self,
        reference: torch.Tensor,
        balanced: torch.Tensor,
        layer: torch.Tensor,
    ) -> torch.Tensor:
        """Return the weight, detached so it contributes no gradient.

        Args:
            reference: Loss whose gradient sets the scale.
            balanced: Loss being weighted.
            layer: Parameter both gradients are taken with respect to.
        """
        ref_grad = torch.autograd.grad(reference, layer, retain_graph=True)
        balanced_grad = torch.autograd.grad(balanced, layer, retain_graph=True)
        weight = nn.utils.get_total_norm(ref_grad) / (
            nn.utils.get_total_norm(balanced_grad) + self.eps
        )
        return weight.clamp(*self.bounds).detach() * self.scale


@dataclass(frozen=True, slots=True)
class Term:
    """One named contribution to an objective.

    Attributes:
        name: Identifies the term, and keys it in `Loss.parts`.
        criterion: Produces this term's loss.
        weight: Scales the term in the total.
        groups: Update groups this term feeds, or all of them when empty.
        after: Steps before the term switches on.
        every: Steps between evaluations once it is on.
    """

    name: str
    criterion: Any = None
    weight: float | AdaptiveWeight = 1.0
    groups: tuple[str, ...] = ()
    after: int = 0
    every: int = 1

    def __post_init__(self) -> None:
        """Validate the name and the schedule.

        Raises:
            ValueError: If the name is unusable, or the schedule is not
                positive.
        """
        if not self.name or "." in self.name:
            raise ValueError(f"A term needs a name without dots, got {self.name!r}.")
        if self.after < 0:
            raise ValueError(
                f"A term switches on at step 0 or later, got {self.after!r}."
            )
        if self.every < 1:
            raise ValueError(f"A term needs a positive interval, got {self.every!r}.")

    def contributing(self, group: str | None = None, step: int = 0) -> bool:
        """Whether this term contributes at a step.

        Args:
            group: Update group being applied, or `None` for all of them.
            step: Steps taken so far.
        """
        if self.groups and group is not None and group not in self.groups:
            return False
        return step >= self.after and (step - self.after) % self.every == 0


class Objective(nn.Module):
    """A weighted sum of named terms.

    A module, so terms that carry weights of their own move and checkpoint
    with it.

    """

    def __init__(self, terms: Sequence[Term]):
        """Constructor.

        Args:
            terms: What the objective sums.

        Raises:
            ValueError: If two terms share a name.
        """
        super().__init__()
        names = [term.name for term in terms]
        if len(set(names)) != len(names):
            raise ValueError(f"Every term needs its own name, got {names}.")
        self._terms = tuple(terms)
        self.criteria = nn.ModuleDict(
            {t.name: t.criterion for t in terms if isinstance(t.criterion, nn.Module)}
        )

    def __repr__(self) -> str:
        """Return a short description of the objective."""
        terms = [term.name for term in self._terms]
        return f"{type(self).__name__}({terms=})"

    def terms(
        self, group: str | None = None, step: int | None = None
    ) -> tuple[Term, ...]:
        """Return the terms contributing at a step, or all of them.

        Args:
            group: Update group being applied, or `None` for all of them.
            step: Steps taken so far, or `None` to ignore the schedule.
        """
        if group is None and step is None:
            return self._terms
        return tuple(
            term for term in self._terms if term.contributing(group, step or 0)
        )

    def combine(
        self,
        values: Mapping[str, torch.Tensor],
        weights: Mapping[str, float | torch.Tensor] | None = None,
    ) -> Loss:
        """Return the weighted total and the parts it came from.

        Args:
            values: What each term evaluated to, keyed by term name.
            weights: Overrides the terms' own weights, for adaptive ones.

        Raises:
            ValueError: If a value belongs to no term of this objective.
        """
        by_name = {term.name: term for term in self._terms}
        unknown = sorted(set(values) - set(by_name))
        if unknown:
            raise ValueError(
                f"No term named {unknown} in this objective; it holds "
                f"{sorted(by_name)}."
            )
        weights = weights or {}
        total = None
        for name, value in values.items():
            weight = weights.get(name, by_name[name].weight)
            if isinstance(weight, AdaptiveWeight):
                raise ValueError(
                    f"Term {name!r} weighs adaptively, so its weight must be "
                    "computed and passed in `weights`."
                )
            scaled = value * weight
            total = scaled if total is None else total + scaled
        if total is None:
            total = torch.zeros(())
        return Loss(total=total, parts=dict(values))
