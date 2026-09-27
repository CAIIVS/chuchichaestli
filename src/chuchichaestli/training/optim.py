# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Optimizer and learning-rate schedule specifications."""

from __future__ import annotations
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT
from torch.optim.lr_scheduler import LRScheduler
from chuchichaestli.models.spec import ModelSpec
from chuchichaestli.utils.registry import require


__all__ = [
    "OptimizerTypes",
    "SchedulerTypes",
    "IntervalTypes",
    "OPTIMIZER_MAP",
    "SCHEDULER_MAP",
    "CLOSURE_OPTIMIZERS",
    "SchedulerSpec",
    "OptimSpec",
    "disjoint_params",
]


OptimizerTypes = Literal["adam", "adamw", "sgd", "rmsprop", "adagrad", "lbfgs"]
SchedulerTypes = Literal[
    "cosine",
    "cosine_warm_restarts",
    "exponential",
    "linear",
    "multistep",
    "onecycle",
    "plateau",
    "step",
]
IntervalTypes = Literal["step", "epoch"]

OPTIMIZER_MAP: dict[str, type[Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
    "lbfgs": torch.optim.LBFGS,
}

SCHEDULER_MAP: dict[str, type[LRScheduler]] = {
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "cosine_warm_restarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
    "linear": torch.optim.lr_scheduler.LinearLR,
    "multistep": torch.optim.lr_scheduler.MultiStepLR,
    "onecycle": torch.optim.lr_scheduler.OneCycleLR,
    "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "step": torch.optim.lr_scheduler.StepLR,
}

CLOSURE_OPTIMIZERS: frozenset[str] = frozenset({"lbfgs"})


def disjoint_params(
    groups: Mapping[str, Iterable[nn.Parameter]],
) -> dict[str, list[nn.Parameter]]:
    """Return the groups with each parameter kept only in the first that claims it.

    Two optimizers stepping the same parameter would apply its update twice.

    Args:
        groups: Parameters each update group selected, in priority order.
    """
    seen: set[int] = set()
    disjoint: dict[str, list[nn.Parameter]] = {}
    for name, params in groups.items():
        disjoint[name] = [
            p for p in params if id(p) not in seen and not seen.add(id(p))
        ]
    return disjoint


@dataclass(frozen=True, slots=True)
class SchedulerSpec:
    """Describes a learning-rate scheduler.

    Attributes:
        cls: Registered name, or `"module:QualName"` for one of your own.
        kwargs: Arguments the scheduler is built with.
        interval: Whether the runtime steps it per optimizer step or per
            epoch.
        monitor: Binding a plateau schedule reads, e.g. `"probe/psnr"`.
    """

    cls: str = "cosine"
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    interval: IntervalTypes = "epoch"
    monitor: str | None = None

    def __post_init__(self) -> None:
        """Reject a plateau schedule with nothing to read.

        Raises:
            ValueError: If a plateau schedule names no monitored binding.
        """
        if self.cls == "plateau" and not self.monitor:
            raise ValueError(
                "A plateau schedule steps on a monitored value; give it "
                "monitor=, e.g. monitor='probe/psnr'."
            )

    def build(self, optimizer: Optimizer, **overrides: Any) -> LRScheduler:
        """Build the scheduler.

        Args:
            optimizer: Optimizer whose learning rate is scheduled.
            overrides: Arguments to replace, for building a variant.
        """
        scheduler = require(
            self.cls,
            SCHEDULER_MAP,
            "scheduler",
            fallback=ModelSpec.import_class if ":" in self.cls else None,
        )
        return scheduler(optimizer, **{**self.kwargs, **overrides})


@dataclass(frozen=True, slots=True)
class OptimSpec:
    """Describes an optimizer and the schedule attached to it.

    Attributes:
        cls: Registered name, or `"module:QualName"` for one of your own.
        lr: Learning rate, the one argument every optimizer takes.
        kwargs: Further arguments the optimizer is built with.
        params: Which parameters this optimizer owns: a binding name, several
            of them, or a callable selecting them. Defaults to the stage's
            own model.
        scheduler: The schedule attached to this optimizer, if any.
    """

    cls: str = "adamw"
    lr: float = 1e-4
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    params: str | Sequence[str] | Callable[..., ParamsT] | None = None
    scheduler: SchedulerSpec | None = None

    @staticmethod
    def needs_closure(optimizer: Optimizer | str) -> bool:
        """Whether an optimizer requires a closure to step.

        Args:
            optimizer: A built optimizer, or the name of one.
        """
        if isinstance(optimizer, str):
            return optimizer in CLOSURE_OPTIMIZERS
        return isinstance(
            optimizer, tuple(OPTIMIZER_MAP[name] for name in CLOSURE_OPTIMIZERS)
        )

    @property
    def requires_closure(self) -> bool:
        """Whether stepping this optimizer needs a closure."""
        return OptimSpec.needs_closure(self.cls)

    def build(self, params: ParamsT, **overrides: Any) -> Optimizer:
        """Build the optimizer.

        Args:
            params: Parameters or parameter groups to optimize.
            overrides: Arguments to replace, for building a variant.
        """
        optimizer = require(
            self.cls,
            OPTIMIZER_MAP,
            "optimizer",
            fallback=ModelSpec.import_class if ":" in self.cls else None,
        )
        return optimizer(params, lr=self.lr, **{**self.kwargs, **overrides})

    @classmethod
    def adam(
        cls,
        lr: float = 1e-3,
        params: str | Sequence[str] | Callable[..., ParamsT] | None = None,
        **kwargs: Any,
    ) -> OptimSpec:
        """Describe an Adam optimizer.

        Args:
            lr: Learning rate.
            params: Which parameters this optimizer owns.
            kwargs: Further arguments, e.g. `betas`, `eps`, `weight_decay`.
        """
        return cls(cls="adam", lr=lr, kwargs=kwargs, params=params)

    @classmethod
    def adamw(
        cls,
        lr: float = 1e-4,
        params: str | Sequence[str] | Callable[..., ParamsT] | None = None,
        **kwargs: Any,
    ) -> OptimSpec:
        """Describe an AdamW optimizer.

        Args:
            lr: Learning rate.
            params: Which parameters this optimizer owns.
            kwargs: Further arguments, e.g. `betas`, `eps`, `weight_decay`.
        """
        return cls(cls="adamw", lr=lr, kwargs=kwargs, params=params)

    @classmethod
    def sgd(
        cls,
        lr: float = 1e-2,
        params: str | Sequence[str] | Callable[..., ParamsT] | None = None,
        **kwargs: Any,
    ) -> OptimSpec:
        """Describe an SGD optimizer.

        Args:
            lr: Learning rate.
            params: Which parameters this optimizer owns.
            kwargs: Further arguments, e.g. `momentum`, `nesterov`.
        """
        return cls(cls="sgd", lr=lr, kwargs=kwargs, params=params)

    @classmethod
    def rmsprop(
        cls,
        lr: float = 1e-2,
        params: str | Sequence[str] | Callable[..., ParamsT] | None = None,
        **kwargs: Any,
    ) -> OptimSpec:
        """Describe an RMSprop optimizer.

        Args:
            lr: Learning rate.
            params: Which parameters this optimizer owns.
            kwargs: Further arguments, e.g. `alpha`, `momentum`.
        """
        return cls(cls="rmsprop", lr=lr, kwargs=kwargs, params=params)

    @classmethod
    def adagrad(
        cls,
        lr: float = 1e-2,
        params: str | Sequence[str] | Callable[..., ParamsT] | None = None,
        **kwargs: Any,
    ) -> OptimSpec:
        """Describe an Adagrad optimizer.

        Args:
            lr: Learning rate.
            params: Which parameters this optimizer owns.
            kwargs: Further arguments, e.g. `lr_decay`, `weight_decay`.
        """
        return cls(cls="adagrad", lr=lr, kwargs=kwargs, params=params)

    @classmethod
    def lbfgs(
        cls,
        lr: float = 1.0,
        params: str | Sequence[str] | Callable[..., ParamsT] | None = None,
        **kwargs: Any,
    ) -> OptimSpec:
        """Describe an L-BFGS optimizer.

        Args:
            lr: Learning rate.
            params: Which parameters this optimizer owns.
            kwargs: Further arguments, e.g. `max_iter`, `history_size`.
        """
        return cls(cls="lbfgs", lr=lr, kwargs=kwargs, params=params)

    def with_params(
        self, params: str | Sequence[str] | Callable[..., ParamsT]
    ) -> OptimSpec:
        """Return a copy owning the given parameters.

        Args:
            params: A binding name, several of them, or a callable.
        """
        return replace(self, params=params)

    def with_schedule(
        self,
        schedule: str,
        interval: IntervalTypes = "epoch",
        monitor: str | None = None,
        **kwargs: Any,
    ) -> OptimSpec:
        """Return a copy with a schedule attached.

        Args:
            schedule: Registered name, or `"module:QualName"`.
            interval: Whether the runtime steps it per step or per epoch.
            monitor: Binding a plateau schedule reads.
            kwargs: Arguments the scheduler is built with.
        """
        return replace(
            self,
            scheduler=SchedulerSpec(
                cls=schedule, kwargs=kwargs, interval=interval, monitor=monitor
            ),
        )

    def with_cosine_schedule(
        self, t_max: int, eta_min: float = 0.0, interval: IntervalTypes = "epoch"
    ) -> OptimSpec:
        """Return a copy with a cosine annealing schedule.

        Args:
            t_max: Steps or epochs over which the rate reaches `eta_min`.
            eta_min: Rate at the end of the schedule.
            interval: Whether the runtime steps it per step or per epoch.
        """
        return self.with_schedule(
            "cosine", interval=interval, T_max=t_max, eta_min=eta_min
        )

    def with_cosine_warm_restarts_schedule(
        self,
        t_0: int,
        t_mult: int = 1,
        eta_min: float = 0.0,
        interval: IntervalTypes = "epoch",
    ) -> OptimSpec:
        """Return a copy with a cosine annealing schedule that restarts.

        Args:
            t_0: Steps or epochs until the first restart.
            t_mult: Factor the period grows by after each restart.
            eta_min: Rate at the end of each period.
            interval: Whether the runtime steps it per step or per epoch.
        """
        return self.with_schedule(
            "cosine_warm_restarts",
            interval=interval,
            T_0=t_0,
            T_mult=t_mult,
            eta_min=eta_min,
        )

    def with_step_schedule(
        self, step_size: int, gamma: float = 0.1, interval: IntervalTypes = "epoch"
    ) -> OptimSpec:
        """Return a copy with a schedule dropping the rate at a fixed interval.

        Args:
            step_size: Steps or epochs between drops.
            gamma: Factor the rate is multiplied by at each drop.
            interval: Whether the runtime steps it per step or per epoch.
        """
        return self.with_schedule(
            "step", interval=interval, step_size=step_size, gamma=gamma
        )

    def with_multistep_schedule(
        self,
        milestones: Sequence[int],
        gamma: float = 0.1,
        interval: IntervalTypes = "epoch",
    ) -> OptimSpec:
        """Return a copy with a schedule dropping the rate at given milestones.

        Args:
            milestones: Steps or epochs at which the rate drops.
            gamma: Factor the rate is multiplied by at each drop.
            interval: Whether the runtime steps it per step or per epoch.
        """
        return self.with_schedule(
            "multistep", interval=interval, milestones=list(milestones), gamma=gamma
        )

    def with_exponential_schedule(
        self, gamma: float = 0.95, interval: IntervalTypes = "epoch"
    ) -> OptimSpec:
        """Return a copy with an exponential decay schedule.

        Args:
            gamma: Factor the rate is multiplied by each time.
            interval: Whether the runtime steps it per step or per epoch.
        """
        return self.with_schedule("exponential", interval=interval, gamma=gamma)

    def with_linear_schedule(
        self,
        start_factor: float = 1.0,
        end_factor: float = 0.1,
        total_iters: int = 10,
        interval: IntervalTypes = "epoch",
    ) -> OptimSpec:
        """Return a copy with a linear schedule.

        Args:
            start_factor: Fraction of the rate to begin at.
            end_factor: Fraction of the rate to end at.
            total_iters: Steps or epochs the line spans.
            interval: Whether the runtime steps it per step or per epoch.
        """
        return self.with_schedule(
            "linear",
            interval=interval,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=total_iters,
        )

    def with_onecycle_schedule(
        self,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        interval: IntervalTypes = "step",
    ) -> OptimSpec:
        """Return a copy with a one-cycle schedule.

        Args:
            max_lr: Rate at the peak.
            total_steps: Steps the cycle spans.
            pct_start: Fraction of the cycle spent warming up.
            interval: Whether the runtime steps it per step or per epoch.
        """
        return self.with_schedule(
            "onecycle",
            interval=interval,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
        )

    def with_plateau_schedule(
        self,
        monitor: str,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        interval: IntervalTypes = "epoch",
    ) -> OptimSpec:
        """Return a copy with a schedule dropping the rate on a plateau.

        Args:
            monitor: Binding the schedule reads, e.g. `"probe/psnr"`.
            mode: `"min"` when lower is better, `"max"` when higher is.
            factor: Factor the rate is multiplied by on a plateau.
            patience: Readings without improvement before dropping.
            interval: Whether the runtime steps it per step or per epoch.
        """
        return self.with_schedule(
            "plateau",
            interval=interval,
            monitor=monitor,
            mode=mode,
            factor=factor,
            patience=patience,
        )
