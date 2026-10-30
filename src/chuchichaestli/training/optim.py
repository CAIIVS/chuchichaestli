# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Optimizer and learning-rate schedule specifications."""

from __future__ import annotations
import inspect
from bisect import bisect_right
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau, SequentialLR
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
    "SequentialSchedulerSpec",
    "SequentialScheduler",
    "OptimSpec",
    "disjoint_params",
]


OptimizerTypes = Literal["adam", "adamw", "sgd", "rmsprop", "adagrad", "lbfgs"]
SchedulerTypes = Literal[
    "constant",
    "cosine",
    "cosine_warm_restarts",
    "exponential",
    "lambda",
    "linear",
    "multistep",
    "onecycle",
    "plateau",
    "polynomial",
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
    "constant": torch.optim.lr_scheduler.ConstantLR,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "cosine_warm_restarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
    "lambda": torch.optim.lr_scheduler.LambdaLR,
    "linear": torch.optim.lr_scheduler.LinearLR,
    "multistep": torch.optim.lr_scheduler.MultiStepLR,
    "onecycle": torch.optim.lr_scheduler.OneCycleLR,
    "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "polynomial": torch.optim.lr_scheduler.PolynomialLR,
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
        interval: Whether the runner steps it per optimizer step or per
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

    def _with_length(self, length: int) -> SchedulerSpec:
        """Return a copy spanning `length` intervals, if its schedule counts them.

        Schedules that take `total_iters` (linear, constant, polynomial) are
        given it unless it is set; any other is returned as it is.

        Args:
            length: Steps or epochs the schedule is to span.
        """
        if "total_iters" in self.kwargs:
            return self
        scheduler = require(
            self.cls,
            SCHEDULER_MAP,
            "scheduler",
            fallback=ModelSpec.import_class if ":" in self.cls else None,
        )
        if "total_iters" not in inspect.signature(scheduler).parameters:
            return self
        return replace(self, kwargs={**self.kwargs, "total_iters": length})


class SequentialScheduler(SequentialLR):
    """Torch's `SequentialLR`, admitting a plateau schedule among its members.

    A plateau member is stepped with the value it reads; on entering it, the
    rate returns to the optimizer's initial one, from which it then drops.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: Sequence[LRScheduler],
        milestones: Sequence[int],
    ):
        """Constructor.

        Args:
            optimizer: Optimizer whose learning rate is scheduled.
            schedulers: The schedules, run one after another.
            milestones: Steps or epochs at which each schedule after the first
                takes over.

        Raises:
            ValueError: If there is no schedule, the milestones do not number
                one fewer than the schedules or do not increase, or a schedule
                belongs to another optimizer.
        """
        if not schedulers:
            raise ValueError("A sequence of schedules needs at least one.")
        if len(milestones) != len(schedulers) - 1:
            raise ValueError(
                f"{len(schedulers)} schedules take {len(schedulers) - 1} "
                f"milestones, got {len(milestones)}."
            )
        if any(b <= a for a, b in zip(milestones, milestones[1:])) or any(
            m < 1 for m in milestones
        ):
            raise ValueError(
                f"Milestones must be positive and increasing, got {list(milestones)}."
            )
        for scheduler in schedulers:
            if scheduler.optimizer is not optimizer:
                raise ValueError(
                    f"{type(scheduler).__name__} schedules another optimizer."
                )
        self._schedulers = list(schedulers)
        self._milestones = list(milestones)
        self.last_epoch = 0
        self.optimizer = optimizer
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"]
        self.recursive_undo()
        first = self._schedulers[0]
        if isinstance(first, ReduceLROnPlateau):
            first._last_lr = [group["lr"] for group in optimizer.param_groups]
        else:
            first._initial_step()
        self._last_lr = first.get_last_lr()

    def _upcoming(self) -> tuple[LRScheduler, bool]:
        """Return the schedule the next step goes to, and whether it starts there."""
        epoch = self.last_epoch + 1
        index = bisect_right(self._milestones, epoch)
        starts = index > 0 and self._milestones[index - 1] == epoch
        return self._schedulers[index], starts

    @property
    def reads_value(self) -> bool:
        """Whether the next step goes to a plateau schedule, needing its value."""
        scheduler, starts = self._upcoming()
        return isinstance(scheduler, ReduceLROnPlateau) and not starts

    def step(self, metrics: float | None = None) -> None:  # type: ignore[override]
        """Advance the schedule in charge.

        Args:
            metrics: The value a plateau schedule reads; ignored by the others.

        Raises:
            ValueError: If a plateau schedule is in charge and no value is given.
        """
        scheduler, starts = self._upcoming()
        if self.reads_value and metrics is None:
            raise ValueError(
                "The schedule in charge is a plateau one, which steps on a value."
            )
        self.last_epoch += 1
        if isinstance(scheduler, ReduceLROnPlateau):
            if starts:
                for group in self.optimizer.param_groups:
                    group["lr"] = group["initial_lr"]
                scheduler._last_lr = [g["lr"] for g in self.optimizer.param_groups]
            else:
                scheduler.step(metrics)
        elif starts:
            scheduler._update_lr(0)
        else:
            scheduler.step()
        self._last_lr = scheduler.get_last_lr()


@dataclass(frozen=True, slots=True)
class SequentialSchedulerSpec:
    """Describes schedules run one after another, e.g. a warmup then a decay.

    Each schedule counts from zero when it takes over, so a cosine taking over
    at milestone 10 with `T_max=90` reaches its minimum at step 100.

    Attributes:
        schedulers: The schedules, in the order they run.
        milestones: Steps or epochs at which each schedule after the first
            takes over.
    """

    schedulers: Sequence[SchedulerSpec]
    milestones: Sequence[int]

    def __post_init__(self) -> None:
        """Check that the schedules can run as one.

        Raises:
            ValueError: If the milestones do not number one fewer than the
                schedules, the schedules step at different intervals, or more
                than one reads a monitored value.
        """
        object.__setattr__(self, "schedulers", tuple(self.schedulers))
        object.__setattr__(self, "milestones", tuple(self.milestones))
        if len(self.milestones) != len(self.schedulers) - 1:
            raise ValueError(
                f"{len(self.schedulers)} schedules take {len(self.schedulers) - 1} "
                f"milestones, got {len(self.milestones)}."
            )
        intervals = {spec.interval for spec in self.schedulers}
        if len(intervals) > 1:
            raise ValueError(
                "Schedules run in sequence step at one interval, got "
                f"{sorted(intervals)}."
            )
        monitors = {spec.monitor for spec in self.schedulers if spec.monitor}
        if len(monitors) > 1:
            raise ValueError(
                f"Schedules run in sequence read one value, got {sorted(monitors)}."
            )

    @property
    def interval(self) -> IntervalTypes:
        """Whether the runner steps the sequence per step or per epoch."""
        return self.schedulers[0].interval

    @property
    def monitor(self) -> str | None:
        """Binding a plateau schedule in the sequence reads, if any."""
        return next((s.monitor for s in self.schedulers if s.monitor), None)

    def build(self, optimizer: Optimizer) -> SequentialScheduler:
        """Build every schedule and run them in sequence.

        Args:
            optimizer: Optimizer whose learning rate is scheduled.
        """
        return SequentialScheduler(
            optimizer,
            [spec.build(optimizer) for spec in self.schedulers],
            self.milestones,
        )

    def after(self, spec: SchedulerSpec, length: int) -> SequentialSchedulerSpec:
        """Return the sequence with a schedule run for `length` before it.

        Args:
            spec: The schedule to run first.
            length: Steps or epochs it runs for.
        """
        return SequentialSchedulerSpec(
            (spec, *self.schedulers),
            (length, *(m + length for m in self.milestones)),
        )


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
    scheduler: SchedulerSpec | SequentialSchedulerSpec | None = None

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
            interval: Whether the runner steps it per step or per epoch.
            monitor: Binding a plateau schedule reads.
            kwargs: Arguments the scheduler is built with.
        """
        return replace(
            self,
            scheduler=SchedulerSpec(
                cls=schedule, kwargs=kwargs, interval=interval, monitor=monitor
            ),
        )

    def with_warmup(
        self, schedule: str | SchedulerSpec, length: int, **kwargs: Any
    ) -> OptimSpec:
        """Return a copy running a warmup schedule before the attached one.

        Without an attached schedule, the rate holds at `lr` once the warmup
        ends. The warmup steps at the attached schedule's interval, and is
        given `total_iters=length` if it takes one and it is not set.

        Args:
            schedule: Registered name, `"module:QualName"`, or a spec.
            length: Steps or epochs the warmup runs for.
            kwargs: Arguments the warmup is built with, when named.

        Raises:
            ValueError: If `length` is not positive, or a warmup spec steps at
                another interval than the attached schedule.
        """
        if length < 1:
            raise ValueError(f"A warmup lasts at least one interval, got {length!r}.")
        main = self.scheduler
        interval = main.interval if main is not None else "epoch"
        if isinstance(schedule, str):
            warmup = SchedulerSpec(cls=schedule, kwargs=kwargs, interval=interval)
        elif main is not None and schedule.interval != interval:
            raise ValueError(
                f"The warmup steps per {schedule.interval}, the schedule after "
                f"it per {interval}; give both the same interval."
            )
        else:
            warmup = schedule
        warmup = warmup._with_length(length)
        if main is None:
            main = SchedulerSpec(
                cls="constant",
                kwargs={"factor": 1.0, "total_iters": 0},
                interval=warmup.interval,
            )
        if isinstance(main, SchedulerSpec):
            main = SequentialSchedulerSpec((main,), ())
        return replace(self, scheduler=main.after(warmup, length))

    def with_cosine_schedule(
        self, t_max: int, eta_min: float = 0.0, interval: IntervalTypes = "epoch"
    ) -> OptimSpec:
        """Return a copy with a cosine annealing schedule.

        Args:
            t_max: Steps or epochs over which the rate reaches `eta_min`.
            eta_min: Rate at the end of the schedule.
            interval: Whether the runner steps it per step or per epoch.
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
            interval: Whether the runner steps it per step or per epoch.
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
            interval: Whether the runner steps it per step or per epoch.
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
            interval: Whether the runner steps it per step or per epoch.
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
            interval: Whether the runner steps it per step or per epoch.
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
            interval: Whether the runner steps it per step or per epoch.
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
            interval: Whether the runner steps it per step or per epoch.
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
            interval: Whether the runner steps it per step or per epoch.
        """
        return self.with_schedule(
            "plateau",
            interval=interval,
            monitor=monitor,
            mode=mode,
            factor=factor,
            patience=patience,
        )
