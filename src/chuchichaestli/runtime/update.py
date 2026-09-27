# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Applying one optimizer step, over a single group of parameters or several."""

from __future__ import annotations
from abc import ABC, abstractmethod
from contextlib import nullcontext
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal
import torch
from torch import nn
from torch.optim import Optimizer
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.swa_utils import SWALR
from chuchichaestli.data.batch import BatchType, samples_in_batch
from chuchichaestli.runtime.context import Context
from chuchichaestli.runtime.traits import Objective
from chuchichaestli.training.objective import Loss
from chuchichaestli.training.optim import OptimSpec
from chuchichaestli.training.update import Swa, UpdatePolicy, average_targets


__all__ = [
    "WeightsTypes",
    "SwaWindow",
    "Updater",
    "Step",
    "Alternating",
    "Simultaneous",
]


WeightsTypes = Literal["model", "ema", "swa"]


class SwaWindow:
    """The tail of a run over which weights are averaged equally.

    Attributes:
        request: What to average, in any spelling `average_targets` accepts.
        start: Fraction of the run after which the window opens.
        lr: Rate `SWALR` holds once it does, or `None` for no handover.
        anneal: Passes `SWALR` takes to reach that rate.
        averages: The average kept per binding name.
        schedulers: The `SWALR` bound to each update group.
    """

    def __init__(
        self,
        request: bool | str | Sequence[str] | None = None,
        start: float = 0.75,
        lr: float | None = None,
        anneal: int = 10,
    ):
        """Constructor.

        Args:
            request: What to average, in any spelling `average_targets`
                accepts. `None` averages nothing.
            start: Fraction of the run after which the window opens.
            lr: Rate to hold once it does. Setting it hands the schedule over
                to `SWALR` and retires whatever ran before.
            anneal: Passes `SWALR` takes to reach `lr`.

        Raises:
            ValueError: If `start` lies outside `[0, 1)`, or a rate is given
                with nothing to average.
        """
        if request is not None and not 0.0 <= start < 1.0:
            raise ValueError(f"swa_start is a fraction of the run, got {start!r}.")
        if lr is not None and request is None:
            raise ValueError(
                f"swa_lr={lr} schedules the averaging window, but the stage "
                "was not asked to average."
            )
        self.request = request
        self.start = start
        self.lr = lr
        self.anneal = anneal
        self.averages: dict[str, Swa] = {}
        self.schedulers: dict[str | None, LRScheduler] = {}

    def __bool__(self) -> bool:
        """Whether this window averages anything at all."""
        return bool(self.averages)

    def build(
        self,
        default: str,
        optimizers: Mapping[str | None, Optimizer],
        ctx: Context,
    ) -> None:
        """Build the averages and the rate schedule, and publish each average.

        Args:
            default: Binding to average when the request names none.
            optimizers: The optimizer of each update group.
            ctx: Execution context for this entry.
        """
        self.averages = {}
        for name in average_targets(self.request, default):
            averaged = Swa(ctx.resolve(name))
            ctx.publish(f"{name}/swa", averaged)
            self.averages[name] = averaged
        self.schedulers = (
            {}
            if self.lr is None
            else {
                group: SWALR(optimizer, swa_lr=self.lr, anneal_epochs=self.anneal)
                for group, optimizer in optimizers.items()
            }
        )

    def is_open(self, epoch: int, epochs: int | None) -> bool:
        """Whether a pass falls in the tail being averaged.

        Args:
            epoch: Pass that just finished, counted from zero.
            epochs: Passes the run makes in total, or `None` if open-ended.
        """
        if not self.averages or epochs is None:
            return False
        return epoch + 1 > self.start * epochs

    def hand_over(self, *registries: dict[str | None, LRScheduler]) -> None:
        """Retire the schedules that ran before the window opened.

        Torch's recipe holds the rate `SWALR` asks for, so whatever ran
        before must stop rather than compete with it. Without a rate of its
        own the window takes nothing over, and they are left running.

        Args:
            registries: Where the retired schedules are registered.
        """
        for group in self.schedulers:
            for registry in registries:
                registry.pop(group, None)

    def take(self, ctx: Context) -> None:
        """Fold the current weights into each average and advance the rate.

        Args:
            ctx: Execution context for this entry.
        """
        for scheduler in self.schedulers.values():
            scheduler.step()
        for name, averaged in self.averages.items():
            averaged.update_parameters(ctx.resolve(name))

    def refresh_batch_stats(
        self,
        batches: Callable[[], Iterable[Any]],
        device: torch.device | str | None = None,
    ) -> None:
        """Recompute the batch statistics each average inherited.

        Args:
            batches: Called per average to open a fresh pass over the data.
            device: Device to run the pass on.
        """
        for averaged in self.averages.values():
            if averaged.has_batch_norm:
                averaged.refresh_batch_stats(batches(), device=device)

    def state_dict(self) -> dict[str, Any]:
        """Return the window's resumable state."""
        state: dict[str, Any] = {
            f"swa/{n}": a.state_dict() for n, a in self.averages.items()
        }
        state.update({f"swalr/{g}": s.state_dict() for g, s in self.schedulers.items()})
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state previously returned by `state_dict`.

        Args:
            state: Mapping as returned by `state_dict`.
        """
        for name, averaged in self.averages.items():
            saved = state.get(f"swa/{name}")
            if saved is not None:
                averaged.load_state_dict(saved)
        for group, scheduler in self.schedulers.items():
            saved = state.get(f"swalr/{group}")
            if saved is not None:
                scheduler.load_state_dict(saved)


class Updater(ABC):
    """Shared machinery for applying one optimizer step.

    Attributes:
        frequencies: How often each group steps, relative to the others.
        balance_ranks: Whether a micro-batch is weighed against every
            process's samples rather than this one's alone.
        policy: Clipping, stepping and loss weighting.
        optimizers: The optimizer of each group, once bound.
        schedulers: The per-step scheduler of each group, once bound.
    """

    def __init__(
        self,
        groups: Sequence[str | None] | Mapping[str | None, int] | None = None,
        frequencies: Sequence[int] | Mapping[str | None, int] | None = None,
        policy: UpdatePolicy | None = None,
        precision: torch.dtype | None = None,
    ):
        """Constructor.

        Args:
            groups: The update groups, in the order they are applied; the order
                matters under `Alternating`.
            frequencies: How often a group steps relative to the others, either
                named or one per group in the same order; missing groups step
                every time.
            policy: Clipping, stepping and loss weighting.
            precision: Precision the forward runs in, or `None` for full precision.

        Raises:
            ValueError: If frequencies are given twice, positional ones name
                no groups, there are no groups, a group repeats, a frequency
                names no group, the wrong number of frequencies is given, or
                a frequency is not a positive divisor of the largest one.
        """
        if isinstance(groups, Mapping):
            if frequencies is not None:
                raise ValueError(
                    "Frequencies were given twice; pass them as `groups` or "
                    "as `frequencies`, not both."
                )
            groups, frequencies = None, groups
        if groups is None:
            if isinstance(frequencies, Mapping):
                groups = tuple(frequencies)
            elif frequencies is not None:
                raise ValueError(
                    "Frequencies given by position name no groups; pass the "
                    "groups too, or name them in a mapping."
                )
            else:
                groups = (None,)
        self._groups = tuple(groups)
        if not self._groups:
            raise ValueError("An update needs at least one group.")
        if len(set(self._groups)) != len(self._groups):
            raise ValueError(
                f"Every group needs its own name, got {[str(g) for g in self._groups]}."
            )
        if frequencies is None:
            named: dict[str | None, int] = {}
        elif isinstance(frequencies, Mapping):
            named = dict(frequencies)
            unknown = sorted(str(g) for g in named if g not in self._groups)
            if unknown:
                raise ValueError(
                    f"Frequencies {unknown} name no group of this update; it "
                    f"has {[str(g) for g in self._groups]}."
                )
        else:
            rates = tuple(frequencies)
            if len(rates) != len(self._groups):
                raise ValueError(
                    f"Got {len(rates)} frequencies for {len(self._groups)} "
                    "groups; give one per group, or a mapping naming them."
                )
            named = dict(zip(self._groups, rates))
        self.frequencies: dict[str | None, int] = {
            group: named.get(group, 1) for group in self._groups
        }
        if any(rate < 1 for rate in self.frequencies.values()):
            raise ValueError(f"Frequencies are positive, got {self.frequencies}.")
        most = max(self.frequencies.values())
        if any(most % rate for rate in self.frequencies.values()):
            raise ValueError(
                f"Every frequency must divide the largest ({most}), got "
                f"{self.frequencies}."
            )
        self._cadence = {g: most // rate for g, rate in self.frequencies.items()}
        self.policy = policy or UpdatePolicy()
        self.precision = precision
        self.balance_ranks = False
        self._scalers: dict[str | None, GradScaler] = {}
        self.optimizers: dict[str | None, Optimizer] = {}
        self.schedulers: dict[str | None, LRScheduler] = {}
        self._params: dict[str | None, list[nn.Parameter]] = {}

    def __repr__(self) -> str:
        """Return a short description of the update."""
        return f"{type(self).__name__}({self.frequencies})"

    def groups(self, step: int | None = None) -> tuple[str | None, ...]:
        """Return the groups stepping at a step, or all of them.

        Args:
            step: Optimizer steps taken so far, or `None` to ignore the
                frequencies.
        """
        if step is None:
            return self._groups
        return tuple(g for g in self._groups if step % self._cadence[g] == 0)

    def bind(
        self,
        optimizers: Mapping[str | None, Optimizer],
        schedulers: Mapping[str | None, LRScheduler] | None = None,
    ) -> None:
        """Attach the optimizers this update drives.

        Args:
            optimizers: The optimizer of each group.
            schedulers: Schedulers advanced after every optimizer step; those
                advanced per epoch stay with the stage.

        Raises:
            ValueError: If the optimizers do not match the groups, or a
                multi-group update is given an optimizer needing a closure.
        """
        if set(optimizers) != set(self._groups):
            raise ValueError(
                f"Expected one optimizer per group {sorted(map(str, self._groups))}, "
                f"got {sorted(map(str, optimizers))}."
            )
        closure_groups = sorted(
            str(group)
            for group, optimizer in optimizers.items()
            if OptimSpec.needs_closure(optimizer)
        )
        if closure_groups and len(self._groups) > 1:
            raise ValueError(
                f"Groups {closure_groups} re-evaluate the loss, which a "
                "multi-group update cannot drive; give them a stage of their own."
            )
        self.optimizers = dict(optimizers)
        self.schedulers = dict(schedulers or {})
        self._params = {
            group: [p for pg in optimizer.param_groups for p in pg["params"]]
            for group, optimizer in self.optimizers.items()
        }

    @abstractmethod
    def apply(
        self, objective: Objective, batches: Sequence[BatchType], ctx: Context
    ) -> Loss:
        """Run the objective over a step's micro-batches and step.

        Args:
            objective: Produces the loss for each micro-batch.
            batches: The micro-batches making up this step.
            ctx: Execution context for the stage.
        """

    def _accumulate(
        self,
        objective: Objective,
        batches: Sequence[BatchType],
        ctx: Context,
        group: str | None,
    ) -> Loss:
        """Run the objective over the micro-batches and backpropagate.

        Each micro-batch contributes its own share of the step's samples, so
        a shorter final batch is weighted appropriately.

        Args:
            objective: Produces the loss for each micro-batch.
            batches: The micro-batches making up this step.
            ctx: Execution context for the stage.
            group: Update group being applied, or `None`.
        """
        view = ctx.at_group(group)
        counts = [samples_in_batch(batch) for batch in batches]
        rank_samples = self._rank_samples(sum(counts) or 1, ctx)
        total: torch.Tensor | None = None
        parts: dict[str, torch.Tensor] = {}
        scaler = self._scaler(group, ctx)
        for batch, count in zip(batches, counts):
            with self._autocast(ctx):
                loss = objective.compute(batch, view)
            share = self.policy.share(loss.total, count / rank_samples)
            scaler.scale(share).backward()
            reported = share.detach()
            total = reported if total is None else total + reported
            for name, value in loss.parts.items():
                weighted = self.policy.share(value.detach(), count / rank_samples)
                parts[name] = weighted + parts.get(name, 0)
        return Loss(total=torch.zeros(()) if total is None else total, parts=parts)

    def _rank_samples(self, samples: int, ctx: Context) -> float:
        """Return the samples given, or their average across processes.

        Args:
            samples: Samples this process holds for the step.
            ctx: Execution context, carrying the topology.
        """
        if not self.balance_ranks or ctx.topology.world_size == 1:
            return float(samples)
        total = ctx.topology.reduce(
            torch.tensor(float(samples), device=ctx.device), "sum"
        )
        return float(total) / ctx.topology.world_size

    def _autocast(self, ctx: Context) -> Any:
        """Return the precision the forward runs under.

        Args:
            ctx: Execution context, naming the device to autocast for.
        """
        if self.precision is None:
            return nullcontext()
        return torch.autocast(device_type=ctx.device.type, dtype=self.precision)

    def _scaler(self, group: str | None, ctx: Context) -> GradScaler:
        """Return a group's gradient scaler, building it on first use.

        Scaling is needed only for float16, whose gradients underflow; every
        other precision gets a scaler that passes values through.

        Args:
            group: Update group being applied, or `None`.
            ctx: Execution context, naming the device to scale for.
        """
        if group not in self._scalers:
            self._scalers[group] = GradScaler(
                ctx.device.type, enabled=self.precision is torch.float16
            )
        return self._scalers[group]

    def _step_group(self, group: str | None) -> None:
        """Clip, step the optimizer and advance its per-step scheduler.

        Args:
            group: Update group being applied, or `None`.
        """
        optimizer = self.optimizers[group]
        scaler = self._scalers.get(group)
        if scaler is not None and scaler.is_enabled():
            if self.policy.clip is not None:
                scaler.unscale_(optimizer)
                self.policy.clip_grads(self._params[group])
            scaler.step(optimizer)
            scaler.update()
        else:
            self.policy.step(optimizer, self._params[group])
        scheduler = self.schedulers.get(group)
        if scheduler is not None:
            scheduler.step()

    @staticmethod
    def state_key(prefix: str, group: str | None) -> str:
        """Return the state key of a group's component.

        Args:
            prefix: What the component is, e.g. `"optim"`.
            group: Update group it belongs to, or `None`.
        """
        return prefix if group is None else f"{prefix}/{group}"

    def state_dict(self) -> dict[str, Any]:
        """Return the optimizer and scheduler state of every group."""
        state: dict[str, Any] = {
            self.state_key("optim", g): o.state_dict()
            for g, o in self.optimizers.items()
        }
        state.update(
            {
                self.state_key("sched", g): s.state_dict()
                for g, s in self.schedulers.items()
            }
        )
        state.update(
            {
                self.state_key("scaler", g): s.state_dict()
                for g, s in self._scalers.items()
            }
        )
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state previously returned by `state_dict`.

        Args:
            state: Mapping as returned by `state_dict`.
        """
        for group, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state[self.state_key("optim", group)])
        for group, scheduler in self.schedulers.items():
            scheduler.load_state_dict(state[self.state_key("sched", group)])
        for group, scaler in self._scalers.items():
            saved = state.get(self.state_key("scaler", group))
            if saved is not None:
                scaler.load_state_dict(saved)


class Step(Updater):
    """One optimizer step over a single group of parameters."""

    def __init__(
        self,
        policy: UpdatePolicy | None = None,
        precision: torch.dtype | None = None,
    ):
        """Constructor.

        Args:
            policy: Clipping, stepping and loss weighting.
            precision: Precision the forward runs in, or `None` for full precision.
        """
        super().__init__((None,), policy=policy, precision=precision)

    def apply(
        self, objective: Objective, batches: Sequence[BatchType], ctx: Context
    ) -> Loss:
        """Accumulate the micro-batches and step once.

        Args:
            objective: Produces the loss for each micro-batch.
            batches: The micro-batches making up this step.
            ctx: Execution context for the stage.
        """
        optimizer = self.optimizers[None]
        if OptimSpec.needs_closure(optimizer):
            return self._closure_step(objective, batches, ctx)
        self.policy.zero_grad(optimizer)
        loss = self._accumulate(objective, batches, ctx, None)
        self._step_group(None)
        return loss

    def _closure_step(
        self, objective: Objective, batches: Sequence[BatchType], ctx: Context
    ) -> Loss:
        """Step an optimizer that re-evaluates the loss itself.

        Args:
            objective: Produces the loss for each micro-batch.
            batches: The single micro-batch making up this step.
            ctx: Execution context for the stage.

        Raises:
            ValueError: If the step was given more than one micro-batch.
        """
        optimizer = self.optimizers[None]
        if len(batches) != 1:
            raise ValueError(
                f"{type(optimizer).__name__} re-evaluates the loss, so it cannot "
                f"accumulate {len(batches)} micro-batches."
            )
        view = ctx.at_group(None)
        parts: dict[str, torch.Tensor] = {}

        def compute() -> torch.Tensor:
            loss = objective.compute(batches[0], view)
            parts.clear()
            parts.update({name: v.detach() for name, v in loss.parts.items()})
            return loss.total

        closure = self.policy.closure_for(optimizer, self._params[None], compute)
        total = optimizer.step(closure)
        scheduler = self.schedulers.get(None)
        if scheduler is not None:
            scheduler.step()
        if not isinstance(total, torch.Tensor):
            total = torch.as_tensor(float("nan") if total is None else total)
        return Loss(total=total.detach(), parts=parts)


class Alternating(Updater):
    """Steps each group in turn, so a group sees the previous one's update."""

    def apply(
        self, objective: Objective, batches: Sequence[BatchType], ctx: Context
    ) -> Loss:
        """Accumulate and step one group at a time, in order.

        Args:
            objective: Produces the loss for each micro-batch.
            batches: The micro-batches making up this step.
            ctx: Execution context for the stage.
        """
        results: dict[str | None, Loss] = {}
        for group in self.groups(ctx.progress.step):
            self.policy.zero_grad(self.optimizers[group])
            results[group] = self._accumulate(objective, batches, ctx, group)
            self._step_group(group)
        return Loss.merge(results)


class Simultaneous(Updater):
    """Steps every group against the same pre-step parameters."""

    def apply(
        self, objective: Objective, batches: Sequence[BatchType], ctx: Context
    ) -> Loss:
        """Accumulate every group before any of them steps.

        Args:
            objective: Produces the loss for each micro-batch.
            batches: The micro-batches making up this step.
            ctx: Execution context for the stage.
        """
        groups = self.groups(ctx.progress.step)
        for group in groups:
            self.policy.zero_grad(self.optimizers[group])
        results = {g: self._accumulate(objective, batches, ctx, g) for g in groups}
        for group in groups:
            self._step_group(group)
        return Loss.merge(results)
