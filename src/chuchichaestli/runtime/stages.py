# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""The stage register: blocks that run once, and phases that hold stages."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from dataclasses import replace
from typing import Any
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT
from torch.utils.data import Dataset
from chuchichaestli.data.batch import (
    BatchType,
    input_in_batch,
    unpack_batch,
    batch_to_device,
    samples_in_batch,
)
from chuchichaestli.runtime.context import Context
from chuchichaestli.runtime.data import DataManager
from chuchichaestli.runtime.objective import CompositeObjective, Criterion
from chuchichaestli.runtime.update import Step, Updater, WeightsTypes
from chuchichaestli.runtime.events import EventType, Progress, Signal
from chuchichaestli.runtime.traits import Objective, Stage, Stateful
from chuchichaestli.training.objective import Loss, Term
from chuchichaestli.training.optim import OptimSpec, disjoint_params
from chuchichaestli.training.update import (
    ClipTypes,
    Ema,
    ReductionTypes,
    UpdatePolicy,
)
from chuchichaestli.utils.functools import partialclass
from chuchichaestli.data.save import save_dataset
from chuchichaestli.utils.io import read_state, staged, writer_for
from chuchichaestli.utils.registry import require


__all__ = [
    "StageBlock",
    "Call",
    "Load",
    "Export",
    "Barrier",
    "Phase",
    "Program",
    "Repeat",
    "When",
    "Every",
    "StageLoop",
    "Train",
    "Finetune",
    "Inference",
    "Eval",
    "Predict",
]


class StageBlock(ABC):
    """A stage that performs its work in a single event.

    Subclasses implement `core`, the stage's main function.

    Attributes:
        name: Identifies the stage within its parent.
        requires: Binding names that must resolve before the run starts.
        provides: Binding names this stage publishes.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        requires: Sequence[str] = (),
        provides: Sequence[str] = (),
    ):
        """Constructor.

        Args:
            name: Identifies the stage within its parent; defaults to the
                lowercased class name.
            requires: Binding names that must resolve before the run starts.
            provides: Binding names this stage publishes.
        """
        self.name = name or type(self).__name__.lower()
        self.requires = tuple(requires)
        self.provides = tuple(provides)
        self._progress = Progress()

    @abstractmethod
    def core(self, ctx: Context) -> None:
        """Perform the stage's core execution unit.

        Args:
            ctx: Execution context for this entry.
        """

    def __repr__(self) -> str:
        """Return a short description of the stage."""
        return f"{type(self).__name__}({self.name!r})"

    def enter(self, ctx: Context) -> Signal:
        """Reset to a fresh progress and announce the stage.

        Args:
            ctx: Execution context for this entry.
        """
        self._progress = Progress()
        ctx.progress = self._progress
        return ctx.emit(EventType.STAGE_BEGAN, stage=type(self).__name__)

    def execute(self, ctx: Context) -> Signal:
        """Run `core` once and finish.

        Args:
            ctx: Execution context for this entry.
        """
        self.core(ctx)
        self._progress = self._progress.next_step().finish()
        ctx.progress = self._progress
        return Signal.DONE

    def leave(self, ctx: Context) -> Signal:
        """Announce that the stage is over.

        Args:
            ctx: Execution context for this entry.
        """
        ctx.progress = self._progress
        ctx.emit(EventType.STAGE_ENDED, stage=type(self).__name__)
        return Signal.GO

    def progress(self) -> Progress:
        """Return where the stage currently is in its own work."""
        return self._progress

    def state_dict(self) -> dict[str, Any]:
        """Return the stage's resumable state."""
        return {"progress": self._progress.to_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state previously returned by `state_dict`.

        Args:
            state: Mapping as returned by `state_dict`.
        """
        self._progress = Progress.from_dict(state.get("progress", {}))


class Call(StageBlock):
    """Run one callable."""

    def __init__(
        self,
        name: str | None = None,
        *,
        fn: Callable[[Context], Any],
        requires: Sequence[str] = (),
        provides: Sequence[str] = (),
    ):
        """Constructor.

        Args:
            name: Identifies the stage within its parent.
            fn: Called with the context; its return value may be published.
            requires: Binding names that must resolve before the run starts.
            provides: Binding names this stage publishes.
        """
        super().__init__(name, requires=requires, provides=provides)
        self.fn = fn

    def core(self, ctx: Context) -> None:
        """Invoke the callable and publish whatever it returned.

        Args:
            ctx: Execution context for this entry.
        """
        result = self.fn(ctx)
        if isinstance(result, Mapping):
            for key, value in result.items():
                ctx.publish(key, value)
        elif result is not None and len(self.provides) == 1:
            ctx.publish(self.provides[0], result)


class Load(StageBlock):
    """Restore weights from a file into a target binding.

    Reads safetensors and torch archives (through `weights_only`).
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        path: str | Path,
        target: str = "model",
        strict: bool = True,
    ):
        """Constructor.

        Args:
            name: Identifies the stage within its parent.
            path: File to read; the suffix picks the reader.
            target: Binding name of the module to load into.
            strict: Whether every key must match, as for `load_state_dict`.
        """
        super().__init__(name, requires=(target,))
        self.path = Path(path)
        self.target = target
        self.strict = strict

    def core(self, ctx: Context) -> None:
        """Load the file into the target binding.

        Args:
            ctx: Execution context for this entry.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the suffix names no known format.
        """
        if not self.path.is_file():
            raise FileNotFoundError(f"No weights to load at {str(self.path)!r}.")
        ctx[self.target].load_state_dict(read_state(self.path), strict=self.strict)


class Export(StageBlock):
    """Write a binding's weights out, from rank 0 only.

    The suffix picks the format (.safetensors or .pt/.pth).
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        path: str | Path,
        source: str = "model",
    ):
        """Constructor.

        Args:
            name: Identifies the stage within its parent.
            path: File to write; the suffix picks the writer.
            source: Binding name of the module to export.
        """
        super().__init__(name, requires=(source,))
        self.path = Path(path)
        self.source = source

    def core(self, ctx: Context) -> None:
        """Write the source binding's state to disk.

        Args:
            ctx: Execution context for this entry.

        Raises:
            ValueError: If the suffix names no known format.
        """
        writer = writer_for(self.path)
        if not ctx.topology.is_main:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with staged(self.path) as (scratch,):
            writer(scratch, ctx[self.source].state_dict())


class Barrier(StageBlock):
    """Wait until every process has arrived."""

    def core(self, ctx: Context) -> None:
        """Synchronize the processes taking part.

        Args:
            ctx: Execution context for this entry.
        """
        ctx.topology.barrier()


class Phase:
    """A stage made of stages.

    Nesting is unbounded: checkpoint state, random streams and event labels
    all key off the same path at every depth.

    Attributes:
        name: Identifies the phase within its parent.
        stages: The children, run in order.
        provide: Artifacts bound for the whole subtree on entry.
        requires: Binding names that must resolve before the run starts.
        provides: Binding names this phase publishes.
    """

    def __init__(
        self,
        name: str | None = None,
        stages: Sequence[Stage] = (),
        *,
        provide: Mapping[str, Any] | None = None,
        requires: Sequence[str] = (),
        provides: Sequence[str] = (),
    ):
        """Constructor.

        Args:
            name: Identifies the phase within its parent.
            stages: The children, run in order.
            provide: Artifacts bound for the whole subtree on entry.
            requires: Binding names that must resolve before the run starts.
            provides: Binding names this phase publishes.
        """
        self.name = name or type(self).__name__.lower()
        self.stages = tuple(stages)
        self.provide = dict(provide or {})
        self.requires = tuple(requires)
        self.provides = tuple(provides)
        self._progress = Progress()
        self._index = 0
        self._entered = False
        self._child_ctx: Context | None = None

    def __repr__(self) -> str:
        """Return a short description of the phase."""
        return f"{type(self).__name__}({self.name!r}, {len(self.stages)} stages)"

    def enter(self, ctx: Context) -> Signal:
        """Bind the shared artifacts and reset to the first child.

        Args:
            ctx: Execution context for this entry.
        """
        self._progress = Progress()
        self._index = 0
        self._entered = False
        self._child_ctx = None
        for key, value in self.provide.items():
            ctx.bind(key, value)
        ctx.progress = self._progress
        return ctx.emit(EventType.STAGE_BEGAN, stage=type(self).__name__)

    def execute(self, ctx: Context) -> Signal:
        """Advance the current child by one unit of work.

        Args:
            ctx: Execution context for this entry.
        """
        while self._index < len(self.stages):
            child = self.stages[self._index]
            if self._child_ctx is None:
                self._child_ctx = ctx.child(self._index, child.name)
            if not self._entered:
                self._entered = True
                signal = ctx.topology.broadcast(child.enter(self._child_ctx))
                if signal.halts or signal is Signal.SKIP:
                    self._close(child)
                    continue
            signal = ctx.topology.broadcast(child.execute(self._child_ctx))
            if signal.halts:
                self._close(child)
            self._progress = self._progress.next_step()
            ctx.progress = self._progress
            if self._index >= len(self.stages):
                self._progress = self._progress.finish()
                return Signal.DONE
            return Signal.GO
        self._progress = self._progress.finish()
        ctx.progress = self._progress
        return Signal.DONE

    def _close(self, child: Stage) -> None:
        """Finish with the current child and move to the next.

        Args:
            child: The child being closed.
        """
        child.leave(self._child_ctx)
        self._child_ctx = None
        self._entered = False
        self._index += 1

    def leave(self, ctx: Context) -> Signal:
        """Close any child still open and announce that the phase is over.

        Args:
            ctx: Execution context for this entry.
        """
        if self._entered and self._index < len(self.stages):
            self._close(self.stages[self._index])
        ctx.progress = self._progress
        ctx.emit(EventType.STAGE_ENDED, stage=type(self).__name__)
        return Signal.GO

    def progress(self) -> Progress:
        """Return where the phase currently is in its own work."""
        return self._progress

    def walk(self, path: str | None = None) -> list[tuple[str, str]]:
        """Return this subtree as `(path, class name)` pairs.

        The manifest records it so a resume can refuse a reordered program.

        Args:
            path: Path of this phase; defaults to its own name.
        """
        here = path or self.name
        tree = [(here, type(self).__name__)]
        for index, child in enumerate(self.stages):
            child_path = f"{here}/{index}:{child.name}"
            if isinstance(child, Phase):
                tree.extend(child.walk(child_path))
            else:
                tree.append((child_path, type(child).__name__))
        return tree

    def state_dict(self) -> dict[str, Any]:
        """Return the phase's resumable state.

        Only the live child is recorded; finished ones are implied by `index`.
        """
        live = self._entered and self._index < len(self.stages)
        return {
            "progress": self._progress.to_dict(),
            "index": self._index,
            "entered": self._entered,
            "child": self.stages[self._index].state_dict() if live else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state previously returned by `state_dict`.

        Args:
            state: Mapping as returned by `state_dict`.
        """
        self._progress = Progress.from_dict(state.get("progress", {}))
        self._index = int(state.get("index", 0))
        self._entered = bool(state.get("entered", False))
        self._child_ctx = None
        child = state.get("child")
        if child is not None and self._index < len(self.stages):
            self.stages[self._index].load_state_dict(child)


class Program(Phase):
    """The root of a run."""

    def __init__(
        self,
        stages: Sequence[Stage] = (),
        *,
        name: str = "program",
        provide: Mapping[str, Any] | None = None,
        requires: Sequence[str] = (),
        provides: Sequence[str] = (),
    ):
        """Constructor.

        Args:
            stages: The children, run in order.
            name: Identifies the program; forms the root of every path.
            provide: Artifacts bound for the whole run, built once.
            requires: Binding names that must resolve before the run starts.
            provides: Binding names this program publishes.
        """
        super().__init__(
            name, stages, provide=provide, requires=requires, provides=provides
        )


class Repeat(Phase):
    """Run a stage a fixed number of times."""

    def __init__(self, times: int, stage: Stage, *, name: str | None = None):
        """Constructor.

        Args:
            times: How many times to run the child.
            stage: The child to repeat.
            name: Identifies the phase within its parent.

        Raises:
            ValueError: If `times` is not positive.
        """
        if times < 1:
            raise ValueError(f"Repeat needs a positive count, got {times!r}.")
        super().__init__(name, (stage,) * times)
        self.times = times
        self.stage = stage


class When(Phase):
    """Run a stage only when a predicate holds."""

    def __init__(
        self,
        predicate: Callable[[Context], bool],
        stage: Stage,
        *,
        name: str | None = None,
    ):
        """Constructor.

        Args:
            predicate: Called with the phase's context.
            stage: The child to run conditionally.
            name: Identifies the phase within its parent.
        """
        super().__init__(name, (stage,))
        self.predicate = predicate
        self.stage = stage

    def enter(self, ctx: Context) -> Signal:
        """Evaluate the predicate and skip the child when it is false.

        Args:
            ctx: Execution context for this entry.
        """
        signal = super().enter(ctx)
        if signal.halts:
            return signal
        if not ctx.topology.broadcast(bool(self.predicate(ctx))):
            self._index = len(self.stages)
            self._progress = self._progress.finish()
            return Signal.DONE
        return signal


class Every(Phase):
    """Run a stage on every nth visit."""

    def __init__(self, n: int, stage: Stage, *, name: str | None = None):
        """Constructor.

        Args:
            n: Run the child on every nth entry.
            stage: The child to run periodically.
            name: Identifies the phase within its parent.

        Raises:
            ValueError: If `n` is not positive.
        """
        if n < 1:
            raise ValueError(f"Every needs a positive interval, got {n!r}.")
        super().__init__(name, (stage,))
        self.n = n
        self.stage = stage
        self._visits = 0

    def enter(self, ctx: Context) -> Signal:
        """Count the visit and skip the child unless it is due.

        Args:
            ctx: Execution context for this entry.
        """
        signal = super().enter(ctx)
        self._visits += 1
        if signal.halts:
            return signal
        if self._visits % self.n != 0:
            self._index = len(self.stages)
            self._progress = self._progress.finish()
            return Signal.DONE
        return signal

    def state_dict(self) -> dict[str, Any]:
        """Return the phase's resumable state, including the visit count."""
        return {**super().state_dict(), "visits": self._visits}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state previously returned by `state_dict`.

        Args:
            state: Mapping as returned by `state_dict`.
        """
        super().load_state_dict(state)
        self._visits = int(state.get("visits", 0))


class StageLoop(ABC):
    """A stage that repeats a unit of work until it has done enough.

    One `execute` is one unit (for training, one optimizer step over
    `accumulate` micro-batches).

    Attributes:
        name: Identifies the stage within its parent.
        data: What the loop draws batches from, or the name of a binding.
        batch_size: Samples one forward sees, for a bare dataset.
        epochs: Passes over the data to make, or `None`.
        steps: Units of work to perform, or `None`.
        accumulate: Micro-batches consumed per unit.
        requires: Binding names that must resolve before the run starts.
        provides: Binding names this stage publishes.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        data: DataManager | Dataset | str | None = None,
        batch_size: int | None = None,
        epochs: int | None = None,
        steps: int | None = None,
        accumulate: int = 1,
        requires: Sequence[str] = (),
        provides: Sequence[str] = (),
    ):
        """Constructor.

        Args:
            name: Identifies the stage; defaults to the lowercased class name.
            data: A `DataManager`, a dataset, or the name of a binding.
            batch_size: Samples one forward sees, for a bare dataset. With
                `accumulate` it says what a step consumes; a built
                `DataManager` carries its own.
            epochs: Passes over the data to make; `None` leaves it to `steps`.
            steps: Units of work to perform; `None` leaves it to `epochs`.
            accumulate: Micro-batches consumed per unit of work.
            requires: Binding names that must resolve before the run starts.
            provides: Binding names this stage publishes.

        Raises:
            ValueError: If `accumulate` is not positive.
        """
        if accumulate < 1:
            raise ValueError(f"A step consumes at least one batch, got {accumulate!r}.")
        self.name = name or type(self).__name__.lower()
        self.data = data
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps = steps
        self.accumulate = accumulate
        self.requires = tuple(requires)
        self.provides = tuple(provides)
        self._progress = Progress()
        self._manager: DataManager | None = None
        self._world_size = 1
        self._iterator: Iterator[BatchType] | None = None

    def __repr__(self) -> str:
        """Return a short description of the stage."""
        return f"{type(self).__name__}({self.name!r})"

    @property
    def effective_batch(self) -> int | None:
        """Return the samples one unit of work consumes across every process.

        `None` until the stage is entered, since neither the batch size nor
        the number of processes is settled before then.
        """
        if self._manager is None:
            return None
        return self._manager.batch_size * self.accumulate * self._world_size

    @abstractmethod
    def core(self, batches: Sequence[BatchType], ctx: Context) -> Loss | None:
        """Perform one unit of work.

        Args:
            batches: The micro-batches making up this unit.
            ctx: Execution context for this entry.
        """

    def prepare(self, ctx: Context) -> None:
        """Build whatever the loop needs, once per entry.

        Args:
            ctx: Execution context for this entry.
        """

    def enter(self, ctx: Context) -> Signal:
        """Reset, build what the loop needs, and announce the stage.

        The first `execute` starts the epoch rather than this, so a
        checkpoint restored after `enter` decides where it begins.

        Args:
            ctx: Execution context for this entry.
        """
        self._progress = Progress()
        ctx.progress = self._progress
        dm_kwargs = {} if self.batch_size is None else {"batch_size": self.batch_size}
        self._manager = DataManager.from_source(ctx.resolve(self.data), **dm_kwargs)
        self._world_size = ctx.topology.world_size
        self._iterator = None
        self.prepare(ctx)
        return ctx.emit(EventType.STAGE_BEGAN, stage=type(self).__name__)

    def execute(self, ctx: Context) -> Signal:
        """Perform one unit of work, looping over epochs as they run out.

        Args:
            ctx: Execution context for this entry.
        """
        if self._iterator is None:
            self._draw_sweep(ctx)
        batches = [batch_to_device(b, ctx.device) for b in self._take()]
        if not batches:
            return self._next_sweep(ctx)
        loss = self.core(batches, ctx)
        samples = sum(samples_in_batch(batch) for batch in batches)
        self._progress = self._progress.next_step(samples)
        ctx.progress = self._progress
        reported = {} if loss is None else loss.as_floats()
        signal = ctx.emit(EventType.STEP_ENDED, **reported)
        if self.steps is not None and self._progress.global_step >= self.steps:
            self._finish(ctx)
            return Signal.DONE
        return signal

    def leave(self, ctx: Context) -> Signal:
        """Announce that the stage is over.

        Args:
            ctx: Execution context for this entry.
        """
        self._iterator = None
        ctx.progress = self._progress
        ctx.emit(EventType.STAGE_ENDED, stage=type(self).__name__)
        return Signal.GO

    def progress(self) -> Progress:
        """Return where the stage currently is in its own work."""
        return self._progress

    def state_dict(self) -> dict[str, Any]:
        """Return the stage's resumable state."""
        return {"progress": self._progress.to_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state previously returned by `state_dict`.

        Args:
            state: Mapping as returned by `state_dict`.
        """
        self._progress = Progress.from_dict(state.get("progress", {}))

    def _take(self) -> list[BatchType]:
        """Return the next unit's micro-batches, short at an epoch's end."""
        batches: list[BatchType] = []
        for _ in range(self.accumulate):
            batch = next(self._iterator, None)
            if batch is None:
                break
            batches.append(batch)
        return batches

    def _draw_sweep(self, ctx: Context) -> None:
        """Draw the batches one pass over the data will consume.

        Args:
            ctx: Execution context for this entry.
        """
        self._iterator = self._manager.iter(
            ctx,
            epoch=self._progress.epoch,
            seek=self._progress.step * self.accumulate,
        )
        ctx.emit(EventType.EPOCH_BEGAN, epoch=self._progress.epoch)

    def _next_sweep(self, ctx: Context) -> Signal:
        """Close the finished pass, then start the next one or stop.

        Args:
            ctx: Execution context for this entry.
        """
        ctx.emit(EventType.EPOCH_ENDED, epoch=self._progress.epoch)
        self._progress = self._progress.next_epoch()
        ctx.progress = self._progress
        done = self.epochs is not None and self._progress.epoch >= self.epochs
        if done or (self.epochs is None and self.steps is None):
            self._finish(ctx)
            return Signal.DONE
        self._draw_sweep(ctx)
        return Signal.GO

    def _finish(self, ctx: Context) -> None:
        """Mark the stage finished.

        Args:
            ctx: Execution context for this entry.
        """
        self._progress = self._progress.finish()
        ctx.progress = self._progress


class Train(StageLoop):
    """Updates a model's parameters against an objective.

    Attributes:
        model: Model, or the name of a binding holding one.
        objective: What produces the loss, or the terms it sums.
        optim: The optimizer spec, or one per update group.
        update: How the optimizers are stepped.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        model: nn.Module | str = "model",
        data: DataManager | Dataset | str | None = None,
        batch_size: int | None = None,
        loss: Callable[..., torch.Tensor] | None = None,
        objective: Objective | Sequence[Term] | str | None = None,
        optim: OptimSpec | Mapping[str, OptimSpec] | None = None,
        lr: float | None = None,
        update: Updater | None = None,
        ema: float | Ema | None = None,
        epochs: int | None = None,
        steps: int | None = None,
        accumulate: int = 1,
        clip: float | None = None,
        clip_mode: ClipTypes | None = None,
        reduction: ReductionTypes | None = None,
        precision: torch.dtype | None = None,
        requires: Sequence[str] = (),
        provides: Sequence[str] = (),
    ):
        """Constructor.

        Args:
            name: Identifies the stage; defaults to the lowercased class name.
            model: Model, or the name of a binding holding one.
            data: A `DataManager`, a dataset, or the name of a binding.
            batch_size: Samples one forward sees, for a bare dataset.
            loss: Compares the model's output to the batch's target, for the
                default objective.
            objective: What produces the loss: an object with `compute`, or a
                sequence of `Term`s to sum. Overrides `loss`.
            optim: The optimizer spec, or one per update group.
            lr: Learning rate, when no `optim` spells one out.
            update: How the optimizers are stepped; one `Step` by default.
            ema: Decay of an exponential moving average of the model,
                or a built one. Published as `"<model>/ema"` for later
                stages to evaluate or predict with.
            epochs: Passes over the data to make.
            steps: Optimizer steps to take.
            accumulate: Micro-batches per optimizer step, weighted by their
                sample counts.
            clip: Threshold gradients are clipped to, or `None`.
            clip_mode: Whether `clip` bounds the gradient norm or each value;
                `"norm"` when absent.
            reduction: How the objective reduced over its batch; `"mean"`
                when absent.
            precision: Precision the forward runs in, or `None` for full precision.
            requires: Binding names that must resolve before the run starts.
            provides: Binding names this stage publishes.

        Raises:
            ValueError: If neither `loss` nor `objective` says what to
                compute, or an update is given alongside the settings that
                would configure one.
        """
        if loss is None and objective is None:
            raise ValueError(
                f"Train({name!r}) has neither a loss nor an objective, so "
                "there is nothing to minimise."
            )
        super().__init__(
            name,
            data=data,
            batch_size=batch_size,
            epochs=epochs,
            steps=steps,
            accumulate=accumulate,
            requires=requires,
            provides=provides,
        )
        self.model = model
        self.loss = loss
        self.objective = objective
        self.optim = optim
        self.lr = lr
        if update is not None:
            configured = {
                name
                for name, value in (
                    ("clip", clip),
                    ("clip_mode", clip_mode),
                    ("reduction", reduction),
                    ("precision", precision),
                )
                if value is not None
            }
            if configured:
                raise ValueError(
                    f"{sorted(configured)} configure an update, but {update!r} "
                    "was given and carries its own; set them in one place."
                )
        self.policy = UpdatePolicy(
            clip=clip,
            clip_mode="norm" if clip_mode is None else clip_mode,
            reduction="mean" if reduction is None else reduction,
        )
        self.precision = precision
        self.update = update or Step(policy=self.policy, precision=precision)
        self.ema = ema
        self._averaged: Ema | None = None
        self._objective: Objective | None = None

    def _build_objective(self, ctx: Context) -> Objective:
        """Return what this stage optimizes.

        Args:
            ctx: Execution context for this entry.
        """
        if self.objective is None:
            return Criterion(self.loss, model=self.model)
        resolved = ctx.resolve(self.objective)
        if isinstance(resolved, Sequence) and not isinstance(resolved, (str, bytes)):
            return CompositeObjective(resolved, model=self.model)
        return resolved

    def prepare(self, ctx: Context) -> None:
        """Build the objective and bind the optimizers to the update.

        Args:
            ctx: Execution context for this entry.
        """
        self._objective = self._build_objective(ctx)
        self.update.balance_ranks = bool(getattr(self._manager, "balance_ranks", False))
        self.update.bind(self._optimizers(ctx))
        self._averaged = self._build_ema(ctx)

    def core(self, batches: Sequence[BatchType], ctx: Context) -> Loss | None:
        """Take one optimizer step over the micro-batches.

        Args:
            batches: The micro-batches making up this step.
            ctx: Execution context for this entry.
        """
        ctx.clear_cache()
        loss = self.update.apply(self._objective, batches, ctx)
        if self._averaged is not None:
            self._averaged.update_parameters(ctx.resolve(self.model))
        return loss

    def _build_ema(self, ctx: Context) -> Ema | None:
        """Build the moving average this stage keeps, and publish it.

        Args:
            ctx: Execution context for this entry.
        """
        if self.ema is None:
            return None
        model = ctx.resolve(self.model)
        averaged = self.ema if isinstance(self.ema, Ema) else Ema(model, self.ema)
        target = self.model if isinstance(self.model, str) else "model"
        ctx.publish(f"{target}/ema", averaged)
        return averaged

    def _params(self, spec: OptimSpec, ctx: Context) -> ParamsT:
        """Return the trainable parameters an optimizer spec selects.

        Frozen parameters are left out, so an optimizer never holds state for
        weights it cannot move.

        Args:
            spec: Says which bindings the optimizer owns.
            ctx: Execution context for this entry.

        Raises:
            ValueError: If nothing it selects can be trained.
        """
        chosen = spec.params if spec.params is not None else self.model
        if isinstance(chosen, nn.Module):
            found = list(chosen.parameters())
        elif callable(chosen) and not isinstance(chosen, str):
            found = list(chosen(ctx))
        else:
            names = [chosen] if isinstance(chosen, str) else list(chosen)
            found = [p for name in names for p in ctx.resolve(name).parameters()]
        params = [p for p in found if p.requires_grad]
        if not params:
            raise ValueError(
                f"Train({self.name!r}) selects {len(found)} parameter(s), none "
                "of them trainable; a frozen model has nothing to optimise."
            )
        return params

    def _optimizers(self, ctx: Context) -> dict[str | None, Optimizer]:
        """Build one optimizer per update group.

        Args:
            ctx: Execution context for this entry.

        Raises:
            ValueError: If the optimizers do not cover the update's groups.
        """
        groups = self.update.groups()
        if not isinstance(self.optim, Mapping):
            spec = self.optim or OptimSpec.adamw(lr=self.lr if self.lr else 1e-4)
            if self.lr is not None and self.optim is not None:
                spec = replace(spec, lr=self.lr)
            return {group: spec.build(self._params(spec, ctx)) for group in groups}
        missing = sorted(str(g) for g in groups if g not in self.optim)
        if missing:
            raise ValueError(
                f"Train({self.name!r}) has no optimizer for groups {missing}; "
                f"it was given {sorted(self.optim)}."
            )
        selected = {group: self._params(self.optim[group], ctx) for group in groups}
        disjoint = disjoint_params(selected)
        return {group: self.optim[group].build(disjoint[group]) for group in groups}

    def state_dict(self) -> dict[str, Any]:
        """Return the stage's resumable state."""
        state = {**super().state_dict(), **self.update.state_dict()}
        if self._averaged is not None:
            state["ema"] = self._averaged.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state previously returned by `state_dict`.

        Args:
            state: Mapping as returned by `state_dict`.
        """
        super().load_state_dict(state)
        self.update.load_state_dict(state)
        if self._averaged is not None and "ema" in state:
            self._averaged.load_state_dict(state["ema"])


Finetune = partialclass(
    "Finetune",
    Train,
    lr=1e-5,
    __doc__="A `Train` preset that refines an already-trained model gently.",
)


class Inference(StageLoop):
    """A loop that reads a model without changing it.

    Attributes:
        model: Model, or the name of a binding holding one.
        weights: Which parameters to read: `"model"` or `"ema"`.
        inputs: Key the model's input is read from.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        model: nn.Module | str = "model",
        data: DataManager | Dataset | str | None = None,
        batch_size: int | None = None,
        weights: WeightsTypes = "model",
        inputs: str = "x",
        epochs: int | None = None,
        steps: int | None = None,
        requires: Sequence[str] = (),
        provides: Sequence[str] = (),
    ):
        """Constructor.

        Args:
            name: Identifies the stage; defaults to the lowercased class name.
            model: Model, or the name of a binding holding one.
            data: A `DataManager`, a dataset, or the name of a binding.
            batch_size: Samples one forward sees, for a bare dataset.
            weights: Which parameters to read: the model's own, or the moving
                average a `Train` published beside it.
            inputs: Key the model's input is read from, for mapping batches.
            epochs: Passes over the data; one when absent.
            steps: Units of work; the whole pass when absent.
            requires: Binding names that must resolve before the run starts.
            provides: Binding names this stage publishes.

        Raises:
            ValueError: If `weights` names neither.
        """
        require(weights, ("model", "ema"), "weights")
        super().__init__(
            name,
            data=data,
            batch_size=batch_size,
            epochs=epochs,
            steps=steps,
            requires=requires,
            provides=provides,
        )
        self.model = model
        self.weights = weights
        self.inputs = inputs

    def resolved_model(self, ctx: Context) -> nn.Module:
        """Return the module whose parameters this stage reads.

        Args:
            ctx: Execution context for this entry.
        """
        if self.weights == "model":
            return ctx.resolve(self.model)
        target = self.model if isinstance(self.model, str) else "model"
        return ctx[f"{target}/ema"]


class Eval(Inference):
    """Accumulates metrics over a pass, publishing each when it ends.

    Attributes:
        metrics: What to accumulate, keyed by the name each publishes under.
        targets: Key the target is read from.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        metrics: Sequence[Any] | Mapping[str, Any] = (),
        targets: str = "y",
        **kwargs: Any,
    ):
        """Constructor.

        Args:
            name: Identifies the stage; defaults to the lowercased class name.
            metrics: What to accumulate, either named or keyed by their own
                lowercased class names.
            targets: Key the target is read from, for mapping batches.
            kwargs: Passed to `Inference`.

        Raises:
            ValueError: If no metric was given.
        """
        super().__init__(name, **kwargs)
        self.metrics = (
            dict(metrics)
            if isinstance(metrics, Mapping)
            else {type(m).__name__.lower(): m for m in metrics}
        )
        if not self.metrics:
            raise ValueError(f"Eval({self.name!r}) was given no metric to compute.")
        self.targets = targets

    def prepare(self, ctx: Context) -> None:
        """Move the metrics to the device and clear what they hold.

        Args:
            ctx: Execution context for this entry.
        """
        for metric in self.metrics.values():
            metric.to(ctx.device)
            metric.reset()

    def core(self, batches: Sequence[BatchType], ctx: Context) -> Loss | None:
        """Update every metric from the batches, changing nothing.

        Args:
            batches: The micro-batches making up this unit.
            ctx: Execution context for this entry.
        """
        model = self.resolved_model(ctx)
        with torch.inference_mode():
            for batch in batches:
                inputs, targets = unpack_batch(
                    batch, self.inputs, self.targets, reader=type(self).__name__
                )
                prediction = model(inputs)
                for metric in self.metrics.values():
                    metric.update(prediction, targets)
        return None

    def leave(self, ctx: Context) -> Signal:
        """Publish what each metric computed, for later siblings to read.

        Args:
            ctx: Execution context for this entry.
        """
        for key, metric in self.metrics.items():
            value = metric.compute()
            if value is not None:
                ctx.publish(f"{self.name}/{key}", float(value))
        return super().leave(ctx)

    def state_dict(self) -> dict[str, Any]:
        """Return the stage's resumable state, metrics included."""
        state = super().state_dict()
        state["metrics"] = {
            key: metric.state_dict()
            for key, metric in self.metrics.items()
            if isinstance(metric, Stateful)
        }
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state previously returned by `state_dict`.

        Args:
            state: Mapping as returned by `state_dict`.
        """
        super().load_state_dict(state)
        for key, saved in state.get("metrics", {}).items():
            metric = self.metrics.get(key)
            if isinstance(metric, Stateful):
                metric.load_state_dict(saved)


class Predict(Inference):
    """Runs a model over a pass and writes what it produced.

    Attributes:
        archive: File the predictions are written to, or `None` to keep
            them in memory only.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        archive: str | Path | None = None,
        key: str = "data",
        **kwargs: Any,
    ):
        """Constructor.

        Args:
            name: Identifies the stage; defaults to the lowercased class name.
            archive: File the predictions are written to; its suffix picks the
                format. Kept in memory when absent.
            key: Name the predictions are stored under, for formats that key.
            kwargs: Passed to `Inference`.
        """
        super().__init__(name, **kwargs)
        self.archive = Path(archive) if archive is not None else None
        self.key = key
        self._predictions: list[torch.Tensor] = []

    def prepare(self, ctx: Context) -> None:
        """Drop anything kept from an earlier entry.

        Args:
            ctx: Execution context for this entry.
        """
        self._predictions = []

    @property
    def predictions(self) -> torch.Tensor | None:
        """Return what the stage produced, or `None` before it has run."""
        if not self._predictions:
            return None
        return torch.cat(self._predictions)

    def core(self, batches: Sequence[BatchType], ctx: Context) -> Loss | None:
        """Run the model and keep what it produced.

        Args:
            batches: The micro-batches making up this unit.
            ctx: Execution context for this entry.
        """
        model = self.resolved_model(ctx)
        with torch.inference_mode():
            for batch in batches:
                prediction = model(input_in_batch(batch, self.inputs))
                self._predictions.append(prediction.detach().cpu().clone())
        return None

    def leave(self, ctx: Context) -> Signal:
        """Write the predictions out, and publish them for later siblings.

        Args:
            ctx: Execution context for this entry.
        """
        produced = self.predictions
        if produced is not None:
            if self.archive is not None and ctx.topology.is_main:
                save_dataset(self.archive, produced, key=self.key)
            ctx.publish(f"{self.name}/predictions", produced)
        return super().leave(ctx)
