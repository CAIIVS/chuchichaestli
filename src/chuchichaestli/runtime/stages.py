# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""The stage register: blocks that run once, and phases that hold stages."""

from __future__ import annotations
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from dataclasses import replace
from typing import Any
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import ParamsT
from torch.utils.data import Dataset
from chuchichaestli.data.archive import Archive, archive_for
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
from chuchichaestli.runtime.update import (
    Step,
    SwaWindow,
    Updater,
    WeightsTypes,
)
from chuchichaestli.runtime.events import EventType, Progress, Signal
from chuchichaestli.runtime.topology import lockstep, reduce_metrics
from chuchichaestli.runtime.traits import Objective, Stage, Stateful
from chuchichaestli.training.objective import Loss, Term
from chuchichaestli.training.optim import OptimSpec, disjoint_params
from chuchichaestli.training.update import (
    ClipTypes,
    Ema,
    ReductionTypes,
    UpdatePolicy,
    average_targets,
)
from chuchichaestli.utils.functools import partialclass
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
        self._restored: dict[str, Any] | None = None

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
        self._restored = None
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
                signal = lockstep(ctx.topology, lambda: child.enter(self._child_ctx))
                restored, self._restored = self._restored, None
                if signal.halts or signal is Signal.SKIP:
                    self._close(child)
                    continue
                if restored is not None:
                    child.load_state_dict(restored)
            signal = lockstep(ctx.topology, lambda: child.execute(self._child_ctx))
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
            self.stages[self._index].leave(self._child_ctx)
            self._child_ctx = None
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
        self._entered = False
        self._child_ctx = None
        child = state.get("child")
        self._restored = child if self._index < len(self.stages) else None


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

    @property
    def model_binding(self) -> str:
        """Name the model is bound under.

        A model handed over outright is bound under the default name rather
        than one of its own.
        """
        return self.model if isinstance(self.model, str) else "model"

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
        ctx.progress = self._progress
        ctx.emit(EventType.EPOCH_BEGAN, epoch=self._progress.epoch)

    def finalize_sweep(self, ctx: Context) -> None:
        """React to a pass over the data finishing.

        Args:
            ctx: Execution context for this entry.
        """

    def _next_sweep(self, ctx: Context) -> Signal:
        """Close the finished pass, then start the next one or stop.

        Args:
            ctx: Execution context for this entry.
        """
        self.finalize_sweep(ctx)
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
        swa_window: The tail of the run over which weights are averaged.
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
        ema: float | Ema | Mapping[str, float | Ema] | None = None,
        swa: bool | str | Sequence[str] | None = None,
        swa_start: float = 0.75,
        swa_lr: float | None = None,
        swa_anneal: int = 10,
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
            ema: Decay of an exponential moving average, a built one, or one
                per binding name. Each is published as `"<binding>/ema"` for
                later stages to evaluate or predict with.
            swa: Whether to average weights equally over the tail of the run,
                or the bindings to average. Published as `"<binding>/swa"`.
            swa_start: Fraction of the epochs after which averaging begins;
                averaging from the start would drag the untrained beginning
                into an equally weighted mean.
            swa_lr: Rate to hold while averaging. Setting it hands the
                schedule over to `SWALR` at `swa_start` and stops whatever
                schedule ran before, as torch's recipe does.
            swa_anneal: Number of passes `SWALR` takes to reach `swa_lr`.
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
                compute, an update is given alongside the settings that would
                configure one, `swa_start` lies outside `[0, 1)`, or `swa` is
                asked for without the epochs it averages over.
        """
        if swa is not None and epochs is None:
            raise ValueError(
                f"Train({name!r}) averages weights once a pass, so swa "
                "needs epochs= to average over."
            )
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
        self.swa_window = SwaWindow(swa, swa_start, swa_lr, swa_anneal)
        self._ema: dict[str, Ema] = {}
        self._sweepwise_schedulers: dict[str | None, LRScheduler] = {}
        self._specs: dict[str | None, OptimSpec] = {}
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

        The objective is placed on the device here rather than by the
        runtime, since it is built per entry and belongs to this stage alone.

        Args:
            ctx: Execution context for this entry.
        """
        if not isinstance(self.model, str):
            ctx.bind("model", self.model)
        self._objective = self._build_objective(ctx)
        if isinstance(self._objective, nn.Module):
            self._objective.to(ctx.device)
        self.update.balance_ranks = bool(getattr(self._manager, "balance_ranks", False))
        optimizers = self._optimizers(ctx)
        stepwise_schedulers, self._sweepwise_schedulers = self._schedulers(optimizers)
        self.update.bind(optimizers, stepwise_schedulers)
        self._ema = self._build_ema(ctx)
        self.swa_window.build(self.model_binding, optimizers, ctx)

    def core(self, batches: Sequence[BatchType], ctx: Context) -> Loss | None:
        """Take one optimizer step over the micro-batches.

        Args:
            batches: The micro-batches making up this step.
            ctx: Execution context for this entry.
        """
        ctx.clear_cache()
        loss = self.update.apply(self._objective, batches, ctx)
        for name, averaged in self._ema.items():
            averaged.update_parameters(ctx.resolve(name))
        return loss

    def finalize_sweep(self, ctx: Context) -> None:
        """Run actions at the end of a pass, for instance updating schedulers.

        Args:
            ctx: Execution context for this entry.
        """
        if self.swa_window.is_open(self._progress.epoch, self.epochs):
            self.swa_window.hand_over(
                self._sweepwise_schedulers, self.update.schedulers
            )
            self.swa_window.take(ctx)
        for scheduler in self._sweepwise_schedulers.values():
            scheduler.step()

    def _build_ema(self, ctx: Context) -> dict[str, Ema]:
        """Build every moving average this stage keeps, and publish them.

        Args:
            ctx: Execution context for this entry.
        """
        built: dict[str, Ema] = {}
        for name, decay in average_targets(self.ema, self.model_binding).items():
            averaged = (
                decay if isinstance(decay, Ema) else Ema(ctx.resolve(name), decay)
            )
            ctx.publish(f"{name}/ema", averaged)
            built[name] = averaged
        return built

    def _params(self, spec: OptimSpec, ctx: Context) -> ParamsT:
        """Return the trainable parameters an optimizer spec selects.

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
        self._specs = self._optim_specs(groups)
        if not isinstance(self.optim, Mapping):
            spec = self._specs[groups[0]]
            return {group: spec.build(self._params(spec, ctx)) for group in groups}
        selected = {group: self._params(self._specs[group], ctx) for group in groups}
        disjoint = disjoint_params(selected)
        return {group: self._specs[group].build(disjoint[group]) for group in groups}

    def _optim_specs(self, groups: Sequence[str | None]) -> dict[str | None, OptimSpec]:
        """Return the optimizer spec of every update group.

        Args:
            groups: The update groups to cover.

        Raises:
            ValueError: If the optimizers do not cover the update's groups.
        """
        if not isinstance(self.optim, Mapping):
            spec = self.optim or OptimSpec.adamw(lr=self.lr if self.lr else 1e-4)
            if self.lr is not None and self.optim is not None:
                spec = replace(spec, lr=self.lr)
            return dict.fromkeys(groups, spec)
        missing = sorted(str(g) for g in groups if g not in self.optim)
        if missing:
            raise ValueError(
                f"Train({self.name!r}) has no optimizer for groups {missing}; "
                f"it was given {sorted(self.optim)}."
            )
        return {group: self.optim[group] for group in groups}

    def _schedulers(
        self, optimizers: Mapping[str | None, Optimizer]
    ) -> tuple[dict[str | None, LRScheduler], dict[str | None, LRScheduler]]:
        """Return each group's scheduler, split by how often it is stepped.

        The first advance once per optimizer step, the second once per pass.

        Args:
            optimizers: The optimizer of each group.
        """
        stepwise_schedulers: dict[str | None, LRScheduler] = {}
        sweepwise_schedulers: dict[str | None, LRScheduler] = {}
        for group, optimizer in optimizers.items():
            spec = self._specs[group].scheduler
            if spec is None:
                continue
            landing = (
                stepwise_schedulers if spec.interval == "step" else sweepwise_schedulers
            )
            landing[group] = spec.build(optimizer)
        return stepwise_schedulers, sweepwise_schedulers

    def leave(self, ctx: Context) -> Signal:
        """Refresh any averaged batch statistics, then end the stage.

        Args:
            ctx: Execution context for this entry.
        """
        self.swa_window.refresh_batch_stats(
            lambda: (
                input_in_batch(batch, getattr(self._objective, "inputs", "x"))
                for batch in self._manager.iter(ctx, epoch=0)
            ),
            device=ctx.device,
        )
        return super().leave(ctx)

    def state_dict(self) -> dict[str, Any]:
        """Return the stage's resumable state."""
        state = {**super().state_dict(), **self.update.state_dict()}
        state.update({f"ema/{n}": a.state_dict() for n, a in self._ema.items()})
        state.update(self.swa_window.state_dict())
        state.update(
            {
                f"sched/{g}": s.state_dict()
                for g, s in self._sweepwise_schedulers.items()
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state previously returned by `state_dict`.

        Args:
            state: Mapping as returned by `state_dict`.
        """
        super().load_state_dict(state)
        self.update.load_state_dict(state)
        for name, averaged in self._ema.items():
            saved = state.get(f"ema/{name}")
            if saved is not None:
                averaged.load_state_dict(saved)
        self.swa_window.load_state_dict(state)
        for group, scheduler in self._sweepwise_schedulers.items():
            saved = state.get(f"sched/{group}")
            if saved is not None:
                scheduler.load_state_dict(saved)


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
            weights: Which parameters to read: the model's own, or an
                average a `Train` published beside it.
            inputs: Key the model's input is read from, for mapping batches.
            epochs: Passes over the data; one when absent.
            steps: Units of work; the whole pass when absent.
            requires: Binding names that must resolve before the run starts.
            provides: Binding names this stage publishes.

        Raises:
            ValueError: If `weights` names neither.
        """
        require(weights, ("model", "ema", "swa"), "weights")
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
        return ctx[f"{self.model_binding}/{self.weights}"]


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

        Every process sees the same value, since a later `When` reading one
        must reach the same decision on all of them.

        Args:
            ctx: Execution context for this entry.
        """
        reduce_metrics(self.metrics.values(), ctx.topology)
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

    Given an `archive` the predictions stream to it and are not kept; without
    one they are kept in memory and published for later siblings.

    Attributes:
        archive: File the predictions are written to, or `None` to keep them
            in memory only.
        key: Name the predictions are stored under.
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
            key: Name the predictions are stored under, for formats that name.
            kwargs: Passed to `Inference`.
        """
        super().__init__(name, **kwargs)
        self.archive = Path(archive) if archive is not None else None
        self.key = key
        self._writer: Archive | None = None
        self._predictions: list[torch.Tensor] = []

    def prepare(self, ctx: Context) -> None:
        """Open the archive, or clear what an earlier entry kept.

        Args:
            ctx: Execution context for this entry.
        """
        self._predictions = []
        self._writer = None
        if self.archive is None:
            return
        if ctx.topology.world_size > 1:
            warnings.warn(
                f"Predict({self.name!r}) writes only what rank 0 produced; "
                "the other processes hold the rest.",
                stacklevel=2,
            )
        if ctx.topology.is_main:
            self._writer = archive_for(self.archive, self.key)

    @property
    def predictions(self) -> torch.Tensor | None:
        """Return what the stage kept, or `None` when it streamed instead."""
        if not self._predictions:
            return None
        return torch.cat(self._predictions)

    def core(self, batches: Sequence[BatchType], ctx: Context) -> Loss | None:
        """Run the model, writing or keeping what it produced.

        Args:
            batches: The micro-batches making up this unit.
            ctx: Execution context for this entry.
        """
        model = self.resolved_model(ctx)
        with torch.inference_mode():
            for batch in batches:
                y = model(input_in_batch(batch, self.inputs))
                y = y.detach().cpu().clone()
                if self._writer is not None:
                    self._writer.write(y)
                elif self.archive is None:
                    self._predictions.append(y)
        return None

    def leave(self, ctx: Context) -> Signal:
        """Finish the archive, or publish what was kept.

        Args:
            ctx: Execution context for this entry.
        """
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self.archive is not None:
            ctx.publish(f"{self.name}/archive", self.archive)
        else:
            y = self.predictions
            if y is not None:
                ctx.publish(f"{self.name}/predictions", y)
        return super().leave(ctx)
