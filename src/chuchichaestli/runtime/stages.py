# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""The stage register: blocks that run once, and phases that hold stages."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from chuchichaestli.runtime.context import Context
from chuchichaestli.runtime.events import EventType, Progress, Signal
from chuchichaestli.runtime.traits import Stage
from chuchichaestli.utils.io import read_state, staged, writer_for


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
