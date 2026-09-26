# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Observers of runtime events: modular methods for monitoring purposes."""

from __future__ import annotations
import json
import math
import signal
import sys
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TextIO, get_args
from chuchichaestli.utils.ansi import (
    ANSIShade,
    Pinned,
    ansi_supported,
    cli_pbar,
    paint,
)
from chuchichaestli.runtime.ckpt import CheckpointFormats, CheckpointStore
from chuchichaestli.runtime.events import (
    C3liRuntimeError,
    Event,
    EventType,
    Progress,
    Signal,
    counts_toward,
    units,
)
from chuchichaestli.utils.registry import require

if TYPE_CHECKING:
    from chuchichaestli.runtime.context import Context
    from chuchichaestli.runtime.runtime import Runtime


ModeTypes = Literal["min", "max"]
ThresholdModeTypes = Literal["rel", "abs"]
CheckpointUnitTypes = Literal["advance", "epoch", "step"]
BarUnitTypes = Literal["step", "epoch"]

MODES: frozenset[str] = frozenset(get_args(ModeTypes))
THRESHOLD_MODES: frozenset[str] = frozenset(get_args(ThresholdModeTypes))
CHECKPOINT_UNIT_MAP: dict[str, EventType] = units(CheckpointUnitTypes)
BAR_UNIT_MAP: dict[str, EventType] = units(BarUnitTypes)

__all__ = [
    "ProgressBar",
    "Console",
    "Checkpointer",
    "Jsonl",
    "Timer",
    "EarlyStop",
    "GracefulStop",
    "ModeTypes",
    "ThresholdModeTypes",
    "CheckpointUnitTypes",
    "BarUnitTypes",
]


class Console:
    """Print a human-readable line as the run proceeds."""

    RENDERERS: ClassVar[dict[EventType, str]] = {
        EventType.STAGE_BEGAN: "_stage_began",
        EventType.STAGE_ENDED: "_stage_ended",
        EventType.STEP_ENDED: "_step_ended",
        EventType.EPOCH_ENDED: "_epoch_ended",
        EventType.CHECKPOINT: "_checkpoint",
        EventType.RUN_ENDED: "_run_ended",
    }

    def __init__(
        self,
        every: int = 1,
        stream: TextIO | None = None,
        bar_length: int = 24,
        color: bool | None = None,
        timing: bool = False,
    ):
        """Constructor.

        Args:
            every: Print a step line every nth step.
            stream: Where to write; defaults to stdout.
            bar_length: Width of the progress bar when a total is known.
            color: Force colour on or off; detected from the stream when
                `None`, so a redirected log never receives escape codes.
            timing: Whether to report pass timings, measured by a `Timer`.
        """
        self.every = max(1, every)
        self.stream = stream if stream is not None else sys.stdout
        self.bar_length = bar_length
        self.color = color
        self.pinned: Pinned | None = None
        self.timer = Timer(report=False) if timing else None

    def __repr__(self) -> str:
        """Return a short description of the hook."""
        every, color = self.every, self.colored
        return f"Console({every=}, {color=})"

    @property
    def colored(self) -> bool:
        """Whether this reporter writes in colour."""
        return ansi_supported(self.stream) if self.color is None else self.color

    def _stage_began(self, event: Event, tint: bool) -> str | None:
        """Render a stage opening.

        Args:
            event: The event to render.
            tint: Whether to colour the line.
        """
        return paint(f"> {event.path}", ANSIShade.CYAN, on=tint)

    def _stage_ended(self, event: Event, tint: bool) -> str | None:
        """Render a stage closing, and whatever it reported as it did.

        Args:
            event: The event to render.
            tint: Whether to colour the line.
        """
        line = paint(f"< {event.path}", ANSIShade.GREEN, on=tint)
        return f"{line} {self._numbers(event, tint)}".rstrip()

    def _checkpoint(self, event: Event, tint: bool) -> str | None:
        """Render a checkpoint having been written.

        Args:
            event: The event to render.
            tint: Whether to colour the line.
        """
        return paint(
            f"  saved {event.payload.get('path', '')}", ANSIShade.YELLOW, on=tint
        )

    def _epoch_ended(self, event: Event, tint: bool) -> str | None:
        """Render a pass over the data having finished, and what it took.

        Args:
            event: The event to render.
            tint: Whether to colour the line.
        """
        if self.timer is None or event.progress.epoch % self.every:
            return None
        taken = self.timer.passes.get(event.path)
        if not taken:
            return None
        pass_no = event.payload.get("epoch", len(taken) - 1)
        return paint(
            f"  {event.path} pass {pass_no} in {taken[-1]:.3f}s",
            ANSIShade.DIM,
            on=tint,
        )

    def _run_ended(self, event: Event, tint: bool) -> str | None:
        """Render why a run stopped, or nothing when it simply finished.

        Args:
            event: The event to render.
            tint: Whether to colour the line.
        """
        aborted = event.payload.get("aborted")
        if not aborted:
            return None
        return paint(f"! {aborted}", ANSIShade.RED, on=tint)

    def _numbers(self, event: Event, tint: bool) -> str:
        """Render an event's numeric payload as `key=value` pairs.

        Args:
            event: The event to read.
            tint: Whether to colour the values.
        """
        return " ".join(
            f"{k}={paint(f'{v:.4g}', ANSIShade.BOLD, on=tint)}"
            if isinstance(v, float)
            else f"{k}={v}"
            for k, v in event.payload.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "total"
        )

    def _step_ended(self, event: Event, tint: bool) -> str | None:
        """Render a step, every nth one, with a bar when a total is known.

        Args:
            event: The event to render.
            tint: Whether to colour the line.
        """
        if event.progress.global_step % self.every:
            return None
        detail = self._numbers(event, tint)
        prefix = paint(
            f"  {event.path} step {event.progress.global_step}", ANSIShade.DIM, on=tint
        )
        return f"{prefix} {detail}".rstrip()

    def on(self, event: Event) -> Signal:
        """Print the event, when it is one this reporter shows.

        Args:
            event: What the runtime just did.
        """
        if self.timer is not None:
            self.timer.on(event)
        name = self.RENDERERS.get(event.type)
        if name is None:
            return Signal.GO
        line = getattr(self, name)(event, self.colored)
        if line is None:
            return Signal.GO
        if self.pinned is not None:
            self.pinned.scroll(line)
        else:
            print(line, file=self.stream, flush=True)
        return Signal.GO


class Jsonl:
    """Append every event to a JSON Lines file, recording the run's trace.

    The file goes beside the run's checkpoints by default, so a trace and the
    state it describes are found together and a second run cannot overwrite
    the first unless it was given the same store.

    Attributes:
        name: What the file is called within the store.
        store: Directory it is written to.
        needs_store: Whether the runtime must be given a store, which it need
            not be when this hook was told where to write.
    """

    needs_store: bool = True

    def __init__(
        self,
        name: str = "trace.jsonl",
        store: str | Path | None = None,
        only: Sequence[EventType] | None = None,
        flush_every: int = 1,
    ):
        """Constructor.

        Args:
            name: File to append to within the store. A name, not a path:
                where a run records itself is the store's to decide.
            store: Directory to write into, overriding the run's own. Given
                one, this hook needs nothing from the runtime.
            only: Record just these event types, or all of them when `None`.
            flush_every: Flush periodicity. Flushing costs almost nothing;
                so raise this only for a trace busy enough for that to matter,
                e.g. micro-batching.

        Raises:
            ValueError: If `flush_every` is not positive, or `name` is a path.
        """
        if flush_every < 1:
            raise ValueError(
                f"Jsonl needs a positive flush interval, got {flush_every!r}."
            )
        if Path(name).name != str(name):
            raise ValueError(
                f"Jsonl takes a name within the run's store, got {name!r}. "
                "Pass store= to the runtime to say where it goes."
            )
        self.name = str(name)
        self.only = frozenset(only) if only is not None else None
        self.flush_every = flush_every
        self.store = Path(store) if store is not None else None
        self.needs_store = self.store is None
        self._write_counter = 0
        self._handle: TextIO | None = None

    @property
    def path(self) -> Path:
        """Return the file this hook appends to."""
        return (self.store or Path()) / self.name

    def attach(self, runtime: Runtime, ctx: Context) -> None:
        """Receive the run whose trace this hook records.

        Args:
            runtime: The engine executing the program.
            ctx: Root context of the run.
        """
        if self.store is None:
            self.store = runtime.store

    def __repr__(self) -> str:
        """Return a short description of the hook."""
        path, flush_every = str(self.path), self.flush_every
        return f"Jsonl({path!r}, {flush_every=})"

    def on(self, event: Event) -> Signal:
        """Write the event as one JSON object on its own line.

        Args:
            event: What the runtime just did.
        """
        if self.only is not None and event.type not in self.only:
            return Signal.GO
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        self._handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        self._write_counter += 1
        if self._write_counter % self.flush_every == 0:
            self._handle.flush()
        if event.type is EventType.RUN_ENDED:
            self.close()
        return Signal.GO

    def close(self) -> None:
        """Close the file if it is open."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    @staticmethod
    def read(path: str | Path, only: Sequence[EventType] | None = None) -> list[Event]:
        """Return the events a trace recorded.

        Args:
            path: Trace file to read, as this hook wrote it.
            only: Read just these event types, or all of them when `None`.

        Raises:
            FileNotFoundError: If there is no trace at the path.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No trace to read at {str(path)!r}.")
        kinds = frozenset(only) if only is not None else None
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            event = Event.from_dict(json.loads(line))
            if kinds is None or event.type in kinds:
                events.append(event)
        return events


class Timer:
    """Measure how long stages take, and report once at they end.

    Attributes:
        elapsed: Seconds spent in each stage, summed over its entries.
        passes: Seconds each pass over the data took, by stage.
        depth: How far below the root to report, or `Nonfe` for every stage.
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        report: bool = True,
        depth: int | None = None,
    ):
        """Constructor.

        Args:
            stream: Where to write the summary; defaults to stderr.
            report: Whether to write it at all. `Console` measures through a
                timer of its own and renders the result in its own style, so
                it asks for one that stays quiet.
            depth: How far below the root to report. `0` is the run as a whole,
                `1` reports at the root's children stage granularity, etc.
        """
        self.stream = stream if stream is not None else sys.stderr
        self.report = report
        self.depth = depth
        self.elapsed: dict[str, float] = {}
        self.passes: dict[str, list[float]] = {}
        self._started: dict[str, float] = {}
        self._opened: dict[str, float] = {}

    def __repr__(self) -> str:
        """Return a short description of the hook."""
        return f"Timer({len(self.elapsed)} stages)"

    def on(self, event: Event) -> Signal:
        """Accumulate stage durations, and print a summary when the stage ends.

        Args:
            event: What the runtime just did.
        """
        if event.type is EventType.STAGE_BEGAN:
            self._started[event.path] = time.perf_counter()
        elif event.type is EventType.EPOCH_BEGAN:
            self._opened[event.path] = time.perf_counter()
        elif event.type is EventType.EPOCH_ENDED:
            opened = self._opened.pop(event.path, None)
            if opened is not None:
                taken = time.perf_counter() - opened
                self.passes.setdefault(event.path, []).append(taken)
        elif event.type is EventType.STAGE_ENDED:
            started = self._started.pop(event.path, None)
            if started is not None:
                taken = time.perf_counter() - started
                self.elapsed[event.path] = self.elapsed.get(event.path, 0.0) + taken
        elif event.type is EventType.RUN_ENDED and self.report:
            for line in self.summary():
                print(line, file=self.stream)
        return Signal.GO

    def summary(self) -> list[str]:
        """Return one line per stage, saying what it and its passes took."""
        return [
            f"{taken:8.3f}s  {path}{self._over(path)}"
            for path, taken in self.elapsed.items()
            if self.depth is None or path.count("/") <= self.depth
        ]

    def _over(self, path: str) -> str:
        """Return what the passes over one stage's data cost.

        Args:
            path: Stage the passes belong to.
        """
        taken = self.passes.get(path)
        if not taken:
            return ""
        mean = sum(taken) / len(taken)
        return (
            f"  ({len(taken)} passes, mean {mean:.3f}s, "
            f"min {min(taken):.3f}s, max {max(taken):.3f}s)"
        )


class EarlyStop:
    """End a stage once a monitored value stops improving."""

    def __init__(
        self,
        monitor: str,
        mode: ModeTypes = "min",
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: ThresholdModeTypes = "rel",
    ):
        """Constructor.

        Args:
            monitor: Payload key to watch.
            mode: `"min"` if smaller is better, `"max"` if larger is.
            patience: Observations without improvement that are tolerated;
                the stage ends on the one after.
            threshold: How much better a value must be to count as improved.
            threshold_mode: Whether `threshold` is read as a fraction of the
                best value so far (`"rel"`) or as a plain difference (`"abs"`).

        Raises:
            ValueError: If `mode` or `threshold_mode` is not one of its two.
        """
        require(mode, MODES, "mode")
        require(threshold_mode, THRESHOLD_MODES, "threshold mode")
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = math.inf if mode == "min" else -math.inf
        self.waited = 0

    def __repr__(self) -> str:
        """Return a short description of the hook."""
        return f"EarlyStop({self.monitor!r}, patience={self.patience})"

    def state_dict(self) -> dict[str, Any]:
        """Return the hook's resumable state."""
        return {"best": self.best, "waited": self.waited}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state previously returned by `state_dict`.

        Args:
            state: Mapping as returned by `state_dict`.
        """
        self.best = float(state.get("best", self.best))
        self.waited = int(state.get("waited", 0))

    def _improved(self, value: float) -> bool:
        """Whether a value beats the best seen by enough to count.

        Args:
            value: The newly observed value.
        """
        if self.mode == "min" and self.threshold_mode == "rel":
            return value < self.best * (1.0 - self.threshold)
        if self.mode == "min" and self.threshold_mode == "abs":
            return value < self.best - self.threshold
        if self.mode == "max" and self.threshold_mode == "rel":
            return value > self.best * (self.threshold + 1.0)
        return value > self.best + self.threshold

    def on(self, event: Event) -> Signal:
        """Watch the monitored value and ask to stop when it stalls.

        Args:
            event: What the runtime just did.
        """
        value = event.payload.get(self.monitor)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return Signal.GO
        if self._improved(float(value)):
            self.best = float(value)
            self.waited = 0
            return Signal.GO
        self.waited += 1
        return Signal.BREAK if self.waited > self.patience else Signal.GO


class GracefulStop:
    """Turn a cancellation signal into a graceful stop.

    A `scancel`, a job time limit, or a Ctrl-C otherwise kills the process
    outright, losing everything since the last checkpoint. Catching the signal
    and raising instead unwinds through every stage's `leave`, lets a
    checkpointer write a final checkpoint, and closes the trace honestly. A
    second signal restores the default handler and dies at once.

    Attributes:
        critical: Set, since a muted canceller would silently stop responding.
    """

    critical = True

    def __init__(self, signals: Sequence[str] = ("SIGINT", "SIGTERM")):
        """Constructor.

        Args:
            signals: Names of the signals to catch. SLURM sends `SIGTERM` on
                `scancel` and at a time limit, and `SIGUSR1` ahead of one when
                a job asks for warning with `--signal=B:USR1@<seconds>`.
        """
        self.signals = tuple(signals)
        self.caught: str | None = None
        self._raised = False
        self._previous: dict[int, Any] = {}

    def __repr__(self) -> str:
        """Return a short description of the hook."""
        return f"GracefulStop({', '.join(self.signals)})"

    def _install(self) -> None:
        """Take over the configured signals, remembering what was there."""
        for name in self.signals:
            number = getattr(signal, name, None)
            if number is None:
                warnings.warn(
                    f"No signal named {name!r} on this platform.", stacklevel=2
                )
                continue
            try:
                self._previous[number] = signal.signal(number, self._catch)
            except ValueError:
                warnings.warn(
                    f"Cannot catch {name} off the main thread; "
                    "cancellation will not be graceful.",
                    stacklevel=2,
                )

    def _restore(self) -> None:
        """Put back whatever handled these signals before the run."""
        for number, previous in self._previous.items():
            signal.signal(number, previous)
        self._previous.clear()

    def _catch(self, number: int, frame: FrameType | None) -> None:
        """Record the first signal, and let a second one through.

        Args:
            number: Signal that fired.
            frame: Stack frame at delivery, unused.
        """
        if self.caught is not None:
            signal.signal(number, self._previous.get(number, signal.SIG_DFL))
            signal.raise_signal(number)
            return
        self.caught = signal.Signals(number).name

    def on(self, event: Event) -> Signal:
        """Install the handlers, then stop the run once one has fired.

        Args:
            event: What the runtime just did.

        Raises:
            C3liRuntimeError: On the first event after a signal is caught.
                Raising again interrupts the program immediately.
        """
        if event.type is EventType.RUN_BEGAN:
            self.caught, self._raised = None, False
            self._install()
        elif event.type is EventType.RUN_ENDED:
            self._restore()
            return Signal.GO
        if self.caught is not None and not self._raised:
            self._raised = True
            raise C3liRuntimeError(f"cancelled by {self.caught}")
        return Signal.GO


class ProgressBar:
    """Hold a bar at the foot of the stream, redrawn as the run proceeds.

    A `Console` writing to the same stream is routed through the same pinned
    writer, so its lines scroll above the bar rather than through it.

    Attributes:
        every: Redraw on every nth unit.
        unit: What the bar measures, `"step"` or `"epoch"`, named as a
            `Checkpointer` names the same thing.
        bar_length: Width of the bar itself.
    """

    def __init__(
        self,
        every: int = 1,
        unit: BarUnitTypes = "step",
        stream: TextIO | None = None,
        bar_length: int = 24,
        live: bool | None = None,
    ):
        """Constructor.

        Args:
            every: Redraw on every nth unit.
            unit: What the bar measures. Steps are finer, but a regimen
                that re-enters a stage for each pass gives each visit only a
                few of them; `"epoch"` spans the whole regimen instead.
            stream: Where to draw; defaults to stdout.
            bar_length: Width of the bar itself.
            live: Whether the stream redraws, detected from it when `None`,
                so a redirected log is not filled with half-drawn bars.
        """
        self.every = max(1, every)
        self.unit = require(unit, BAR_UNIT_MAP, "bar unit")
        self.stream = stream if stream is not None else sys.stdout
        self.bar_length = bar_length
        self.seen = 0
        self.pinned = Pinned(
            self.stream, ansi_supported(self.stream) if live is None else live
        )

    def __repr__(self) -> str:
        """Return a short description of the hook."""
        every, live = self.every, self.pinned.live
        unit = self.unit.value
        return f"ProgressBar({every=}, {unit=}, {live=})"

    def attach(self, runtime: Runtime, ctx: Context) -> None:
        """Route any reporter sharing this stream through the same writer.

        Args:
            runtime: The engine executing the program.
            ctx: Root context of the run.
        """
        for hook in runtime.hooks:
            if isinstance(hook, Console) and hook.stream is self.stream:
                hook.pinned = self.pinned

    def on(self, event: Event) -> Signal:
        """Redraw the bar, holding it until the run is over.

        Args:
            event: What the runtime just did.
        """
        if event.type is self.unit:
            total = event.payload.get("total")
            if not isinstance(total, int) or total <= 0:
                return Signal.GO
            self.seen = (
                event.progress.global_step
                if self.unit is EventType.STEP_ENDED
                else self.seen + 1
            )
            if not self.seen % self.every:
                self.pinned.pin(
                    cli_pbar(
                        min(1.0, self.seen / total),
                        prefix=f"  {event.path}",
                        postfix=f"{self.seen}/{total}",
                        bar_length=self.bar_length,
                    )
                )
        elif event.type is EventType.RUN_ENDED:
            self.pinned.drop()
        return Signal.GO


class Checkpointer:
    """Write a checkpoint as the run proceeds.

    Checkpoints are taken at stage boundaries, so they are always complete.

    Attributes:
        critical: Whether a failed write stops the run.
        store: Directory they are written to.
        needs_store: Whether the runtime must be given a store, which it need
            not be when this hook was told where to write.
    """

    critical: bool = True
    needs_store: bool = True

    def __init__(
        self,
        every: int = 1,
        unit: CheckpointUnitTypes = "epoch",
        keep: int | None = None,
        format: CheckpointFormats = "safetensors",
        prefix: str = "ckpt_",
        rng: bool = True,
        at_end: bool = True,
        store: str | Path | None = None,
    ):
        """Constructor.

        Args:
            every: Write a checkpoint every nth unit.
            unit: What to count:
                - `"epoch"` counts epochs, i.e. dataset passes
                - `"step"` counts optimizer steps
                - `"advance"` counts program steps, i.e. a stage's unit of work
            store: Directory to write into, overriding the run's own. Given
                one, this hook needs nothing from the runtime. A resume reads
                the runtime's store rather than this one, so point `store=`
                at the same directory to carry on from what this wrote.
            keep: How many checkpoints to retain, or `None` to keep all.
            format: `"safetensors"` or `"torch"`.
            prefix: What each checkpoint directory is named before its number.
            rng: Whether to record the ambient RNG state as well.
            at_end: Whether to write a final checkpoint when the run ends,
                cancelled and aborted runs included.

        Raises:
            ValueError: If `every` is not positive, or `unit` is unknown.
        """
        if every < 1:
            raise ValueError(f"Checkpointer needs a positive interval, got {every!r}.")
        require(unit, CHECKPOINT_UNIT_MAP, "checkpoint unit")
        self.every = every
        self.unit = unit
        self.keep = keep
        self.format = format
        self.prefix = prefix
        self.store = Path(store) if store is not None else None
        self.needs_store = self.store is None
        self.rng = rng
        self.at_end = at_end
        self._store: CheckpointStore | None = None
        self._runtime: Runtime | None = None
        self._ctx: Context | None = None
        self._counted = 0
        self._due = False
        self._at: Progress | None = None
        self._at_path: str | None = None

    def __repr__(self) -> str:
        """Return a short description of the hook."""
        every, unit, keep = self.every, self.unit, self.keep
        return f"Checkpointer({every=}, {unit=}, {keep=})"

    @property
    def trigger(self) -> EventType:
        """The event this hook counts."""
        return CHECKPOINT_UNIT_MAP[self.unit]

    def attach(self, runtime: Runtime, ctx: Context) -> None:
        """Receive the run this hook writes checkpoints for.

        Args:
            runtime: The engine executing the program.
            ctx: Root context of the run.
        """
        self._runtime = runtime
        self._ctx = ctx
        self._counted = 0
        self._due = False
        self._at = None
        self._at_path = None
        self._store = CheckpointStore(
            self.store or runtime.store,
            keep=self.keep,
            format=self.format,
            prefix=self.prefix,
        )

    def on(self, event: Event) -> Signal:
        """Note that a checkpoint is due, and write it at the next boundary.

        Epoch and step events come from inside a stage's `execute`; writing
        there would record a stage that has not been closed yet, so the write
        waits for the advance that follows.

        Args:
            event: What the runtime just did.
        """
        if counts_toward(event, self.unit):
            self._counted += 1
            self._due = self._due or self._counted % self.every == 0
            self._at = event.progress
            self._at_path = event.path
        if event.type is EventType.STAGE_ADVANCED and self._due:
            self._due = False
            self.save(event, announce=True)
        elif event.type is EventType.RUN_ENDED and (self.at_end or self._due):
            self._due = False
            self.save(event, announce=False)
        return Signal.GO

    def save(self, event: Event, announce: bool = True) -> None:
        """Write one checkpoint of the run as it currently stands.

        The counters recorded are the triggering event's, not the boundary
        event's: an epoch is reported by the stage that ran it, while the
        boundary carries the program's own.

        Args:
            event: The event that called for it, for its position in the run.
            announce: Whether to emit a `CHECKPOINT` event afterwards. The
                final one does not, the hooks having been told the run ended.

        Raises:
            C3liRuntimeError: If the hook was never attached to a run.
        """
        if self._store is None or self._runtime is None or self._ctx is None:
            raise C3liRuntimeError(
                "This Checkpointer was never attached to a run; it can only be "
                "used through Runtime(hooks=[...])."
            )
        checkpoint = self._store.save(
            index=event.progress.global_step,
            program=self._runtime.program,
            bindings=self._ctx.stateful(),
            topology=self._ctx.topology,
            seed=self._runtime.seed,
            progress=self._at if self._at is not None else event.progress,
            unit=self.unit,
            at=self._at_path,
            rng=self.rng,
        )
        if checkpoint is not None and announce:
            self._ctx.emit(
                EventType.CHECKPOINT,
                path=str(checkpoint.path),
                index=checkpoint.index,
            )
