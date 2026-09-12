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
from typing import Any, ClassVar, Literal, TextIO
from chuchichaestli.debug import cli_pbar
from chuchichaestli.utils.ansi import ANSIShade, ansi_supported, paint
from chuchichaestli.runtime.events import C3liRuntimeError, Event, EventType, Signal


ModeTypes = Literal["min", "max"]
ThresholdModeTypes = Literal["rel", "abs"]

MODES: frozenset[str] = frozenset({"min", "max"})
THRESHOLD_MODES: frozenset[str] = frozenset({"rel", "abs"})

__all__ = [
    "Console",
    "Jsonl",
    "Timer",
    "EarlyStop",
    "Cancel",
    "ModeTypes",
    "ThresholdModeTypes",
]


class Console:
    """Print a human-readable line as the run proceeds."""

    RENDERERS: ClassVar[dict[EventType, str]] = {
        EventType.STAGE_BEGAN: "_stage_began",
        EventType.STAGE_ENDED: "_stage_ended",
        EventType.STEP_ENDED: "_step_line",
        EventType.CHECKPOINT: "_checkpoint",
        EventType.RUN_ENDED: "_run_ended",
    }

    def __init__(
        self,
        every: int = 1,
        stream: TextIO | None = None,
        bar_length: int = 24,
        color: bool | None = None,
    ):
        """Constructor.

        Args:
            every: Print a step line every nth step.
            stream: Where to write; defaults to stdout.
            bar_length: Width of the progress bar when a total is known.
            color: Force colour on or off; detected from the stream when
                `None`, so a redirected log never receives escape codes.
        """
        self.every = max(1, every)
        self.stream = stream if stream is not None else sys.stdout
        self.bar_length = bar_length
        self.color = color

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
        """Render a stage closing.

        Args:
            event: The event to render.
            tint: Whether to colour the line.
        """
        return paint(f"< {event.path}", ANSIShade.GREEN, on=tint)

    def _checkpoint(self, event: Event, tint: bool) -> str | None:
        """Render a checkpoint having been written.

        Args:
            event: The event to render.
            tint: Whether to colour the line.
        """
        return paint(
            f"  saved {event.payload.get('path', '')}", ANSIShade.YELLOW, on=tint
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

    def _step_line(self, event: Event, tint: bool) -> str | None:
        """Render a step, every nth one, with a bar when a total is known.

        Args:
            event: The event to render.
            tint: Whether to colour the line.
        """
        if event.progress.global_step % self.every:
            return None
        detail = " ".join(
            f"{k}={paint(f'{v:.4g}', ANSIShade.BOLD, on=tint)}"
            if isinstance(v, float)
            else f"{k}={v}"
            for k, v in event.payload.items()
            if isinstance(v, (int, float))
        )
        prefix = paint(
            f"  {event.path} step {event.progress.global_step}", ANSIShade.DIM, on=tint
        )
        total = event.payload.get("total")
        if isinstance(total, int) and total > 0:
            return cli_pbar(
                min(1.0, event.progress.step / total),
                prefix=prefix,
                postfix=detail,
                bar_length=self.bar_length,
            )
        return f"{prefix} {detail}".rstrip()

    def on(self, event: Event) -> Signal:
        """Print the event, when it is one this reporter shows.

        Args:
            event: What the runtime just did.
        """
        name = self.RENDERERS.get(event.type)
        if name is None:
            return Signal.GO
        line = getattr(self, name)(event, self.colored)
        if line is not None:
            print(line, file=self.stream, flush=True)
        return Signal.GO


class Jsonl:
    """Append every event to a JSON Lines file, recording the run's trace."""

    def __init__(
        self,
        path: str | Path,
        only: Sequence[EventType] | None = None,
        flush_every: int = 1,
    ):
        """Constructor.

        Args:
            path: File to append to; parent directories are created.
            only: Record just these event types, or all of them when `None`.
            flush_every: Flush periodicity. Flushing costs almost nothing;
                so raise this only for a trace busy enough for that to matter,
                e.g. micro-batching.

        Raises:
            ValueError: If `flush_every` is not positive.
        """
        if flush_every < 1:
            raise ValueError(
                f"Jsonl needs a positive flush interval, got {flush_every!r}."
            )
        self.path = Path(path)
        self.only = frozenset(only) if only is not None else None
        self.flush_every = flush_every
        self._write_counter = 0
        self._handle: TextIO | None = None

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


class Timer:
    """Measure how long stages take, and report once at they end.

    Reports separately rather than into event payloads.
    """

    def __init__(self, stream: TextIO | None = None):
        """Constructor.

        Args:
            stream: Where to write the summary; defaults to stderr.
        """
        self.stream = stream if stream is not None else sys.stderr
        self.elapsed: dict[str, float] = {}
        self._started: dict[str, float] = {}

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
        elif event.type is EventType.STAGE_ENDED:
            started = self._started.pop(event.path, None)
            if started is not None:
                taken = time.perf_counter() - started
                self.elapsed[event.path] = self.elapsed.get(event.path, 0.0) + taken
        elif event.type is EventType.RUN_ENDED:
            for path, taken in self.elapsed.items():
                print(f"{taken:8.3f}s  {path}", file=self.stream)
        return Signal.GO


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
        if mode not in MODES:
            raise ValueError(f"Unsupported mode: {mode!r}. Use one of {sorted(MODES)}.")
        if threshold_mode not in THRESHOLD_MODES:
            raise ValueError(
                f"Unsupported threshold mode: {threshold_mode!r}. "
                f"Use one of {sorted(THRESHOLD_MODES)}."
            )
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

        The four cases are `ReduceLROnPlateau._is_better` verbatim, so a run
        configured with both agrees with itself about what a plateau is.

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


class Cancel:
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
        return f"Cancel({', '.join(self.signals)})"

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

    def _catch(self, number: int, frame: Any) -> None:
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
