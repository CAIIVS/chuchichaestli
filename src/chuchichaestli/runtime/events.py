# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""The runtime's shared vocabulary: what happened, and what to do next.

`Event` is one dataclass over one string enum, so it is JSON-serializable
without a custom encoder and a run's trace can be compared for equality.
"""

from __future__ import annotations
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


__all__ = [
    "Signal",
    "C3liRuntimeError",
    "Progress",
    "EventType",
    "Event",
    "filter_priority",
]


class Signal(str, Enum):
    """What a stage or hook asks the runtime to do next."""

    GO = "go"
    DONE = "done"
    SKIP = "skip"
    BREAK = "break"

    @property
    def halts(self) -> bool:
        """Whether the current stage is over, for any reason."""
        return self in (Signal.DONE, Signal.BREAK)


_PRIORITY = {Signal.GO: 0, Signal.SKIP: 1, Signal.DONE: 2, Signal.BREAK: 3}


def filter_priority(signals: Iterable[Signal]) -> Signal:
    """Return the highest-priority signal of several.

    Priority order: `GO` < `SKIP` < `DONE` < `BREAK`.

    Args:
        signals: Signals to reduce.
    """
    return max(signals, key=_PRIORITY.__getitem__, default=Signal.GO)


class C3liRuntimeError(RuntimeError):
    """Raised to stop the program, needs to be broadcasted to all ranks."""


@dataclass(frozen=True, slots=True)
class Progress:
    """Counters locating a stage within its own work (essential for resume).

    Attributes:
        epoch: Completed passes over the data.
        step: Optimizer steps taken within the current epoch.
        global_step: Optimizer steps taken since the stage began.
        samples: Samples seen since the stage began.
        index: Position within a parent's children, used by phases.
        done: Whether the stage has finished.
    """

    epoch: int = 0
    step: int = 0
    global_step: int = 0
    samples: int = 0
    index: int = 0
    done: bool = False

    def next_step(self, samples: int = 0) -> Progress:
        """Return the progress after one more step.

        Args:
            samples: Samples consumed by that step.
        """
        return replace(
            self,
            step=self.step + 1,
            global_step=self.global_step + 1,
            samples=self.samples + samples,
        )

    def next_epoch(self) -> Progress:
        """Return the progress at the start of the following epoch."""
        return replace(self, epoch=self.epoch + 1, step=0)

    def at(self, index: int) -> Progress:
        """Return the progress positioned at a child index.

        Args:
            index: Position within the parent's children.
        """
        return replace(self, index=index)

    def finish(self) -> Progress:
        """Return the progress marked as finished."""
        return replace(self, done=True)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of the counters."""
        return {
            "epoch": self.epoch,
            "step": self.step,
            "global_step": self.global_step,
            "samples": self.samples,
            "index": self.index,
            "done": self.done,
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> Progress:
        """Rebuild progress from its mapping form.

        Args:
            state: Mapping as returned by `to_dict`.
        """
        return cls(**{k: v for k, v in state.items() if k in cls.__slots__})


class EventType(str, Enum):
    """What an event records.

    One enum with dotted values rather than a type-plus-boundary pair, so
    meaningless combinations are unrepresentable and a hook matches one field.
    """

    RUN_BEGAN = "run.began"
    RUN_ENDED = "run.ended"
    STAGE_BEGAN = "stage.began"
    STAGE_ENDED = "stage.ended"
    EPOCH_BEGAN = "epoch.began"
    EPOCH_ENDED = "epoch.ended"
    STEP_ENDED = "step.ended"
    CHECKPOINT = "checkpoint"


@dataclass(frozen=True, slots=True)
class Event:
    """One record of something the runtime did.

    Attributes:
        type: What the event records.
        path: Stage path the event originates from.
        progress: Counters at the time of the event.
        payload: Extra JSON-serializable detail (losses, metrics, timings).
    """

    type: EventType
    path: str
    progress: Progress = field(default_factory=Progress)
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of the event."""
        return {
            "type": self.type.value,
            "path": self.path,
            "progress": self.progress.to_dict(),
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> Event:
        """Rebuild an event from its mapping form.

        Args:
            state: Mapping as returned by `to_dict`.
        """
        return cls(
            type=EventType(state["type"]),
            path=state["path"],
            progress=Progress.from_dict(state.get("progress", {})),
            payload=dict(state.get("payload", {})),
        )
