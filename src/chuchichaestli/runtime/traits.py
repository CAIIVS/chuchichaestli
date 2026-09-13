# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Structural interfaces the runtime expects of its components."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
import torch
from torch import nn
from chuchichaestli.runtime.events import Event, Progress, Signal

if TYPE_CHECKING:
    from chuchichaestli.runtime.context import Context
    from chuchichaestli.runtime.runtime import Runtime


__all__ = [
    "Stateful",
    "Stage",
    "Hook",
    "CriticalHook",
    "StoreWriterHook",
    "RunAwareHook",
    "is_critical",
    "needs_store",
    "Topology",
]


@runtime_checkable
class Stateful(Protocol):
    """Anything whose state belongs in a checkpoint.

    Satisfied as-is by most torch-native modules and optimizers:
    `nn.Module`, `torch.optim.Optimizer`, `LRScheduler`, and `GradScaler`.
    """

    def state_dict(self) -> dict[str, Any]:
        """Return the component's state."""
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state previously returned by `state_dict`.

        Args:
            state: Mapping as returned by `state_dict`.
        """
        ...


@runtime_checkable
class Stage(Stateful, Protocol):
    """A unit of execution.

    `enter` once, `execute` until it halts, then `leave`. One `execute` is one
    unit of work, and `enter` resets the stage for reuse (e.g. by `Repeat`).

    Attributes:
        name: Identifies the stage within its parent, and forms its path.
    """

    name: str

    def enter(self, ctx: Context) -> Signal:
        """Prepare to run, resetting to a fresh `Progress`.

        Args:
            ctx: Execution context for this entry.
        """
        ...

    def execute(self, ctx: Context) -> Signal:
        """Perform one unit of work.

        Args:
            ctx: Execution context for this entry.
        """
        ...

    def leave(self, ctx: Context) -> Signal:
        """Release anything acquired in `enter`.

        Args:
            ctx: Execution context for this entry.
        """
        ...

    def progress(self) -> Progress:
        """Return where the stage currently is in its own work."""
        ...


@runtime_checkable
class Hook(Protocol):
    """An observer of runtime events."""

    def on(self, event: Event) -> Signal:
        """Handle one event and say whether the run should continue.

        Args:
            event: What the runtime just did.
        """
        ...


@runtime_checkable
class CriticalHook(Hook, Protocol):
    """A hook whose own failure should stop the run.

    Attributes:
        critical: Whether a raised exception propagates instead of dropping
            the hook for the rest of the run.
    """

    critical: bool


@runtime_checkable
class StoreWriterHook(Hook, Protocol):
    """A hook that has nowhere to write without a store.

    Attributes:
        needs_store: Whether the runtime must be given a `store` for this hook
            to work.
    """

    needs_store: bool


@runtime_checkable
class RunAwareHook(Hook, Protocol):
    """A hook that needs the run itself, not just the events it emits."""

    def attach(self, runtime: Runtime, ctx: Context) -> None:
        """Receive the run this hook observes.

        Args:
            runtime: The engine executing the program.
            ctx: Root context of the run.
        """
        ...


def is_critical(hook: Hook) -> bool:
    """Whether a hook's own failure should stop the run.

    Args:
        hook: The hook to ask.
    """
    return isinstance(hook, CriticalHook) and hook.critical


def needs_store(hook: Hook) -> bool:
    """Whether a hook cannot do its job without a store.

    Args:
        hook: The hook to ask.
    """
    return isinstance(hook, StoreWriterHook) and hook.needs_store


@runtime_checkable
class Topology(Protocol):
    """The process layout a run executes across.

    Stages never branch on rank inline; they ask the topology instead.

    Attributes:
        rank: Index of this process across the whole run.
        local_rank: Index of this process on its own host.
        world_size: Number of processes taking part.
        device: Device this process computes on, carrying its own index so
            each rank of a multi-device run places its work correctly.
    """

    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        """Whether this process is the one that writes."""
        ...

    def barrier(self) -> None:
        """Block until every process has arrived."""
        ...

    def reduce(self, value: torch.Tensor, op: str = "mean") -> torch.Tensor:
        """Combine a tensor across processes.

        Args:
            value: Tensor held by this process.
            op: Reduction to apply, `"mean"` or `"sum"`.
        """
        ...

    def wrap(self, module: nn.Module) -> nn.Module:
        """Prepare a module for distributed training.

        Args:
            module: Module to wrap; returned unchanged when single-process.
        """
        ...

    def broadcast(self, value: Any) -> Any:
        """Return rank 0's value on every process.

        Every control decision passes through here, so ranks cannot diverge and
        deadlock on the next collective.

        Args:
            value: This process's candidate value.
        """
        ...

    def state_of(self, stateful: Stateful) -> dict[str, Any]:
        """Capture a component's state, gathered across processes.

        Args:
            stateful: Component whose state is wanted.
        """
        ...

    def load_state(self, stateful: Stateful, state: dict[str, Any]) -> None:
        """Restore state captured by `state_of`, scattered across processes.

        Args:
            stateful: Component to restore.
            state: Mapping as returned by `state_of`.
        """
        ...
