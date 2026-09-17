# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Execution contexts: one per stage entry, chained to the enclosing stage."""

from __future__ import annotations
from collections.abc import Callable, Iterator
from typing import Any
import torch
from chuchichaestli.runtime.events import Event, EventType, Progress, Signal
from chuchichaestli.utils.rng import WorkerSeeder, derive_seed, rng_generator
from chuchichaestli.runtime.topology import Local
from chuchichaestli.runtime.traits import Stateful, Topology


__all__ = ["Context", "C3liContextError"]


class C3liContextError(KeyError):
    """Raised when a stage asks for a binding nothing provided."""


class Context:
    """Everything a stage needs that it does not own itself.

    One per stage entry, chained to the enclosing one.

    Attributes:
        path: This stage's logical coordinate, e.g. `"program/1:cycle/0:fit"`.
        seed: Seed of the run as a whole.
        parent: Enclosing context, or `None` at the root.
        topology: Process layout the run executes across.
        device: Device tensors created here should land on.
        dtype: Default dtype for tensors created here, if the run pins one.
        group: Update group currently being applied, or `None`.
        progress: Counters as of the stage's latest step.
    """

    def __init__(
        self,
        path: str,
        seed: int = 0,
        *,
        parent: Context | None = None,
        topology: Topology | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        dispatch: Callable[[Event], Signal] | None = None,
        bindings: dict[str, Any] | None = None,
        group: str | None = None,
    ):
        """Constructor.

        Args:
            path: This stage's logical coordinate.
            seed: Seed of the run as a whole.
            parent: Enclosing context, or `None` at the root.
            topology: Process layout; defaults to a single process.
            device: Device tensors created here should land on.
            dtype: Default dtype for tensors created here.
            dispatch: Receives emitted events and returns the hooks' verdict.
            bindings: Artifacts bound at this level.
            group: Update group currently being applied.
        """
        self.path = path
        self.seed = seed
        self.parent = parent
        self.topology = topology if topology is not None else Local()
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.dtype = dtype
        self.group = group
        self.progress = Progress()
        self._bindings: dict[str, Any] = dict(bindings or {})
        self._cache: dict[tuple[str | None, str], Any] = {}
        self._dispatch = dispatch

    def __repr__(self) -> str:
        """Return a short description of the context."""
        group = f", group={self.group!r}" if self.group else ""
        return f"Context({self.path!r}, seed={self.seed}{group})"

    def __getitem__(self, key: str) -> Any:
        """Resolve a binding, searching enclosing contexts outwards.

        Args:
            key: Name the artifact was bound under.

        Raises:
            C3liContextError: If no enclosing context binds `key`.
        """
        ctx: Context | None = self
        while ctx is not None:
            if key in ctx._bindings:
                return ctx._bindings[key]
            ctx = ctx.parent
        raise C3liContextError(
            f"Nothing bound as {key!r} at {self.path!r}. Bound here or above: "
            f"{sorted(self.names())}."
        )

    def __contains__(self, key: str) -> bool:
        """Whether a binding resolves from here.

        Args:
            key: Name to look for.
        """
        ctx: Context | None = self
        while ctx is not None:
            if key in ctx._bindings:
                return True
            ctx = ctx.parent
        return False

    def get(self, key: str, default: Any = None) -> Any:
        """Resolve a binding, or return a default.

        Args:
            key: Name the artifact was bound under.
            default: Returned when nothing binds `key`.
        """
        return self[key] if key in self else default

    def names(self) -> Iterator[str]:
        """Iterate over every binding name visible from here, innermost first."""
        seen: set[str] = set()
        ctx: Context | None = self
        while ctx is not None:
            for key in ctx._bindings:
                if key not in seen:
                    seen.add(key)
                    yield key
            ctx = ctx.parent

    def stateful(self) -> dict[str, Any]:
        """Return every binding that carries state, keyed by name."""
        return {
            name: self[name]
            for name in self.names()
            if isinstance(self[name], Stateful)
        }

    def bind(self, key: str, value: Any) -> None:
        """Bind an artifact for this stage and its children.

        Args:
            key: Name to bind under.
            value: Artifact to bind.
        """
        self._bindings[key] = value

    def publish(self, key: str, value: Any) -> None:
        """Bind an artifact into the enclosing phase, so later siblings see it.

        Args:
            key: Name to bind under.
            value: Artifact to bind.
        """
        (self.parent or self)._bindings[key] = value

    def resolve(self, value: Any) -> Any:
        """Resolve a string to its binding, passing anything else through.

        Lets a stage be given either a model or the name of one.

        Args:
            value: A binding name, or an object to use directly.
        """
        return self[value] if isinstance(value, str) else value

    def _keyed_path(self, key: str) -> str:
        """Qualify a key with this stage's path.

        Args:
            key: What the randomness is for, or `""` for the stage itself.
        """
        return f"{self.path}/{key}" if key else self.path

    def seed_for(self, key: str = "") -> int:
        """Derive a seed for a key at this position.

        Args:
            key: What the randomness is for, e.g. `"data/epoch=3"`.
        """
        return derive_seed(self.seed, self._keyed_path(key))

    def rng(
        self, key: str = "", device: torch.device | str | None = None
    ) -> torch.Generator:
        """Build a generator seeded for a key at this position.

        Args:
            key: What the randomness is for, e.g. `"data/epoch=3"`.
            device: Device the generator draws for; defaults to the CPU.
        """
        return rng_generator(self.seed, self._keyed_path(key), device)

    def seeder(self, key: str = "") -> WorkerSeeder:
        """Build a worker seeder for a key at this position.

        Args:
            key: What the randomness is for, e.g. `"data/epoch=3"`.
        """
        return WorkerSeeder(self.seed, self._keyed_path(key))

    def cache(self, key: str, fn: Callable[[], Any]) -> Any:
        """Compute a value once per step, reusing it across objective terms.

        Keyed by update group as well as name, since the graphs differ.

        Args:
            key: Name for the cached value.
            fn: Produces the value on first request within a step.
        """
        slot = (self.group, key)
        if slot not in self._cache:
            self._cache[slot] = fn()
        return self._cache[slot]

    def clear_cache(self) -> None:
        """Drop cached intermediates, at the end of a step."""
        self._cache.clear()

    def emit(self, event_type: EventType, **payload: Any) -> Signal:
        """Report something to the hooks and return their verdict.

        Args:
            event_type: What the event records.
            payload: Extra JSON-serializable detail.
        """
        event = Event(event_type, self.path, self.progress, payload)
        return self._dispatch(event) if self._dispatch is not None else Signal.GO

    def child(self, index: int, name: str) -> Context:
        """Build the context for a child stage.

        The only way paths are constructed, so that the checkpoint key, the
        random stream and the event label of a stage can never disagree.

        Args:
            index: Position of the child within this phase.
            name: The child stage's name.
        """
        return Context(
            f"{self.path}/{index}:{name}",
            self.seed,
            parent=self,
            topology=self.topology,
            device=self.device,
            dtype=self.dtype,
            dispatch=self._dispatch,
            group=self.group,
        )

    def at_group(self, group: str | None) -> Context:
        """Return a view of this context bound to an update group.

        Shares bindings and the intermediate cache with the original; only the
        group differs, so the cache can key on it.

        Args:
            group: Update group being applied, or `None`.
        """
        view = Context.__new__(Context)
        view.__dict__.update(self.__dict__)
        view.group = group
        return view
