# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Execution contexts: one per stage entry, chained to the enclosing stage."""

from __future__ import annotations
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any
import torch
from chuchichaestli.programs.events import (
    UNIT_MAP,
    Event,
    EventType,
    Progress,
    Signal,
    counts_toward,
)
from chuchichaestli.utils.rng import WorkerSeeder, derive_seed, rng_generator, root_seed
from chuchichaestli.programs.topology import Local
from chuchichaestli.programs.traits import Stateful, Topology


__all__ = [
    "Context",
    "C3liContextError",
    "subpath",
    "join_path",
    "split_segment",
    "split_path",
    "stage_name",
]


def subpath(path: str, key: object) -> str:
    """Return the path of a key below a path; an empty part is left out.

    Args:
        path: The path above, e.g. a stage's.
        key: What goes below it, e.g. a binding name or a tree index.
    """
    return "/".join(part for part in (path, str(key)) if part)


def join_path(path: str, index: int, name: str) -> str:
    """Return the path of a phase's child.

    Args:
        path: Path of the phase.
        index: Position of the child within it.
        name: The child stage's name.
    """
    return subpath(path, f"{index}:{name}")


def split_segment(segment: str) -> tuple[int | None, str]:
    """Return a path segment's position and stage name; the root has no position.

    Args:
        segment: One segment of a path, e.g. `"0:fit"`.
    """
    index, sep, name = segment.partition(":")
    return (int(index), name) if sep else (None, segment)


def split_path(path: str) -> list[tuple[int | None, str]]:
    """Return each segment of a path as its position and stage name.

    Args:
        path: A stage's path, e.g. `"program/1:cycle/0:fit"`.
    """
    return [split_segment(segment) for segment in path.split("/")]


def stage_name(path: str) -> str:
    """Return the name of the stage a path ends in.

    Args:
        path: A stage's path, e.g. `"program/1:cycle/0:fit"`.
    """
    return split_path(path)[-1][1]


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
        seed: int | None = None,
        *,
        parent: Context | None = None,
        topology: Topology | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        dispatch: Callable[[Event], Signal] | None = None,
        watch: Callable[..., None] | None = None,
        bindings: dict[str, Any] | None = None,
        group: str | None = None,
        counts: dict[str, int] | None = None,
        registry: dict[str, Any] | None = None,
        watcher: Callable[[str, Any], None] | None = None,
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
            watch: Receives each step's gradients, or `None` when no hook
                watches them.
            bindings: Artifacts bound at this level.
            group: Update group currently being applied.
            counts: Units completed so far, shared with every child.
            registry: Every binding of the run, shared with every child.
            watcher: Receives each binding as it is registered.
        """
        self.path = path
        self.seed = root_seed(seed)
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
        self._counts = {} if counts is None else counts
        self._registry = {} if registry is None else registry
        self._watcher = watcher
        self._dispatch = dispatch
        self._watch = watch

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
        self._register(key, value)

    def publish(self, key: str, value: Any) -> None:
        """Bind an artifact into the enclosing phase, so later siblings see it.

        A published artifact is derived from one already bound, whose own
        stage records it, so it is not registered for the run.

        Args:
            key: Name to bind under.
            value: Artifact to bind.
        """
        (self.parent or self)._bindings[key] = value

    def _register(self, key: str, value: Any) -> None:
        """Record a binding run-wide, wherever in the program it was bound.

        Args:
            key: Name it was bound under.
            value: Artifact that was bound.
        """
        if any(held is value for held in self._registry.values()):
            return
        held = self._registry.get(key)
        if held is not None:
            warnings.warn(
                f"Two artifacts are bound as {key!r}; only the later one is "
                "recorded for the run. Give them distinct names.",
                stacklevel=3,
            )
        self._registry[key] = value
        if self._watcher is not None:
            self._watcher(key, value)

    def run_stateful(self) -> dict[str, Any]:
        """Return every stateful binding of the run, keyed by binding name."""
        return {
            name: value
            for name, value in self._registry.items()
            if isinstance(value, Stateful)
        }

    def watch_bindings(self, watcher: Callable[[str, Any], None] | None) -> None:
        """Call `watcher` with the name and value of every binding from now on.

        Args:
            watcher: Receives each binding as it is registered, or `None`.
        """
        self._watcher = watcher

    def resolve(self, value: Any) -> Any:
        """Resolve a string to its binding, passing anything else through.

        Lets a stage be given either a model or the name of one.

        Args:
            value: A binding name, or an object to use directly.
        """
        return self[value] if isinstance(value, str) else value

    def seed_for(self, key: str = "") -> int:
        """Derive a seed for a key at this position.

        Args:
            key: What the randomness is for, e.g. `"data/epoch=3"`.
        """
        return derive_seed(self.seed, subpath(self.path, key))

    def rng(
        self, key: str = "", device: torch.device | str | None = None
    ) -> torch.Generator:
        """Build a generator seeded for a key at this position.

        Args:
            key: What the randomness is for, e.g. `"data/epoch=3"`.
            device: Device the generator draws for; defaults to the CPU.
        """
        return rng_generator(self.seed, subpath(self.path, key), device)

    def seeder(self, key: str = "") -> WorkerSeeder:
        """Build a worker seeder for a key at this position.

        Args:
            key: What the randomness is for, e.g. `"data/epoch=3"`.
        """
        return WorkerSeeder(self.seed, subpath(self.path, key))

    def cache(self, key: str, fn: Callable[[], Any]) -> Any:
        """Compute a value once per micro-batch, reusing it across terms.

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
        """Drop cached intermediates, before each micro-batch is run."""
        self._cache.clear()

    @property
    def watches_gradients(self) -> bool:
        """Whether any hook watches the gradients of each step."""
        return self._watch is not None

    def watch_gradients(
        self,
        params: Sequence[torch.nn.Parameter],
        norm: Callable[[], torch.Tensor],
        group: str | None,
    ) -> None:
        """Hand an update group's gradients to the hooks watching them, if any.

        Args:
            params: The group's parameters, holding the gradients.
            norm: Returns the gradients' total norm, computed once.
            group: Update group the gradients belong to, or `None`.
        """
        if self._watch is not None:
            self._watch(params, norm, group, self.path)

    def emit(self, event_type: EventType, **payload: Any) -> Signal:
        """Report something to the hooks and return their verdict.

        Args:
            event_type: What the event records.
            payload: Extra JSON-serializable detail.
        """
        event = Event(event_type, self.path, self.progress, payload)
        for unit in UNIT_MAP:
            if counts_toward(event, unit):
                self._counts[unit] = self._counts.get(unit, 0) + 1
        return self._dispatch(event) if self._dispatch is not None else Signal.GO

    def count(self, unit: str) -> int:
        """Return how many of a unit the run has completed.

        Args:
            unit: `"advance"`, `"epoch"` or `"step"`.
        """
        return self._counts.get(unit, 0)

    def counts(self) -> dict[str, int]:
        """Return how many of each unit the run has completed."""
        return {unit: self.count(unit) for unit in UNIT_MAP}

    def restore_counts(self, counts: Mapping[str, int]) -> None:
        """Carry on counting from counts a checkpoint recorded.

        Args:
            counts: Units completed, as `counts` returns them.
        """
        self._counts.clear()
        self._counts.update({unit: int(count) for unit, count in counts.items()})

    def child(self, index: int, name: str) -> Context:
        """Build the context for a child stage.

        The only way paths are constructed, so that the checkpoint key, the
        random stream and the event label of a stage can never disagree.

        Args:
            index: Position of the child within this phase.
            name: The child stage's name.
        """
        return Context(
            join_path(self.path, index, name),
            self.seed,
            parent=self,
            topology=self.topology,
            device=self.device,
            dtype=self.dtype,
            dispatch=self._dispatch,
            watch=self._watch,
            group=self.group,
            counts=self._counts,
            registry=self._registry,
            watcher=self._watcher,
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
