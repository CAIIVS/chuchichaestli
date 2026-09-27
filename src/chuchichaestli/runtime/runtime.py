# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""The engine: it owns the driver loop and everything a run needs set up once."""

from __future__ import annotations
import os
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple
import torch
from torch import nn
from chuchichaestli.runtime.ckpt import CheckpointStore
from chuchichaestli.runtime.serialize import C3liCheckpointError
from chuchichaestli.runtime.context import Context
from chuchichaestli.runtime.events import (
    C3liRuntimeError,
    Event,
    EventType,
    Progress,
    Signal,
    filter_priority,
)
from chuchichaestli.runtime.hooks import Console
from chuchichaestli.utils.rng import restore_rng_state, seed_ambient
from chuchichaestli.runtime.stages import Phase, Train
from chuchichaestli.runtime.topology import auto_topology, lockstep
from chuchichaestli.runtime.traits import (
    Hook,
    RunAwareHook,
    Stage,
    Topology,
    is_critical,
    needs_store,
)
from chuchichaestli.utils.registry import require


__all__ = [
    "Runtime",
    "BackendsPresets",
    "BACKENDS_PRESETS_MAP",
    "C3liProgramError",
]


BackendsPresets = Literal["fast", "default", "deterministic", "strict"]


class BackendsSettings(NamedTuple):
    """The torch backend settings collection.

    Attributes:
        deterministic: Whether deterministic algorithms are required, or `None`
            to leave the current setting alone.
        warn_only: Whether a non-deterministic operation warns instead of
            raising.
        benchmark: Whether cuDNN autotunes its algorithms, or `None` to leave
            the current setting alone.
    """

    deterministic: bool | None
    warn_only: bool
    benchmark: bool | None


BACKENDS_PRESETS_MAP: dict[str, BackendsSettings] = {
    "fast": BackendsSettings(deterministic=False, warn_only=False, benchmark=True),
    "default": BackendsSettings(deterministic=None, warn_only=False, benchmark=None),
    "deterministic": BackendsSettings(
        deterministic=True, warn_only=True, benchmark=False
    ),
    "strict": BackendsSettings(deterministic=True, warn_only=False, benchmark=False),
}


class C3liProgramError(ValueError):
    """Raised when a program cannot possibly run as written.

    Everything it covers is detectable before any compute happens.
    """


def apply_backends_settings(level: BackendsPresets) -> None:
    """Apply one backend preset for a run.

    Must happen before anything touches CUDA: `CUBLAS_WORKSPACE_CONFIG` is
    read when the cuBLAS handle is created and silently ignored afterwards.

    Args:
        level: `"fast"`, `"default"`, `"deterministic"` or `"strict"`.
            `"default"` leaves every setting exactly as it is rather than
            restoring torch's own defaults.

    Raises:
        ValueError: If `level` is not one of the four.
    """
    settings = require(level, BACKENDS_PRESETS_MAP, "backends preset")
    if settings.deterministic is not None:
        if settings.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(
            settings.deterministic, warn_only=settings.warn_only
        )
        torch.backends.cudnn.deterministic = settings.deterministic
    if settings.benchmark is not None:
        torch.backends.cudnn.benchmark = settings.benchmark


class Runtime:
    """Runs a program.

    Owns what is set once per run: determinism, seed, device, topology, hooks.
    """

    def __init__(
        self,
        program: Stage,
        seed: int = 0,
        store: str | Path | None = None,
        resume: str | Path | int | None = None,
        hooks: Sequence[Hook] | None = None,
        device: torch.device | str | None = None,
        topology: Topology | None = None,
        backends: BackendsPresets = "default",
        allow_signature_change: bool = False,
    ):
        """Constructor.

        Args:
            program: Any stage; a bare stage needs no `Program` wrapper.
            seed: Root seed every random stream in the run derives from.
            store: Directory checkpoints and results are written under;
                resolved to an absolute path.
            resume: An index, `"first"`, `"last"`, `"last~N"`, `"best"`, a
                checkpoint directory, or the manifest file inside one.
            hooks: Observers of the run; defaults to a console reporter.
            device: Device to place modules and batches on; the topology
                picks one per process when absent.
            topology: Process layout; detected from the environment if absent.
            backends: Torch backend preset; one of `"fast"`, `"default"`,
                `"deterministic"` or `"strict"`.
            allow_signature_change: Resume even though the program's stages
                no longer match the original ones.
        """
        self.program = program
        self.seed = seed
        self.store = Path(store).resolve() if store is not None else None
        self.resume = resume
        self.hooks = list(hooks) if hooks is not None else [Console()]
        self.backends = backends
        self.allow_signature_change = allow_signature_change
        self._device = torch.device(device) if device is not None else None
        self._topology = topology
        self._muted: set[int] = set()
        self._checkpoints: CheckpointStore | None = None

    @property
    def device(self) -> torch.device:
        """Device this run places modules and batches on.

        Comes from the topology unless pinned explicitly.
        """
        return self._device if self._device is not None else self.topology.device

    @property
    def topology(self) -> Topology:
        """Process layout this run executes across."""
        if self._topology is None:
            self._topology = auto_topology(self._device)
        return self._topology

    @property
    def checkpoints(self) -> CheckpointStore | None:
        """The run's checkpoints, or `None` when it persists nothing."""
        if self.store is None:
            return None
        if self._checkpoints is None:
            self._checkpoints = CheckpointStore(self.store)
        return self._checkpoints

    def check(self) -> None:
        """Reject a program that cannot run, before any compute happens.

        Checks that every requirement is bound or published ahead of it.

        Raises:
            C3liProgramError: If the program or the runtime options are unusable.
        """
        problems: list[str] = []
        if self.resume is not None and self.store is None:
            problems.append(
                f"resume={self.resume!r} needs a store to resume from, but "
                f"store is {self.store!r}."
            )
        for hook in self.hooks:
            if needs_store(hook) and self.store is None:
                problems.append(f"{hook!r} needs a store to write to.")
        store = self.checkpoints
        if self.resume is not None and store is not None:
            try:
                store.resolve(self.resume)
            except C3liCheckpointError as exc:
                problems.append(str(exc))
        self._check_stage(self.program, set(), problems, self.program.name)
        if problems:
            raise C3liProgramError(
                "This program cannot run:\n  - " + "\n  - ".join(problems)
            )

    def _check_stage(
        self, stage: Stage, available: set[str], problems: list[str], path: str
    ) -> set[str]:
        """Walk a subtree checking requirements against what is bound.

        Returns the names visible to this stage's later siblings: a stage
        publishes to its parent, so what it provides outlives it.

        Args:
            stage: The stage to check.
            available: Binding names resolvable at this point.
            problems: Accumulates human-readable failures.
            path: This stage's logical coordinate.
        """
        for name in getattr(stage, "requires", ()):
            if name not in available:
                problems.append(
                    f"{path!r} requires {name!r}, which nothing provides "
                    f"(available: {sorted(available) or 'nothing'})."
                )
        if isinstance(stage, Train) and stage.epochs is None and stage.steps is None:
            problems.append(
                f"{path!r} has neither epochs nor steps, causing infinite loop."
            )
        if isinstance(stage, Phase):
            inner = available | set(stage.provide)
            for index, child in enumerate(stage.stages):
                inner = self._check_stage(
                    child, inner, problems, f"{path}/{index}:{child.name}"
                )
        return available | set(getattr(stage, "provides", ()))

    def _provision(self) -> None:
        """Ready what the program provides for this run.

        Each module is placed on the device before the topology wraps it,
        since a distributed wrapper requires a module already on its own.
        """
        provide = getattr(self.program, "provide", None)
        if provide is None:
            provide = {}
        for key, value in list(provide.items()):
            if isinstance(value, nn.Module):
                provide[key] = self.topology.wrap(value.to(self.device))
        self._provision_stage(self.program, provide, self.program.name)

    def _provision_stage(
        self, stage: Stage, provide: dict[str, Any], path: str
    ) -> None:
        """Place a model a stage holds directly rather than as a binding.

        Such a model is bound for the whole run under the stage's path.

        Args:
            stage: The stage to provision, and whose children to walk.
            provide: Run-wide bindings a direct model is promoted into.
            path: Path of this stage, which names what it holds.
        """
        model = getattr(stage, "model", None)
        if isinstance(model, nn.Module):
            stage.model = self.topology.wrap(model.to(self.device))
            provide[f"{path}/model"] = stage.model
        for index, child in enumerate(getattr(stage, "stages", ())):
            self._provision_stage(child, provide, f"{path}/{index}:{child.name}")

    def run(self) -> Progress:
        """Execute the program and return where it finished.

        Raises:
            C3liRuntimeError: If any stage or hook aborted the run.
        """
        apply_backends_settings(self.backends)
        self._muted.clear()
        seed_ambient(self.seed)
        self.check()
        self._provision()

        ctx = Context(
            self.program.name,
            self.seed,
            topology=self.topology,
            device=self.device,
            dispatch=self._dispatch,
        )
        for hook in self.hooks:
            if isinstance(hook, RunAwareHook):
                hook.attach(self, ctx)
        ctx.emit(EventType.RUN_BEGAN, seed=self.seed, backends=self.backends)
        failure: str | None = None
        try:
            signal = self._lockstep(lambda: self.program.enter(ctx))
            if self.resume is not None:
                self._restore(ctx)
            while not signal.halts:
                signal = self._lockstep(lambda: self.program.execute(ctx))
                if signal.halts:
                    break
                signal = self._lockstep(lambda: self._advanced(ctx))
        except C3liRuntimeError as exc:
            failure = str(exc)
            raise
        finally:
            self.program.leave(ctx)
            ctx.progress = self.program.progress()
            ctx.emit(EventType.RUN_ENDED, aborted=failure)
        return self.program.progress()

    def _advanced(self, ctx: Context) -> Signal:
        """Announce that the program took one step.

        Args:
            ctx: Root context of the run.
        """
        ctx.progress = self.program.progress()
        return ctx.emit(EventType.STAGE_ADVANCED)

    def _restore(self, ctx: Context) -> None:
        """Put a checkpoint's state back before the first program step.

        Runs after `enter`.

        Args:
            ctx: Root context of the run.

        Raises:
            C3liCheckpointError: If the checkpoint cannot be found or trusted,
                or was written by a run with a different seed.
        """
        store = self.checkpoints
        if store is None:
            raise C3liCheckpointError(
                f"resume={self.resume!r} needs a store to resume from."
            )
        checkpoint = store.load(self.resume)
        if checkpoint.seed != self.seed:
            raise C3liCheckpointError(
                f"This run seeds with {self.seed}, but {self.resume!r} was "
                f"written by a run seeded with {checkpoint.seed}. Randomness "
                "is derived from the seed, so resuming would diverge from the "
                "run being continued; seed this run the same."
            )
        store.restore(
            checkpoint,
            program=self.program,
            bindings=ctx.stateful(),
            topology=self.topology,
            allow_signature_change=self.allow_signature_change,
        )
        if checkpoint.rng is not None:
            restore_rng_state(dict(checkpoint.rng))
        ctx.progress = self.program.progress()

    def _dispatch(self, event: Event) -> Signal:
        """Deliver an event to every hook and combine their verdicts.

        A hook that raises is dropped for the rest of the run unless it sets
        `critical`, so a failing console does not end a long training run.

        Args:
            event: What the runtime just did.
        """
        signals = []
        for hook in self.hooks:
            if id(hook) in self._muted:
                continue
            try:
                signals.append(hook.on(event))
            except C3liRuntimeError:
                raise
            except Exception as exc:
                if is_critical(hook):
                    raise
                self._muted.add(id(hook))
                warnings.warn(
                    f"Hook {hook!r} raised {exc!r}; dropping it for this run.",
                    stacklevel=2,
                )
        return filter_priority(signals)

    def _lockstep(self, call: Callable[[], Signal]) -> Signal:
        """Run a driver call, broadcasting outcomes to every rank.

        Args:
            call: Performs the driver call.

        Raises:
            C3liRuntimeError: If rank 0 reports that some process aborted.
        """
        return lockstep(self.topology, call)

    def __repr__(self) -> str:
        """Return a short description of the runtime."""
        return f"Runtime({self.program!r}, seed={self.seed})"
