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
from chuchichaestli.programs.ckpt import CheckpointStore
from chuchichaestli.programs.serialize import C3liCheckpointError
from chuchichaestli.programs.context import Context
from chuchichaestli.programs.events import (
    C3liRunError,
    Event,
    EventType,
    Progress,
    Signal,
    filter_priority,
)
from chuchichaestli.programs.hooks import Console
from chuchichaestli.utils.rng import restore_rng_state, seed_ambient
from chuchichaestli.programs.stages import Phase, Train
from chuchichaestli.programs.topology import auto_topology, lockstep
from chuchichaestli.programs.traits import (
    Hook,
    RunAwareHook,
    Stage,
    Topology,
    is_critical,
    needs_store,
)
from chuchichaestli.utils.registry import require


__all__ = [
    "Runner",
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


class Runner:
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
        self._unplaced: dict[str, Any] = {}

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
            C3liProgramError: If the program or the runner options are unusable.
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
        provisioned: dict[int, nn.Module] = {}
        for key, value in list((provide or {}).items()):
            if isinstance(value, nn.Module):
                provide[key] = self._provision_module(value, provisioned)
        self._provision_stage(self.program, provide, self.program.name, provisioned)

    def _provision_module(
        self, module: nn.Module, provisioned: dict[int, nn.Module]
    ) -> nn.Module:
        """Move a module to the device and wrap it, once however often it recurs.

        Args:
            module: The module, or a wrapper this run already made of it.
            provisioned: Wrappers made so far, keyed by the id of the module
                and of the wrapper itself.
        """
        if id(module) not in provisioned:
            wrapped = self.topology.wrap(module.to(self.device))
            provisioned[id(module)] = provisioned[id(wrapped)] = wrapped
        return provisioned[id(module)]

    def _provision_stage(
        self,
        stage: Stage,
        provide: dict[str, Any] | None,
        path: str,
        provisioned: dict[int, nn.Module],
    ) -> None:
        """Place a model a stage holds directly rather than as a binding.

        Such a model is bound for the whole run under the path of the first
        stage holding it, so a repeated stage or a shared module is bound once.

        Args:
            stage: The stage to provision, and whose children to walk.
            provide: Run-wide bindings a direct model is promoted into, or
                `None` for a program that provides none.
            path: Path of this stage, which names what it holds.
            provisioned: Wrappers made so far, as `_provision_module` keeps them.
        """
        model = getattr(stage, "model", None)
        if isinstance(model, nn.Module):
            seen = id(model) in provisioned
            stage.model = self._provision_module(model, provisioned)
            if provide is not None and not seen:
                provide[f"{path}/model"] = stage.model
        for index, child in enumerate(getattr(stage, "stages", ())):
            self._provision_stage(
                child, provide, f"{path}/{index}:{child.name}", provisioned
            )

    def run(self) -> Progress:
        """Execute the program and return where it finished.

        Raises:
            C3liRunError: If any stage or hook aborted the run.
        """
        apply_backends_settings(self.backends)
        self._muted.clear()
        self._unplaced.clear()
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
        except C3liRunError as exc:
            failure = str(exc)
            raise
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.program.leave(ctx)
            ctx.progress = self.program.progress()
            self._report_unplaced()
            try:
                ctx.emit(EventType.RUN_ENDED, aborted=failure)
            except Exception as exc:
                if failure is None:
                    raise
                warnings.warn(
                    f"Closing the failed run also raised {exc!r}.", stacklevel=2
                )
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
        self._unplaced = store.restore(
            checkpoint,
            program=self.program,
            bindings=ctx.run_stateful(),
            topology=self.topology,
            allow_signature_change=self.allow_signature_change,
        )
        ctx.watch_bindings(self._place)
        if checkpoint.rng is not None:
            restore_rng_state(dict(checkpoint.rng))
        ctx.progress = self.program.progress()

    def _place(self, name: str, value: Any) -> None:
        """Restore a checkpoint's state into a binding that has just appeared.

        Args:
            name: Qualified name the binding was registered under.
            value: Artifact that was bound.
        """
        state = self._unplaced.pop(name, None)
        if state is not None:
            self.topology.load_state(value, state)

    def _report_unplaced(self) -> None:
        """Warn about checkpoint state that nothing in the run was bound for."""
        if self._unplaced:
            warnings.warn(
                f"Nothing in this run was bound for {sorted(self._unplaced)}, "
                f"so that part of {self.resume!r} was not restored.",
                stacklevel=2,
            )

    def _dispatch(self, event: Event) -> Signal:
        """Deliver an event to every hook and combine their verdicts.

        A hook that raises is dropped for the rest of the run unless it sets
        `critical`, so a failing console does not end a long training run.

        Args:
            event: What the runner just did.
        """
        signals = []
        for hook in self.hooks:
            if id(hook) in self._muted:
                continue
            try:
                signals.append(hook.on(event))
            except C3liRunError:
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
            C3liRunError: If rank 0 reports that some process aborted.
        """
        return lockstep(self.topology, call)

    def __repr__(self) -> str:
        """Return a short description of the runner."""
        return f"Runner({self.program!r}, seed={self.seed})"
