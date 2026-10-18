# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""The tracker base: what a tracker logs, and how a run's events become it."""

from __future__ import annotations
import platform
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args
import torch
from torch import nn
from chuchichaestli.__about__ import __version__
from chuchichaestli.programs.context import stage_name
from chuchichaestli.programs.events import (
    ARTIFACT_TYPES,
    ArtifactTypes,
    Event,
    EventType,
    Signal,
    counts_toward,
)
from chuchichaestli.programs.hooks import Timer
from chuchichaestli.programs.serialize import (
    as_loggable,
    stage_settings,
    stage_signature,
    stages_by_path,
)
from chuchichaestli.utils.functools import numbers
from chuchichaestli.utils.git import GitState, git_state
from chuchichaestli.utils.memory import peak_memory_allocated, reset_peak_memory
from chuchichaestli.utils.registry import require

if TYPE_CHECKING:
    from chuchichaestli.programs.context import Context
    from chuchichaestli.programs.runner import Runner


__all__ = [
    "TrackingPolicy",
    "Tracker",
    "PresetTypes",
    "RunStatusTypes",
    "CheckpointUploadTypes",
]

PresetTypes = Literal["minimal", "default", "full"]
RunStatusTypes = Literal["finished", "failed", "cancelled"]
CheckpointUploadTypes = Literal["best", "last", "all"]


@dataclass(frozen=True)
class TrackingPolicy:
    """What a tracker logs; the default is rich without uploading weights.

    The run's configuration is logged once, as a fresh run opens. A tracker's
    own `config=` is always logged.

    Attributes:
        metrics: Log each training step's losses and rate, and the numbers a
            stage reports as it ends, e.g. an `Eval`'s scores.
        every: Log the step metrics every nth training step.
        grad_norm: Log each logged step's gradient norm, before clipping, per
            update group; with `metrics` only.
        epochs: Log the count of training passes as each one ends.
        timing: Log each training pass's seconds and samples per second.
        memory: Log each training pass's peak accelerator memory, in MiB;
            nothing on a device without a peak counter, such as the CPU.
        runner: Log the run's seed and backend settings.
        stages: Log the program stages' paths and classes, and each stage's
            constructor arguments.
        model: Log each bound model's class, constructor arguments, and total
            and trainable parameter counts.
        environment: Log the package, torch and Python versions, device,
            world size, and the repository's commit and whether it is dirty;
            a dirty repository's diff is stored as `git.patch`. The
            repository needs GitPython (`chuchichaestli[git]`).
        checkpoints: Record each checkpoint's index, step and score.
        artifacts: The kinds of exported file uploaded: `True` for every kind,
            `False` for none, or a sequence of kinds: `"image"` (an
            `ImageExport`'s plots), `"tensor"` (an `ArchiveExport`'s or a
            `Predict` archive's samples), `"weights"` (a `WeightsExport`'s
            file) and `"checkpoint"` (a `Checkpointer`'s checkpoints).
        checkpoint_uploads: Which checkpoints are uploaded: each that becomes
            the best as it is written (`"best"`, falling back to `"last"`
            without a monitored score), the latest one as the run ends
            (`"last"`), or each one as it is written (`"all"`). A checkpoint
            is uploaded whole, so the run can be resumed from it.
    """

    metrics: bool = True
    every: int = 1
    grad_norm: bool = True
    epochs: bool = True
    timing: bool = True
    memory: bool = True
    runner: bool = True
    stages: bool = True
    model: bool = True
    environment: bool = True
    checkpoints: bool = True
    artifacts: bool | Sequence[ArtifactTypes] = ("image", "tensor")
    checkpoint_uploads: CheckpointUploadTypes = "best"

    def __post_init__(self) -> None:
        """Reject a cadence or an artifact kind that cannot be honoured.

        Raises:
            ValueError: If `every` is not positive or a kind is unknown.
        """
        if self.every < 1:
            raise ValueError(f"A tracker logs every n >= 1 steps, got {self.every!r}.")
        if not isinstance(self.artifacts, bool):
            for kind in self.artifacts:
                require(kind, ARTIFACT_TYPES, "artifact kind")
        require(self.checkpoint_uploads, get_args(CheckpointUploadTypes), "upload mode")

    @property
    def kinds(self) -> frozenset[str]:
        """The artifact kinds this policy uploads."""
        if isinstance(self.artifacts, bool):
            return ARTIFACT_TYPES if self.artifacts else frozenset()
        return frozenset(self.artifacts)

    @classmethod
    def minimal(cls) -> TrackingPolicy:
        """Return a policy logging the metrics, how the run ended and your config."""
        return cls(
            grad_norm=False,
            epochs=False,
            timing=False,
            memory=False,
            runner=False,
            stages=False,
            model=False,
            environment=False,
            checkpoints=False,
            artifacts=False,
        )

    @classmethod
    def full(cls) -> TrackingPolicy:
        """Return a policy logging everything, weights included."""
        return cls(artifacts=True)

    @classmethod
    def resolve(
        cls, log: TrackingPolicy | PresetTypes = "default", **options: Any
    ) -> TrackingPolicy:
        """Return a preset or policy with some of its fields changed.

        Args:
            log: A preset's name, or a policy.
            options: Fields to change, e.g. `memory=False`.

        Raises:
            ValueError: If the preset or an option is unknown.
        """
        if isinstance(log, TrackingPolicy):
            base = log
        else:
            presets = {"minimal": cls.minimal, "default": cls, "full": cls.full}
            base = require(log, presets, "tracking preset")()
        names = {f.name for f in fields(cls)}
        for name in options:
            require(name, names, "tracking option")
        return replace(base, **options)


class Tracker(ABC):
    """Send a run's metrics, artifacts and config to an experiment tracker."""

    critical: bool = False
    needs_store: bool = False

    def __init__(
        self,
        log: TrackingPolicy | PresetTypes = "default",
        config: Mapping[str, Any] | None = None,
        **options: Any,
    ):
        """Constructor.

        Args:
            log: What to log: a preset (`"minimal"`, `"default"`, `"full"`) or
                a `TrackingPolicy`.
            config: Your own configuration, e.g. hyperparameters, logged as
                the `config` section whenever one is given.
            options: `TrackingPolicy` fields changed on top of `log`, e.g.
                `memory=False` or `every=10`.

        Raises:
            ValueError: If the preset, an option or an artifact kind is unknown.
        """
        self.log = TrackingPolicy.resolve(log, **options)
        self.config = dict(config or {})
        self.store: Path | None = None
        self.run_id: str | None = None
        self._given = ({} if log == "default" else {"log": log}) | options
        self._timer = Timer(report=False) if self.log.timing else None
        self._ctx: Context | None = None
        self._runner: Runner | None = None
        self._on_main = True
        self._samples: dict[str, int] = {}
        self._norms: dict[str, float] = {}
        self._uploaded_best = False
        self._latest: tuple[list[Path], int, tuple[str, ...]] | None = None
        self._names: dict[str, str | None] = {}
        self._runner_settings: dict[str, Any] = {}
        self._resumed = False
        self._opened = False
        self._closed = False

    def __repr__(self) -> str:
        """Return the tracker as it was configured."""
        given = ", ".join(f"{k}={v!r}" for k, v in self._given.items())
        return f"{type(self).__name__}({given})"

    @property
    def watches_gradients(self) -> bool:
        """Whether the tracker watches each step's gradients, for their norm."""
        return self.log.metrics and self.log.grad_norm

    def on_gradients(
        self,
        params: Sequence[nn.Parameter],
        norm: Callable[[], torch.Tensor],
        group: str | None,
        path: str,
    ) -> None:
        """Keep an update group's gradient norm, for the step's own log call.

        Nothing is computed off the main process or on a step `every` skips.

        Args:
            params: The group's parameters, holding the gradients.
            norm: Returns the gradients' total norm, computed once.
            group: Update group the gradients belong to, or `None`.
            path: Path of the stage taking the step.
        """
        if not self._on_main or (self._count("step") + 1) % self.log.every:
            return
        key = f"{stage_name(path)}/grad_norm"
        self._norms[key if group is None else f"{key}/{group}"] = float(norm())

    def attach(self, runner: Runner, ctx: Context) -> None:
        """Receive the run this tracker logs.

        Args:
            runner: The engine executing the program.
            ctx: Root context of the run, whose counters give the step axis.
        """
        self._ctx = ctx
        self._runner = runner
        self._on_main = ctx.topology.is_main
        if self.store is None and runner.store is not None:
            self.store = Path(runner.store)

    def state_dict(self) -> dict[str, Any]:
        """Return the backend run to reattach to, opening it if not yet open."""
        if self._on_main and not self._opened and not self._closed:
            self._ensure_open()
        return {"run_id": self.run_id}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Mark the run as resumed, to reattach to the backend run it names.

        Args:
            state: Mapping as returned by `state_dict`.
        """
        self.run_id = state.get("run_id")
        self._resumed = self.run_id is not None

    def on(self, event: Event) -> Signal:
        """Log what the event reports, on the main process only.

        Args:
            event: What the runner just did.
        """
        if not self._on_main:
            return Signal.GO
        if self._timer is not None:
            self._timer.on(event)
        step = self._count("step")
        if event.type is EventType.RUN_BEGAN:
            self._began(event)
        elif event.type is EventType.EPOCH_BEGAN:
            self._samples[event.path] = event.progress.samples
            if self.log.memory and self._ctx is not None:
                reset_peak_memory(self._ctx.device)
        elif counts_toward(event, "step"):
            norms, self._norms = self._norms, {}
            if self.log.metrics and step % self.log.every == 0:
                self._metrics(event, step, norms)
        elif counts_toward(event, "epoch"):
            self._epoch(event, step)
        elif event.type is EventType.STAGE_ENDED:
            if self.log.metrics:
                self._metrics(event, step)
        elif event.type is EventType.CHECKPOINT:
            if self.log.checkpoints:
                self._ensure_open()
                payload = event.payload
                path = Path(payload["path"])
                self._log_checkpoint(payload["index"], path, step, payload.get("score"))
        elif event.type is EventType.EXPORTED:
            kind = event.payload.get("kind")
            if kind == "checkpoint":
                if kind in self.log.kinds:
                    self._checkpoint_written(event, step)
            elif kind in self.log.kinds:
                self._ensure_open()
                paths = [Path(f) for f in event.payload.get("files", ())]
                self._log_artifacts(kind, stage_name(event.path), paths, step)
        elif event.type is EventType.RUN_ENDED:
            self._upload_latest()
            if self._opened:
                self._close(self._status(event))
            self._opened, self._closed = False, True
        return Signal.GO

    def _checkpoint_written(self, event: Event, step: int) -> None:
        """Upload a checkpoint now, or keep it for the run's end, per the policy.

        Args:
            event: The `EXPORTED` event of a checkpoint.
            step: Training step it was taken at.
        """
        payload = event.payload
        paths = [Path(f) for f in payload.get("files", ())]
        best = bool(payload.get("best"))
        tags = ("best",) if best else ()
        mode = self.log.checkpoint_uploads
        if mode == "all" or (mode == "best" and best):
            self._uploaded_best = self._uploaded_best or best
            self._latest = None
            self._ensure_open()
            self._log_artifacts("checkpoint", "checkpoint", paths, step, tags)
        else:
            self._latest = (paths, step, tags)

    def _upload_latest(self) -> None:
        """Upload the latest checkpoint kept back, unless a best one was uploaded."""
        if self._latest is None or self._uploaded_best:
            return
        paths, step, tags = self._latest
        self._latest = None
        if not all(p.is_file() for p in paths):
            warnings.warn(
                "The latest checkpoint was pruned before it could be uploaded; "
                "keep more than one, or list the tracker before the Checkpointer.",
                stacklevel=2,
            )
            return
        self._ensure_open()
        self._log_artifacts("checkpoint", "checkpoint", paths, step, tags)

    @staticmethod
    def _status(event: Event) -> RunStatusTypes:
        """Return how a run ended, from its closing event.

        Args:
            event: The `RUN_ENDED` event.
        """
        if event.payload.get("cancelled"):
            return "cancelled"
        return "failed" if event.payload.get("aborted") is not None else "finished"

    def _count(self, unit: str) -> int:
        """Return how many of a unit the run has completed.

        Args:
            unit: `"advance"`, `"epoch"` or `"step"`.
        """
        return self._ctx.count(unit) if self._ctx is not None else 0

    def _began(self, event: Event) -> None:
        """Keep the run's names, and what it was started with, for opening.

        Args:
            event: The `RUN_BEGAN` event.
        """
        payload = event.payload
        self._names = {
            key: payload.get(key) for key in ("project", "experiment", "run_name")
        }
        self._runner_settings = {
            "seed": payload.get("seed"),
            "backends": payload.get("backends"),
        }
        self._closed = False

    def _metrics(
        self, event: Event, step: int, extra: Mapping[str, float] | None = None
    ) -> None:
        """Log an event's numbers under its stage's name, if there are any.

        Args:
            event: The event to read.
            step: Training step the numbers belong to.
            extra: Numbers already keyed, logged in the same call, e.g. the
                step's gradient norms.
        """
        stage = stage_name(event.path)
        values = numbers(event.payload, skip=("total",))
        metrics = {f"{stage}/{k}": v for k, v in values.items()} | dict(extra or {})
        if metrics:
            self._ensure_open()
            self._log(metrics, step)

    def _epoch(self, event: Event, step: int) -> None:
        """Log a training pass: its count, duration, throughput and peak memory.

        Args:
            event: The `EPOCH_ENDED` event of a training pass.
            step: Training step the pass ended at.
        """
        stage = stage_name(event.path)
        metrics: dict[str, float | int] = {}
        if self.log.epochs:
            metrics["epoch"] = self._count("epoch")
        taken = self._timer.passes.get(event.path) if self._timer is not None else None
        if taken:
            metrics[f"{stage}/pass_seconds"] = taken[-1]
            samples = event.progress.samples - self._samples.pop(event.path, 0)
            if samples > 0 and taken[-1] > 0:
                metrics[f"{stage}/samples_per_second"] = samples / taken[-1]
        measured = self.log.memory and self._ctx is not None
        peak = peak_memory_allocated(self._ctx.device) if measured else None
        if peak is not None:
            metrics[f"{stage}/peak_memory_mb"] = peak / 2**20
        if metrics:
            self._ensure_open()
            self._log(metrics, step)

    def _ensure_open(self) -> None:
        """Open the backend run before its first log; a fresh one logs its config."""
        if self._opened:
            return
        self.run_id = self._open(dict(self._names), self.run_id, self._resumed)
        self._opened = True
        if not self._resumed:
            repository = self._repository() if self.log.environment else None
            config = self._run_config(repository)
            if config:
                self._log_config(config)
            if repository is not None and repository.diff:
                self._log_text("git.patch", repository.diff)

    @staticmethod
    def _repository() -> GitState | None:
        """Return the state of the repository the run started in, if readable."""
        state = git_state()
        if state is not None and state.dirty is None:
            warnings.warn(
                "Only the repository's commit is recorded; install "
                "'chuchichaestli[git]' to record whether it is dirty, and its diff.",
                stacklevel=2,
            )
        return state

    def _run_config(self, repository: GitState | None = None) -> dict[str, Any]:
        """Return the sections of the run's configuration the policy asks for.

        Args:
            repository: The state of the repository the run started in.
        """
        config: dict[str, Any] = {}
        if self.config:
            config["config"] = as_loggable(self.config)
        if self.log.runner:
            config["runner"] = as_loggable(self._runner_settings)
        if self.log.stages and self._runner is not None:
            program = self._runner.program
            config["program"] = stage_signature(program)
            config["stages"] = {
                path: stage_settings(stage)
                for path, stage in stages_by_path(program).items()
            }
        if self.log.model and self._ctx is not None:
            config["models"] = self._models(self._ctx)
        if self.log.environment and self._ctx is not None:
            config["environment"] = {
                "chuchichaestli": __version__,
                "torch": torch.__version__,
                "python": platform.python_version(),
                "device": str(self._ctx.device),
                "world_size": self._ctx.topology.world_size,
            }
            if repository is not None:
                git = {"commit": repository.commit, "dirty": repository.dirty}
                config["environment"]["git"] = {
                    k: v for k, v in git.items() if v is not None
                }
        return config

    @staticmethod
    def _models(ctx: Context) -> dict[str, Any]:
        """Return each bound model's class, arguments and parameter counts.

        Args:
            ctx: Root context of the run.
        """
        models: dict[str, Any] = {}
        for name, value in ctx.run_stateful().items():
            module = (
                ctx.topology.unwrap(value) if isinstance(value, nn.Module) else None
            )
            if not isinstance(module, nn.Module):
                continue
            params = list(module.parameters())
            models[name] = {
                "cls": type(module).__name__,
                "kwargs": as_loggable(getattr(module, "init_args", None)),
                "parameters": sum(p.numel() for p in params),
                "trainable": sum(p.numel() for p in params if p.requires_grad),
            }
        return models

    @abstractmethod
    def _open(
        self, names: Mapping[str, str | None], run_id: str | None, resumed: bool
    ) -> str:
        """Start or reattach to the backend run and return its identifier.

        Args:
            names: The run's `project`, `experiment` and `run_name`; the first
                two may be `None`.
            run_id: Identifier of the run to reattach to, when resumed.
            resumed: Whether the run continues one a checkpoint recorded.
        """

    @abstractmethod
    def _log(self, metrics: Mapping[str, float | int], step: int) -> None:
        """Log numbers at a training step.

        Args:
            metrics: Values keyed `<stage>/<name>`, or `epoch`.
            step: Training step they belong to.
        """

    @abstractmethod
    def _log_artifacts(
        self,
        kind: ArtifactTypes,
        key: str,
        paths: Sequence[Path],
        step: int,
        tags: Sequence[str] = (),
    ) -> None:
        """Log files a stage or a checkpointer wrote.

        Args:
            kind: What the files hold.
            key: Name of the stage that wrote them, or `"checkpoint"`.
            paths: The files.
            step: Training step they belong to.
            tags: Labels for the backend to version them by, e.g. `"best"`.
        """

    @abstractmethod
    def _log_config(self, config: Mapping[str, Any]) -> None:
        """Log the run's configuration, once, for a fresh run.

        Args:
            config: Nested and JSON-safe: `config` when one was given, and
                `runner`, `program`, `stages`, `models` and `environment` when
                the policy asks for them.
        """

    def _log_text(self, name: str, text: str) -> None:
        """Store a text file with the run, e.g. the repository's uncommitted diff.

        Args:
            name: File name to store it under, e.g. `"git.patch"`.
            text: Its contents.
        """

    def _log_checkpoint(
        self, index: int, path: Path, step: int, score: Mapping[str, Any] | None
    ) -> None:
        """Record a checkpoint the run wrote; nothing is uploaded.

        Args:
            index: The checkpoint's index in its store.
            path: Directory it was written to.
            step: Training step it was taken at.
            score: Its monitored value, with `monitor` and `mode`, if any.
        """

    @abstractmethod
    def _close(self, status: RunStatusTypes) -> None:
        """Finish the backend run.

        Args:
            status: `"finished"`, `"failed"`, or `"cancelled"` by a signal or
                an interrupt.
        """
