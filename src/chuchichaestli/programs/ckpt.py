# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Checkpoints on disk: writing a run's state and reading it back."""

from __future__ import annotations
import json
import math
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from torch import nn
from chuchichaestli.models.spec import ModelSpec, read_model_state, write_model_state
from chuchichaestli.programs.events import Progress
from chuchichaestli.programs.serialize import (
    C3liCheckpointError,
    unpack_tree,
    stage_signature,
    stages_by_path,
    component_states,
    merge_tree,
    writable_spec,
)
from chuchichaestli.programs.traits import Stage, Topology
from chuchichaestli.utils.io import read_state, staged, write_state
from chuchichaestli.utils.rng import capture_rng_state
from chuchichaestli.utils.registry import require


__all__ = [
    "SCHEMA_VERSION",
    "CheckpointFormats",
    "CHECKPOINT_FORMAT_MAP",
    "Checkpoint",
    "CheckpointStore",
]


SCHEMA_VERSION = 1

_BACK = re.compile(r"^last~(\d+)$")
_STEM_LIMIT = 100

CheckpointFormats = Literal["safetensors", "torch"]

CHECKPOINT_FORMAT_MAP: dict[str, str] = {
    "safetensors": ".safetensors",
    "torch": ".pt",
}


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """One saved point in a run.

    Attributes:
        path: Directory the checkpoint was read from or written to.
        index: Where in the run it was taken, counting program steps.
        unit: What the writer was counting, or `None` if it did not say.
        at: Path of the stage that triggered it, or `None`.
        version: Manifest schema version.
        seed: Root seed of the run that wrote it.
        signature: The program's stages, as `[path, class name]` pairs.
        progress: Counters at the time of writing.
        program: The program's own state.
        bindings: Per-binding state, keyed by binding name.
        components: Component states of each stage, keyed by its path.
        hooks: State of each hook that keeps one, keyed as `hook_keys` names it.
        score: The monitored value it was written at, with its `monitor` and
            `mode`, or `None` if the writer monitored nothing.
        specs: How to rebuild each binding that records a model spec.
        weights: File each model binding was written to.
        rng: Ambient RNG capture, or `None` if it was not recorded.
        run_name: Name of the run that wrote it, or `None`.
        counts: Units the run had completed, as `Context.counts` returns them.
    """

    path: Path
    index: int
    version: int = SCHEMA_VERSION
    seed: int = 0
    signature: list[list[str]] = field(default_factory=list)
    progress: Progress = field(default_factory=Progress)
    unit: str | None = None
    at: str | None = None
    program: Stage | None = None
    bindings: Mapping[str, Any] = field(default_factory=dict)
    components: Mapping[str, Any] = field(default_factory=dict)
    hooks: Mapping[str, Any] = field(default_factory=dict)
    score: Mapping[str, Any] | None = None
    specs: Mapping[str, ModelSpec] = field(default_factory=dict)
    weights: Mapping[str, Path] = field(default_factory=dict)
    rng: Mapping[str, Any] | None = None
    run_name: str | None = None
    counts: Mapping[str, int] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return a short description of the checkpoint."""
        index, bindings = self.index, sorted(self.bindings)
        return f"Checkpoint({self.path.name!r}, {index=}, {bindings=})"

    def build(
        self, name: str, strict: bool = True, trusted: bool = False, **overrides: Any
    ) -> nn.Module:
        """Rebuild a binding from its spec and load its weights into it.

        Args:
            name: Binding to rebuild.
            strict: Whether every key must match, as for `load_state_dict`.
            trusted: Let the spec name anything, for a checkpoint you trust.
            overrides: Constructor arguments to replace before building.

        Raises:
            C3liCheckpointError: If no spec was recorded for that binding.
        """
        if name not in self.specs:
            raise C3liCheckpointError(
                f"No model spec for binding {name!r} in {str(self.path)!r}. "
                f"Recorded specs: {sorted(self.specs) or 'none'}."
            )
        model = self.specs[name].build(trusted=trusted, **overrides)
        model.load_state_dict(self.bindings[name], strict=strict)
        return model


class CheckpointStore:
    """The checkpoints of one run, laid out under a single directory.

    ```text
    <root>/ckpt_001000/
        model.safetensors      one file per model, with its own spec
        state.safetensors      optimizers, schedulers, RNG
        manifest.json          program states, all but model and state
    ```

    Attributes:
        root: Directory the checkpoints live under.
        keep: How many to retain, or `None` to keep every one.
        format: Which file format the tensors are written in.
        prefix: What each checkpoint directory is named before its number.
        manifest: Name of the file holding everything but the tensors.
        state_key: Name, without suffix, of the shared tensor file.
    """

    def __init__(
        self,
        root: str | Path,
        keep: int | None = None,
        format: CheckpointFormats = "safetensors",
        prefix: str = "ckpt_",
        manifest: str = "manifest.json",
        state_key: str = "state",
    ):
        """Constructor.

        Args:
            root: Directory with/for checkpoints, resolved to an absolute path.
            keep: How many checkpoints to retain, or `None` to keep every one.
            format: `"safetensors"` or `"torch"`.
            prefix: Checkpoint prefix; ignored if reading an existing store.
            manifest: Name of the file holding program details.
            state_key: Name (without suffix) of the shared tensor file,
                containing optimizer, LR scheduler, and other training states.

        Raises:
            ValueError: If `keep` is not positive, `format` is unknown, or any
                of the names is empty.
        """
        if keep is not None and keep < 1:
            raise ValueError(f"A store keeps at least one checkpoint, got {keep!r}.")
        require(format, CHECKPOINT_FORMAT_MAP, "checkpoint format")
        for label, value in (
            ("prefix", prefix),
            ("manifest", manifest),
            ("state_key", state_key),
        ):
            if not value:
                raise ValueError(f"A store needs a non-empty {label}.")
        self.root = Path(root).resolve()
        self.keep = keep
        self.format = format
        self.prefix = prefix
        self.manifest = manifest
        self.state_key = state_key

    def __repr__(self) -> str:
        """Return a short description of the store."""
        keep, format, prefix = self.keep, self.format, self.prefix
        return f"CheckpointStore({str(self.root)!r}, {keep=}, {format=}, {prefix=})"

    @property
    def suffix(self) -> str:
        """Suffix of the tensor file this store writes."""
        return CHECKPOINT_FORMAT_MAP[self.format]

    def directory_for(self, index: int, create: bool = False) -> Path:
        """Return the directory a checkpoint at given index.

        Args:
            index: Where in the run the checkpoint is taken.
            create: Whether to create the directory, parents included.
        """
        directory = self.root / f"{self.prefix}{index:06d}"
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    def checkpoints(self) -> list[Path]:
        """Return every complete checkpoint directory, oldest first.

        Any directory with a manifest is valid (even with mixed `prefix`es).
        """
        if not self.root.is_dir():
            return []
        found: list[tuple[int, Path]] = []
        for directory in self.root.iterdir():
            # hidden stage directories are skipped
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            manifest = self._read_json(directory / self.manifest)
            if manifest is None:
                continue
            found.append((int(manifest.get("index", 0)), directory))
        return [directory for _, directory in sorted(found)]

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        """Read a JSON file, or return `None` if it is missing or truncated.

        Args:
            path: File to read.
        """
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        """Write a JSON file into place atomically.

        Args:
            path: File to write.
            payload: What to serialize.
        """
        with staged(path) as (scratch,):
            scratch.write_text(
                json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8"
            )

    def resolve(self, resume: str | Path | int) -> Path:
        """Return the checkpoint directory a `resume=` argument names.

        Args:
            resume: An index, `"first"`, `"last"`, `"last~N"` for N before the
                last, `"best"`, a checkpoint directory, or its manifest file.

        Raises:
            C3liCheckpointError: If nothing usable is there.
        """
        if isinstance(resume, int) and not isinstance(resume, bool):
            return self._at_index(resume)
        if isinstance(resume, str) and (
            resume in ("first", "last") or _BACK.match(resume)
        ):
            return self._at_position(resume)
        if resume == "best":
            best = self.best()
            if best is None:
                raise C3liCheckpointError(
                    f"No checkpoint under {str(self.root)!r} records a score to "
                    "rank by; give the Checkpointer a monitor=, or resume from "
                    "'last'."
                )
            return best
        path = Path(resume)
        if path.is_file():
            path = path.parent
        if not (path / self.manifest).is_file():
            raise C3liCheckpointError(
                f"No checkpoint manifest at {str(path / self.manifest)!r}."
            )
        return path

    def _at_position(self, resume: str) -> Path:
        """Return the checkpoint from a positional name.

        Args:
            resume: `"first"`, `"last"`, or `"last~N"`.

        Raises:
            C3liCheckpointError: If the store holds too few checkpoints.
        """
        found = self.checkpoints()
        if not found:
            raise C3liCheckpointError(
                f"No complete checkpoint under {str(self.root)!r} to resume from."
            )
        if resume == "first":
            return found[0]
        back = int(match.group(1)) if (match := _BACK.match(resume)) else 0
        if back >= len(found):
            raise C3liCheckpointError(
                f"{resume!r} asks for {back} before the last, but "
                f"{str(self.root)!r} holds only {len(found)}."
            )
        return found[-1 - back]

    def _at_index(self, index: int) -> Path:
        """Return the checkpoint written at an index.

        Args:
            index: Program step the checkpoint was taken at.

        Raises:
            C3liCheckpointError: If no checkpoint was kept at that index.
        """
        directory = self.directory_for(index)
        if not (directory / self.manifest).is_file():
            kept = [self.load_index(p) for p in self.checkpoints()]
            raise C3liCheckpointError(
                f"No checkpoint at index {index} under {str(self.root)!r}. "
                f"Kept: {kept or 'none'}."
            )
        return directory

    def load_index(self, directory: Path) -> int:
        """Return the index a checkpoint directory was written at.

        Args:
            directory: A complete checkpoint directory.
        """
        manifest = self._read_json(directory / self.manifest) or {}
        return int(manifest.get("index", 0))

    def save(
        self,
        *,
        index: int,
        program: Stage,
        bindings: Mapping[str, Any],
        topology: Topology,
        seed: int = 0,
        progress: Progress | None = None,
        unit: str | None = None,
        at: str | None = None,
        rng: bool = True,
        hooks: Mapping[str, Any] | None = None,
        score: Mapping[str, Any] | None = None,
        run_name: str | None = None,
        counts: Mapping[str, int] | None = None,
    ) -> Checkpoint | None:
        """Write one checkpoint and return it, or `None` off the main process.

        Args:
            index: A checkpoint's position, counting program steps.
            program: The running program, whose state is recorded.
            bindings: Artifacts to checkpoint, keyed by binding name.
            topology: Process layout state is captured through.
            seed: Root seed of the run.
            progress: Counters at the time of writing.
            unit: What the writer was counting.
            at: Path of the stage that triggered the write.
            rng: Whether to capture the ambient RNG state as well.
            hooks: Hook states to record, as `hook_states` returns them.
            score: The monitored value the checkpoint is ranked by, as a
                mapping of `monitor`, `mode` and `value`.
            run_name: Name of the run writing it.
            counts: Units the run has completed.
        """
        directory = self.directory_for(index)
        models = {n: v for n, v in bindings.items() if isinstance(v, nn.Module)}
        rest = {n: v for n, v in bindings.items() if n not in models}
        weights = {n: topology.state_of(v) for n, v in models.items()}
        payload: dict[str, Any] = {
            "program": topology.state_of(program),
            "bindings": {n: topology.state_of(v) for n, v in rest.items()},
            "components": component_states(program),
            "hooks": dict(hooks or {}),
        }
        if rng:
            payload["rng"] = capture_rng_state()
        if not topology.is_main:
            return None

        scratch = directory.with_name(f".{directory.name}.part")
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
        try:
            # write model states
            specs = {n: writable_spec(topology.unwrap(v)) for n, v in models.items()}
            files: dict[str, Path] = {}
            for name, tensors in weights.items():
                filename = self._weight_name(name, set(files.values()))
                write_model_state(scratch / filename, tensors, spec=specs[name])
                files[name] = directory / filename
            # write states for optimizer, LR scheduler, etc.
            tensors, skeleton = unpack_tree(payload)
            state_file = f"{self.state_key}{self.suffix}"
            write_state(scratch / state_file, tensors)
            # write manifest
            progress = progress if progress is not None else Progress()
            signature = stage_signature(program)
            manifest = {
                "version": SCHEMA_VERSION,
                "index": index,
                "seed": seed,
                "unit": unit,
                "at": at,
                "signature": signature,
                "progress": progress.to_dict(),
                "weights": {n: p.name for n, p in files.items()},
                "state_file": state_file,
                "state": skeleton,
            }
            if score is not None:
                manifest["score"] = dict(score)
            if run_name is not None:
                manifest["run_name"] = run_name
            if counts is not None:
                manifest["counts"] = dict(counts)
            self._write_json(scratch / self.manifest, manifest)
            self._publish(scratch, directory)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        # delete old checkpoints if `keep` is set
        self.prune()
        return Checkpoint(
            path=directory,
            index=index,
            seed=seed,
            signature=signature,
            progress=progress,
            unit=unit,
            at=at,
            program=payload["program"],
            bindings={**payload["bindings"], **weights},
            components=payload["components"],
            hooks=payload["hooks"],
            score=dict(score) if score is not None else None,
            specs={n: s for n, s in specs.items() if s is not None},
            weights=files,
            rng=payload.get("rng"),
            run_name=run_name,
            counts=dict(counts or {}),
        )

    @staticmethod
    def _publish(scratch: Path, directory: Path) -> None:
        """Move a finished checkpoint into place, replacing one at its index.

        Args:
            scratch: Hidden directory the checkpoint was written into.
            directory: Where it belongs.
        """
        if not directory.exists():
            scratch.rename(directory)
            return
        retired = directory.with_name(f".{directory.name}.old")
        shutil.rmtree(retired, ignore_errors=True)
        directory.rename(retired)
        scratch.rename(directory)
        shutil.rmtree(retired, ignore_errors=True)

    def _weight_name(self, binding: str, taken: set[Path]) -> str:
        """Return a filename for a model binding, unique within a checkpoint.

        The manifest records which binding each file holds, so a name that
        sanitizes down to nothing readable still loads.

        Args:
            binding: Binding name, which may not be a usable filename.
            taken: Paths already claimed in this checkpoint.
        """
        stem = re.sub(r"[^A-Za-z0-9_.-]", "-", binding)[:_STEM_LIMIT]
        if not stem.strip(".-"):
            stem = "weights"
        claimed = {path.name.lower() for path in taken}
        candidate, index = f"{stem}{self.suffix}", 1
        while candidate.lower() in claimed:
            index += 1
            candidate = f"{stem}-{index}{self.suffix}"
        return candidate

    def _state_path(self, directory: Path, manifest: Mapping[str, Any]) -> Path | None:
        """Return a checkpoint's shared tensor file, whatever format wrote it.

        Args:
            directory: The checkpoint directory.
            manifest: Its parsed manifest, which names the file if recent enough.
        """
        if "state_file" in manifest:
            path = directory / manifest["state_file"]
            return path if path.is_file() else None
        for suffix in CHECKPOINT_FORMAT_MAP.values():
            path = directory / f"{self.state_key}{suffix}"
            if path.is_file():
                return path
        return None

    def best(self) -> Path | None:
        """Return the checkpoint with the best recorded score, or `None`.

        Ties go to the newer checkpoint.

        Raises:
            C3liCheckpointError: If the checkpoints were scored by different
                monitors, which cannot be ranked against each other.
        """
        ranked = self._best_by_monitor()
        if len(ranked) > 1:
            raise C3liCheckpointError(
                f"The checkpoints under {str(self.root)!r} were scored by "
                f"different monitors {sorted(ranked)}; name a directory instead."
            )
        return next(iter(ranked.values()), None)

    def _best_by_monitor(self) -> dict[tuple[str, str], Path]:
        """Return the best-scored checkpoint for each monitor and mode."""
        best: dict[tuple[str, str], tuple[float, Path]] = {}
        for directory in self.checkpoints():
            manifest = self._read_json(directory / self.manifest) or {}
            score = manifest.get("score")
            if not isinstance(score, Mapping):
                continue
            value = float(score.get("value", math.nan))
            if math.isnan(value):
                continue
            key = (str(score.get("monitor")), str(score.get("mode", "min")))
            sign = 1.0 if key[1] == "max" else -1.0
            held = best.get(key)
            if held is None or sign * value >= sign * held[0]:
                best[key] = (value, directory)
        return {key: directory for key, (_, directory) in best.items()}

    def prune(self) -> list[Path]:
        """Delete the oldest checkpoints beyond `keep` and return what went.

        The best-scored checkpoint of each monitor is never pruned.
        """
        if self.keep is None:
            return []
        found = self.checkpoints()
        best = set(self._best_by_monitor().values())
        stale = [d for d in found[: max(0, len(found) - self.keep)] if d not in best]
        for directory in stale:
            shutil.rmtree(directory)
        return stale

    def load(self, resume: str | Path | int) -> Checkpoint:
        """Read a checkpoint back.

        Args:
            resume: An index, `"first"`, `"last"`, `"last~N"` for N before the
                last, `"best"`, a checkpoint directory, or its manifest file.

        Raises:
            C3liCheckpointError: If nothing usable is there, or it was written
                by a schema this version does not know.
        """
        directory = self.resolve(resume)
        # read manifest
        manifest = self.read_manifest(directory)
        version = int(manifest.get("version", 0))
        if version > SCHEMA_VERSION:
            raise C3liCheckpointError(
                f"Checkpoint {str(directory)!r} uses manifest schema {version}, "
                f"but this version understands at most {SCHEMA_VERSION}."
            )
        # read states for optimizer, LR scheduler, etc.
        state_path = self._state_path(directory, manifest)
        tensors = read_state(state_path) if state_path is not None else {}
        payload = merge_tree(tensors, manifest.get("state", {}))

        # read model states
        bindings = dict(payload.get("bindings", {}))
        specs, files = {}, {}
        for name, filename in manifest.get("weights", {}).items():
            path = directory / filename
            bindings[name], spec = read_model_state(path)
            files[name] = path
            if spec is not None:
                specs[name] = spec
        # fully loaded checkpoint
        return Checkpoint(
            path=directory,
            index=int(manifest.get("index", 0)),
            version=version,
            seed=int(manifest.get("seed", 0)),
            signature=manifest.get("signature", []),
            progress=Progress.from_dict(manifest.get("progress", {})),
            unit=manifest.get("unit"),
            at=manifest.get("at"),
            program=payload.get("program"),
            bindings=bindings,
            components=payload.get("components") or {},
            hooks=payload.get("hooks") or {},
            score=manifest.get("score"),
            specs=specs,
            weights=files,
            rng=payload.get("rng"),
            run_name=manifest.get("run_name"),
            counts=manifest.get("counts", {}),
        )

    def read_manifest(self, resume: str | Path | int) -> dict[str, Any]:
        """Read a checkpoint's manifest, without its tensors.

        Args:
            resume: An index, `"first"`, `"last"`, `"last~N"`, `"best"`, a
                checkpoint directory, or its manifest file.

        Raises:
            C3liCheckpointError: If nothing usable is there.
        """
        path = self.resolve(resume) / self.manifest
        manifest = self._read_json(path)
        if manifest is None:
            raise C3liCheckpointError(f"The manifest at {str(path)!r} is unreadable.")
        return manifest

    def run_names(self) -> set[str]:
        """Return the names of the runs that wrote this store's checkpoints."""
        names = (
            self.read_manifest(path).get("run_name") for path in self.checkpoints()
        )
        return {name for name in names if name}

    def restore(
        self,
        checkpoint: Checkpoint,
        *,
        program: Stage,
        bindings: Mapping[str, Any],
        topology: Topology,
        allow_signature_change: bool = False,
    ) -> dict[str, Any]:
        """Put a checkpoint's state back, returning what nothing was bound for.

        Args:
            checkpoint: What was read back.
            program: The program to restore into.
            bindings: Artifacts to restore, keyed by binding name.
            topology: Process layout state goes back through.
            allow_signature_change: Load even though the stages changed.

        Raises:
            C3liCheckpointError: If the stages changed unexpectedly.
        """
        current = stage_signature(program)
        if not allow_signature_change and current != list(checkpoint.signature):
            raise C3liCheckpointError(
                f"Checkpoint {str(checkpoint.path)!r} was written from a "
                "different stage signature, so its keys no longer line up:\n"
                f"  saved:   {checkpoint.signature}\n"
                f"  current: {current}\n"
                "Pass allow_signature_change=True if the change is deliberate."
            )
        if checkpoint.program is not None:
            topology.load_state(program, checkpoint.program)
        held = stages_by_path(program)
        for path, states in checkpoint.components.items():
            defer = getattr(held.get(path), "defer_component_states", None)
            if callable(defer):
                defer(states)
        unplaced: dict[str, Any] = {}
        for name, state in checkpoint.bindings.items():
            if name in bindings:
                topology.load_state(bindings[name], state)
            else:
                unplaced[name] = state
        return unplaced
