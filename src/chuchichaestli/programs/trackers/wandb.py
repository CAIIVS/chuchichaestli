# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Logging a run to Weights & Biases."""

from __future__ import annotations
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar
from chuchichaestli.programs.events import ArtifactTypes
from chuchichaestli.programs.trackers.base import (
    PresetTypes,
    RunStatusTypes,
    Tracker,
    TrackingPolicy,
)
from chuchichaestli.utils.registry import require_module
from chuchichaestli.utils.visualization.images import RASTER_FORMATS

__all__ = ["Wandb"]


class Wandb(Tracker):
    """Log a run to Weights & Biases, through the run object it returns."""

    # set from the run itself, so an `init` mapping may not set them too
    RESERVED: ClassVar[frozenset[str]] = frozenset(
        {"id", "resume", "dir", "project", "group", "name", "config"}
    )

    def __init__(
        self,
        entity: str | None = None,
        mode: str | None = None,
        init: Mapping[str, Any] | None = None,
        log: TrackingPolicy | PresetTypes = "default",
        config: Mapping[str, Any] | None = None,
        **options: Any,
    ):
        """Constructor.

        Args:
            entity: Team or user the run belongs to; W&B's default when absent.
            mode: `"online"`, `"offline"` or `"disabled"`; W&B's default when
                absent.
            init: Further arguments to `wandb.init`, e.g. `tags`, `notes` or
                `settings`.
            log: What to log, as for `Tracker`.
            config: Your own configuration, as for `Tracker`.
            options: `TrackingPolicy` fields changed on top of `log`.

        Raises:
            ValueError: If `init` sets what the tracker sets from the run.
        """
        super().__init__(log=log, config=config, **options)
        clash = sorted(self.RESERVED & set(init or {}))
        if clash:
            raise ValueError(
                f"Wandb sets {clash} from the run itself; name the run on the "
                "Runner (project=, experiment=, run_name=) and pass config= here."
            )
        self.entity = entity
        self.mode = mode
        self.init = dict(init or {})
        self._wandb: Any = None
        self._run: Any = None
        self._floor = 0

    def _open(
        self, names: Mapping[str, str | None], run_id: str | None, resumed: bool
    ) -> str:
        """Start the W&B run, or reattach to it, keeping the steps it already has.

        Args:
            names: The run's `project`, `experiment` and `run_name`.
            run_id: The W&B run's id, when resumed.
            resumed: Whether the run continues one a checkpoint recorded.
        """
        self._wandb = require_module("wandb", "wandb", "logging to Weights & Biases")
        self._run = self._wandb.init(
            entity=self.entity,
            project=names.get("project"),
            group=names.get("experiment"),
            name=names.get("run_name"),
            id=run_id,
            resume="must" if resumed else None,
            dir=str(self.store) if self.store is not None else None,
            mode=self.mode,
            **self.init,
        )
        # W&B keeps the steps logged before the interruption; later ones resume
        self._floor = self._run.step if resumed else 0
        return self._run.id

    def _log(self, metrics: Mapping[str, float | int], step: int) -> None:
        """Log the numbers at their step, unless W&B already has that step.

        Args:
            metrics: Values keyed `<stage>/<name>`, or `epoch`.
            step: Training step they belong to.
        """
        if step >= self._floor:
            self._run.log(dict(metrics), step=step)

    def _log_artifacts(
        self,
        kind: ArtifactTypes,
        key: str,
        paths: Sequence[Path],
        step: int,
        tags: Sequence[str] = (),
    ) -> None:
        """Version the files as an artifact; raster images also go to the panels.

        Args:
            kind: What the files hold; the artifact's type.
            key: Name of the stage that wrote them, or `"checkpoint"`.
            paths: The files.
            step: Training step they belong to; an alias `step-<n>`.
            tags: Further aliases, e.g. `"best"`.
        """
        if step < self._floor:
            return
        if kind == "image":
            shown = [p for p in paths if p.suffix.lower() in RASTER_FORMATS]
            if shown:
                images = [self._wandb.Image(str(p)) for p in shown]
                self._run.log({key: images}, step=step)
        name = re.sub(r"[^\w.-]+", "-", f"{key}-{self._run.id}")
        artifact = self._wandb.Artifact(name, type=kind)
        for path in paths:
            artifact.add_file(str(path))
        self._run.log_artifact(artifact, aliases=[f"step-{step}", *tags])

    def _log_config(self, config: Mapping[str, Any]) -> None:
        """Set the run's config, nested as it is.

        Args:
            config: The run's configuration, nested and JSON-safe.
        """
        self._run.config.update(dict(config), allow_val_change=True)

    def _log_text(self, name: str, text: str) -> None:
        """Save a text file with the run.

        Args:
            name: Its name, e.g. `"git.patch"`.
            text: Its contents.
        """
        path = Path(self._run.dir) / name
        path.write_text(text)
        self._run.save(str(path), base_path=self._run.dir, policy="now")

    def _log_checkpoint(
        self, index: int, path: Path, step: int, score: Mapping[str, Any] | None
    ) -> None:
        """Keep the latest checkpoint in the run's summary.

        Args:
            index: The checkpoint's index in its store.
            path: Directory it was written to.
            step: Training step it was taken at.
            score: Its monitored value, with `monitor` and `mode`, if any.
        """
        record = {"index": index, "path": str(path), "step": step}
        if score is not None:
            record["score"] = dict(score)
        self._run.summary.update({"checkpoint": record})

    def _close(self, status: RunStatusTypes) -> None:
        """Finish the run; a cancelled one is marked as preempted first.

        Args:
            status: `"finished"`, `"failed"` or `"cancelled"`.
        """
        if status == "cancelled":
            self._run.mark_preempting()
        self._run.finish(exit_code=0 if status == "finished" else 1)
        self._run = None
