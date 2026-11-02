# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Logging a run to MLflow."""

from __future__ import annotations
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar
from chuchichaestli.programs.events import ArtifactTypes
from chuchichaestli.programs.serialize import flatten_tree
from chuchichaestli.programs.trackers.base import (
    PresetTypes,
    RunStatusTypes,
    Tracker,
    TrackingPolicy,
)
from chuchichaestli.utils.registry import require_module

__all__ = ["MLflow"]


class MLflow(Tracker):
    """Log a run to MLflow, through an explicit client rather than the active run."""

    STATUSES: ClassVar[dict[str, str]] = {
        "finished": "FINISHED",
        "failed": "FAILED",
        "cancelled": "KILLED",
    }
    # seconds between metric batches, so a remote server is not called each step
    FLUSH_SECONDS: ClassVar[float] = 10.0

    def __init__(
        self,
        tracking_uri: str | None = None,
        artifact_location: str | None = None,
        tags: Mapping[str, str] | None = None,
        log: TrackingPolicy | PresetTypes = "default",
        config: Mapping[str, Any] | None = None,
        **options: Any,
    ):
        """Constructor.

        Args:
            tracking_uri: Where runs are tracked; `MLFLOW_TRACKING_URI`, or a
                SQLite database in the run's store, when absent. Any scheme a
                store plugin serves works.
            artifact_location: Where a new experiment keeps its files; the
                store's `mlartifacts` directory with the default database.
            tags: Tags every run gets, beside `project` and those MLflow's
                run context providers add.
            log: What to log, as for `Tracker`.
            config: Your own configuration, as for `Tracker`.
            options: `TrackingPolicy` fields changed on top of `log`.
        """
        super().__init__(log=log, config=config, **options)
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        self.artifact_location = artifact_location
        self.tags = dict(tags or {})
        self.needs_store = self.tracking_uri is None
        self._entities: Any = None
        self._client: Any = None
        self._buffered: list[Any] = []
        self._flushed = 0.0

    def _open(
        self, names: Mapping[str, str | None], run_id: str | None, resumed: bool
    ) -> str:
        """Create the MLflow run in its experiment, or mark a resumed one running.

        Args:
            names: The run's `project`, `experiment` and `run_name`.
            run_id: The MLflow run's id, when resumed.
            resumed: Whether the run continues one a checkpoint recorded.
        """
        purpose = "logging to MLflow"
        tracking = require_module("mlflow.tracking", "mlflow", purpose)
        registry = require_module("mlflow.tracking.context.registry", "mlflow", purpose)
        self._entities = require_module("mlflow.entities", "mlflow", purpose)
        uri, location = self.tracking_uri, self.artifact_location
        if uri is None:
            store = Path(self.store or ".").resolve()
            store.mkdir(parents=True, exist_ok=True)
            uri = f"sqlite:///{store / 'mlflow.db'}"
            location = location or (store / "mlartifacts").as_uri()
        self._client = tracking.MlflowClient(tracking_uri=uri)
        self._flushed = time.monotonic()
        if resumed and run_id is not None:
            self._client.update_run(run_id, "RUNNING")
            return run_id
        project = names.get("project")
        name = names.get("experiment") or project or "chuchichaestli"
        experiment = self._client.get_experiment_by_name(name)
        experiment_id = (
            experiment.experiment_id
            if experiment is not None
            else self._client.create_experiment(name, artifact_location=location)
        )
        tags = {**self.tags, **({"project": project} if project else {})}
        run = self._client.create_run(
            experiment_id,
            run_name=names.get("run_name"),
            tags=registry.resolve_tags(tags),
        )
        return run.info.run_id

    def _log(self, metrics: Mapping[str, float | int], step: int) -> None:
        """Buffer the numbers, sending a batch when it is full or due.

        Args:
            metrics: Values keyed `<stage>/<name>`, or `epoch`.
            step: Training step they belong to.
        """
        now = int(time.time() * 1000)
        metric = self._entities.Metric
        self._buffered += [metric(k, float(v), now, step) for k, v in metrics.items()]
        due = time.monotonic() - self._flushed >= self.FLUSH_SECONDS
        if due or len(self._buffered) >= 1000:
            self._flush()

    def _flush(self) -> None:
        """Send the buffered numbers, in batches MLflow accepts."""
        for start in range(0, len(self._buffered), 1000):
            batch = self._buffered[start : start + 1000]
            self._client.log_batch(self.run_id, metrics=batch)
        self._buffered = []
        self._flushed = time.monotonic()

    def _log_artifacts(
        self,
        kind: ArtifactTypes,
        key: str,
        paths: Sequence[Path],
        step: int,
        tags: Sequence[str] = (),
    ) -> None:
        """Store the files under their kind, stage and step; tag a best checkpoint.

        Args:
            kind: What the files hold.
            key: Name of the stage that wrote them, or `"checkpoint"`.
            paths: The files.
            step: Training step they belong to.
            tags: Labels, e.g. `"best"`, kept as run tags pointing at the files.
        """
        folder = f"{kind}/{key}/step_{step:06d}"
        for path in paths:
            self._client.log_artifact(self.run_id, str(path), artifact_path=folder)
        for tag in tags:
            self._client.set_tag(self.run_id, f"{tag}_{kind}", folder)

    def _log_config(self, config: Mapping[str, Any]) -> None:
        """Log the config as parameters, flattened and within MLflow's limits.

        Args:
            config: The run's configuration, nested and JSON-safe.
        """
        param, limit = self._entities.Param, 6000
        params = [param(k, str(v)[:limit]) for k, v in flatten_tree(config).items()]
        for start in range(0, len(params), 100):
            self._client.log_batch(self.run_id, params=params[start : start + 100])

    def _log_text(self, name: str, text: str) -> None:
        """Store a text file among the run's artifacts.

        Args:
            name: Its name, e.g. `"git.patch"`.
            text: Its contents.
        """
        self._client.log_text(self.run_id, text, name)

    def _log_checkpoint(
        self, index: int, path: Path, step: int, score: Mapping[str, Any] | None
    ) -> None:
        """Keep the latest checkpoint as a run tag.

        Args:
            index: The checkpoint's index in its store.
            path: Directory it was written to.
            step: Training step it was taken at.
            score: Its monitored value, with `monitor` and `mode`, if any.
        """
        record = {"index": index, "path": str(path), "step": step, "score": score}
        self._client.set_tag(self.run_id, "checkpoint", json.dumps(record))

    def _close(self, status: RunStatusTypes) -> None:
        """Send what is buffered and end the run as it ended.

        Args:
            status: `"finished"`, `"failed"` or `"cancelled"`.
        """
        self._flush()
        self._client.set_terminated(self.run_id, self.STATUSES[status])
