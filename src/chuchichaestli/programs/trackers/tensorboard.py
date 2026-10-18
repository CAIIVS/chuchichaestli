# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Logging a run to TensorBoard."""

from __future__ import annotations
import json
import warnings
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

__all__ = ["TensorBoard"]


class TensorBoard(Tracker):
    """Log a run to TensorBoard, as one event directory per run."""

    # TensorBoard draws these; it stores no files, so nothing else is logged
    DRAWN_FORMATS: ClassVar[frozenset[str]] = frozenset(
        {".png", ".jpg", ".jpeg", ".webp"}
    )

    def __init__(
        self,
        log_dir: str | Path | None = None,
        log: TrackingPolicy | PresetTypes = "default",
        config: Mapping[str, Any] | None = None,
        **options: Any,
    ):
        """Constructor.

        Args:
            log_dir: Directory holding every run, as
                `<log_dir>/<project>/<experiment>/<run_name>`; the run's store's
                `tensorboard` directory when absent.
            log: What to log, as for `Tracker`.
            config: Your own configuration, as for `Tracker`.
            options: `TrackingPolicy` fields changed on top of `log`.
        """
        super().__init__(log=log, config=config, **options)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.needs_store = self.log_dir is None
        self._writer: Any = None
        self._skipped: set[str] = set()

    def _open(
        self, names: Mapping[str, str | None], run_id: str | None, resumed: bool
    ) -> str:
        """Open the run's event directory, purging steps past a resumed checkpoint.

        Args:
            names: The run's `project`, `experiment` and `run_name`.
            run_id: The run's event directory, when resumed.
            resumed: Whether the run continues one a checkpoint recorded.
        """
        tensorboard = require_module(
            "torch.utils.tensorboard", "tensorboard", "logging to TensorBoard"
        )
        if run_id is None:
            root = self.log_dir or Path(self.store or ".") / "tensorboard"
            parts = (names.get(k) for k in ("project", "experiment", "run_name"))
            run_id = str(root.joinpath(*(part for part in parts if part)))
        purge = self._count("step") + 1 if resumed else None
        self._writer = tensorboard.SummaryWriter(log_dir=run_id, purge_step=purge)
        return run_id

    def _log(self, metrics: Mapping[str, float | int], step: int) -> None:
        """Log each number as a scalar.

        Args:
            metrics: Values keyed `<stage>/<name>`, or `epoch`.
            step: Training step they belong to.
        """
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, step)

    def _log_artifacts(
        self,
        kind: ArtifactTypes,
        key: str,
        paths: Sequence[Path],
        step: int,
        tags: Sequence[str] = (),
    ) -> None:
        """Draw images TensorBoard can show, and skip the rest with a warning.

        Args:
            kind: What the files hold.
            key: Name of the stage that wrote them.
            paths: The files.
            step: Training step they belong to.
            tags: Unused; TensorBoard keeps no versions.
        """
        from chuchichaestli.data.image import ImageTensorView

        drawn = [p for p in paths if p.suffix.lower() in self.DRAWN_FORMATS]
        if kind != "image" or len(drawn) < len(paths):
            skipped = f"{kind!r} artifacts" if kind != "image" else "vector images"
            if skipped not in self._skipped:
                self._skipped.add(skipped)
                warnings.warn(
                    f"TensorBoard stores no files, so {skipped} are not logged.",
                    stacklevel=2,
                )
        if kind == "image":
            for index, path in enumerate(drawn):
                image = ImageTensorView(path)[0][:3]
                self._writer.add_image(f"{key}/{index}", image, step)

    def _log_config(self, config: Mapping[str, Any]) -> None:
        """Log the config's scalars as hyperparameters, and the whole as text.

        Args:
            config: The run's configuration, nested and JSON-safe.
        """
        scalars = {
            key: value
            for key, value in flatten_tree(config).items()
            if isinstance(value, (bool, int, float, str))
        }
        if scalars:
            # "." keeps the hyperparameters in the run's own directory
            self._writer.add_hparams(scalars, {}, run_name=".")
        text = json.dumps(config, indent=2)
        self._writer.add_text("config", f"```json\n{text}\n```")

    def _log_text(self, name: str, text: str) -> None:
        """Store a text file as a text summary.

        Args:
            name: Its name, e.g. `"git.patch"`.
            text: Its contents.
        """
        self._writer.add_text(name, f"```\n{text}\n```")

    def _log_checkpoint(
        self, index: int, path: Path, step: int, score: Mapping[str, Any] | None
    ) -> None:
        """Note a checkpoint as text at its step.

        Args:
            index: The checkpoint's index in its store.
            path: Directory it was written to.
            step: Training step it was taken at.
            score: Its monitored value, with `monitor` and `mode`, if any.
        """
        scored = "" if score is None else f", {score['monitor']} {score['value']:.4g}"
        self._writer.add_text("checkpoints", f"index {index}{scored}: `{path}`", step)

    def _close(self, status: RunStatusTypes) -> None:
        """Note how the run ended, and close its event files.

        Args:
            status: `"finished"`, `"failed"` or `"cancelled"`.
        """
        self._writer.add_text("status", status)
        self._writer.close()
        self._writer = None
