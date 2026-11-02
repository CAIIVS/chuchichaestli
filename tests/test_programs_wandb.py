# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for logging a run to Weights & Biases."""

import sys
import types

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from chuchichaestli.metrics import MSE
from chuchichaestli.programs import (
    ArchiveExport,
    Checkpointer,
    Eval,
    EventType,
    ImageExport,
    Phase,
    Program,
    Runner,
    Signal,
    Train,
    Wandb,
)
from chuchichaestli.programs.events import C3liRunCancelled, C3liRunError
from chuchichaestli.training import OptimSpec


class Run:
    """A W&B run that records what it is asked to do."""

    def __init__(self, directory, step=0, **init):
        """Constructor.

        Args:
            directory: The run's own directory.
            step: The step W&B holds the run at.
            init: What `wandb.init` was called with.
        """
        self.id, self.step, self.dir, self.init = "abc123", step, str(directory), init
        self.logs: list[tuple[dict, int]] = []
        self.artifacts: list[tuple[object, list[str]]] = []
        self.saved: list[str] = []
        self.config = types.SimpleNamespace(values={}, update=self._config)
        self.summary: dict = {}
        self.preempting = False
        self.exit_code = None

    def _config(self, values, allow_val_change=False):
        self.config.values.update(values)

    def log(self, data, step=None):
        """Record a log call."""
        self.logs.append((dict(data), step))

    def log_artifact(self, artifact, aliases=None):
        """Record an artifact."""
        self.artifacts.append((artifact, list(aliases or [])))

    def save(self, path, base_path=None, policy="live"):
        """Record a saved file."""
        self.saved.append(path)

    def mark_preempting(self):
        """Record the preemption."""
        self.preempting = True

    def finish(self, exit_code=None):
        """Record the end."""
        self.exit_code = exit_code


class Artifact:
    """A W&B artifact that records its files."""

    def __init__(self, name, type):
        """Constructor."""
        self.name, self.type, self.files = name, type, []

    def add_file(self, path):
        """Record a file."""
        self.files.append(path)


@pytest.fixture
def runs(monkeypatch, tmp_path):
    """Replace `wandb` with a recording module, and return the runs it makes."""
    made = Runs()
    resume_step = {"step": 0}

    def init(**kwargs):
        run = Run(tmp_path / f"run{len(made)}", step=resume_step["step"], **kwargs)
        (tmp_path / f"run{len(made)}").mkdir()
        made.append(run)
        return run

    module = types.SimpleNamespace(
        init=init, Image=lambda path: ("image", path), Artifact=Artifact
    )
    monkeypatch.setitem(sys.modules, "wandb", module)
    made.resume_step = resume_step
    return made


class Runs(list):
    """The runs made, and the step a resumed one starts W&B at."""


def rows() -> TensorDataset:
    """Build eight samples, the same every time."""
    generator = torch.Generator().manual_seed(0)
    return TensorDataset(
        torch.rand(8, 2, generator=generator), torch.rand(8, 1, generator=generator)
    )


def fitted(passes: int = 3, *after) -> Program:
    """Build a program training a pass at a time, scoring after each.

    Args:
        passes: Passes over the data; two steps each.
        after: Further stages to run after each pass.
    """
    torch.manual_seed(1)
    fit = Train(
        "fit", data=rows(), batch_size=4, loss=nn.MSELoss(), optim=OptimSpec.sgd(lr=0.1)
    )
    check = Eval("check", data=rows(), batch_size=4, metrics=[MSE()])
    return Program(
        [Phase.each_pass(passes, fit, check, *after)],
        provide={"model": nn.Linear(2, 1), "images": torch.rand(2, 1, 8, 8)},
    )


def run(program, *hooks, **kwargs) -> None:
    """Run a program on the CPU.

    Args:
        program: What to run.
        hooks: The run's hooks.
        kwargs: Passed on to `Runner`.
    """
    Runner(program, hooks=list(hooks), device="cpu", **kwargs).run()


class FailAt:
    """Raise after a number of training steps."""

    def __init__(self, steps, error=C3liRunError):
        """Constructor.

        Args:
            steps: Steps to allow.
            error: What is raised.
        """
        self.steps, self.error, self.seen = steps, error, 0

    def on(self, event):
        """Raise once enough steps have gone by.

        Args:
            event: What the runner just did.
        """
        if event.type is EventType.STEP_ENDED:
            self.seen += 1
            if self.seen == self.steps:
                raise self.error("stopped")
        return Signal.GO


def test_the_run_s_names_map_onto_project_group_and_name(tmp_path, runs):
    """W&B has no experiment level; a group is what holds related runs."""
    tracker = Wandb(entity="team", mode="offline", init={"tags": ["a"]})
    run(fitted(1), tracker, project="p", experiment="e", run_name="r", store=tmp_path)
    init = runs[0].init
    assert (init["entity"], init["project"], init["group"], init["name"]) == (
        "team",
        "p",
        "e",
        "r",
    )
    assert init["mode"] == "offline" and init["tags"] == ["a"]
    assert (init["id"], init["resume"], init["dir"]) == (None, None, str(tmp_path))


def test_an_init_argument_the_tracker_sets_itself_is_refused():
    """Two sources for one setting would leave one of them silently ignored."""
    with pytest.raises(ValueError, match=r"\['id', 'name'\]"):
        Wandb(init={"name": "x", "id": "y"})


def test_numbers_and_the_config_reach_the_run(runs):
    """Through the run object, never the module-level `wandb.log`."""
    run(fitted(), Wandb(config={"lr": 0.1}), seed=3)
    logged = runs[0].logs
    assert [s for m, s in logged if "fit/loss" in m] == [1, 2, 3, 4, 5, 6]
    assert runs[0].config.values["config"] == {"lr": 0.1}
    assert runs[0].config.values["runner"]["seed"] == 3


def test_images_reach_the_panels_and_every_kind_is_versioned(tmp_path, runs):
    """An artifact per stage, aliased by step, typed by what it holds."""
    pytest.importorskip("matplotlib")
    after = (
        ImageExport(
            "plots", path=tmp_path / "out" / "img_{epoch}.png", source="images"
        ),
        ArchiveExport("save", path=tmp_path / "out" / "s_{epoch}.h5", source="images"),
    )
    run(fitted(1, *after), Wandb())
    panels = [m["plots"] for m, _ in runs[0].logs if "plots" in m]
    assert len(panels) == 1 and all(kind == "image" for kind, _ in panels[0])
    versioned = [(a.name, a.type, aliases) for a, aliases in runs[0].artifacts]
    assert versioned == [
        ("plots-abc123", "image", ["step-2"]),
        ("save-abc123", "tensor", ["step-2"]),
    ]


def test_the_best_checkpoint_is_aliased_best(tmp_path, runs):
    """So the backend can hand back the best model by name."""
    tracker = Wandb(artifacts=("checkpoint",))
    run(
        fitted(),
        Checkpointer(every=1, unit="epoch", monitor="mse"),
        tracker,
        store=tmp_path,
    )
    aliases = [aliases for a, aliases in runs[0].artifacts if a.type == "checkpoint"]
    assert aliases and all("best" in a for a in aliases)
    assert runs[0].summary["checkpoint"]["step"] == 6


def test_a_dirty_repository_s_patch_is_saved_with_the_run(tmp_path, runs, monkeypatch):
    """A file in the run, beside W&B's own."""
    git = pytest.importorskip("git")
    author = git.Actor("Test", "test@example.com")
    repo = git.Repo.init(tmp_path / "repo")
    (tmp_path / "repo" / "a.txt").write_text("one\n")
    repo.index.add(["a.txt"])
    repo.index.commit("init", author=author, committer=author)
    (tmp_path / "repo" / "a.txt").write_text("two\n")
    monkeypatch.chdir(tmp_path / "repo")
    run(fitted(1), Wandb())
    (saved,) = runs[0].saved
    assert saved.endswith("git.patch") and "+two" in open(saved).read()


def test_a_resumed_run_reattaches_and_skips_the_steps_w_b_has(tmp_path, runs):
    """W&B refuses earlier steps, so they are left out rather than warned about."""
    store = tmp_path / "store"
    with pytest.raises(C3liRunError):
        run(
            fitted(),
            Checkpointer(every=1, unit="step"),
            Wandb(),
            FailAt(3),
            store=store,
            seed=5,
        )
    runs.resume_step["step"] = 4
    run(
        fitted(),
        Checkpointer(every=1, unit="step"),
        Wandb(),
        store=store,
        seed=5,
        resume="last",
    )
    second = runs[1]
    assert (second.init["id"], second.init["resume"]) == ("abc123", "must")
    assert min(s for _, s in second.logs) >= 4
    assert second.config.values == {}


@pytest.mark.parametrize(
    ("error", "code", "preempted"),
    [(None, 0, False), (C3liRunError, 1, False), (C3liRunCancelled, 1, True)],
)
def test_the_run_ends_as_it_ended(runs, error, code, preempted):
    """A cancelled run is marked preempted, so W&B tells it from a failure."""
    hooks = [Wandb()] + ([FailAt(2, error)] if error else [])
    if error:
        with pytest.raises(error):
            run(fitted(), *hooks)
    else:
        run(fitted(), *hooks)
    assert (runs[0].exit_code, runs[0].preempting) == (code, preempted)


def test_without_wandb_the_tracker_is_dropped_naming_the_extra(monkeypatch):
    """The run goes on; the warning says what to install."""
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.warns(UserWarning, match=r"chuchichaestli\[wandb\]"):
        run(fitted(1), Wandb())


def test_an_offline_run_completes_against_w_b_itself(tmp_path, monkeypatch):
    """Against the real thing, offline: the run starts, logs and finishes."""
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_SILENT", "true")
    run(fitted(1), Wandb(mode="offline"), store=tmp_path, run_name="r")
    assert list((tmp_path / "wandb").glob("offline-run-*"))
