# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the experiment tracker base, through a recording backend."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from chuchichaestli.metrics import MSE
from chuchichaestli.programs import (
    ArchiveExport,
    Call,
    Checkpointer,
    Eval,
    EventType,
    ImageExport,
    Phase,
    Program,
    Runner,
    Signal,
    Tracker,
    TrackingPolicy,
    Train,
    WeightsExport,
)
from chuchichaestli.programs import trackers
from chuchichaestli.programs.ckpt import CheckpointStore
from chuchichaestli.programs.events import C3liRunCancelled, C3liRunError
from chuchichaestli.programs.serialize import stage_signature
from chuchichaestli.training import OptimSpec


class Recording(Tracker):
    """A tracker whose backend is a set of lists."""

    def __init__(self, new_id: str = "run-1", **kwargs):
        """Constructor.

        Args:
            new_id: Identifier a fresh backend run is given.
            kwargs: Passed on to `Tracker`.
        """
        super().__init__(**kwargs)
        self.new_id = new_id
        self.opens: list[tuple[dict, str | None, bool]] = []
        self.logs: list[tuple[dict, int]] = []
        self.files: list[tuple[str, str, list[str], int]] = []
        self.tags: list[tuple[str, ...]] = []
        self.configs: list[dict] = []
        self.checkpoints: list[tuple[int, int, dict | None]] = []
        self.closes: list[str] = []
        self.texts: list[tuple[str, str]] = []

    def _open(self, names, run_id, resumed):
        """Record the open and return the run's identifier."""
        self.opens.append((dict(names), run_id, resumed))
        return run_id or self.new_id

    def _log(self, metrics, step):
        """Record the numbers."""
        self.logs.append((dict(metrics), step))

    def _log_artifacts(self, kind, key, paths, step, tags=()):
        """Record the files by name, and their tags."""
        self.files.append((kind, key, [p.name for p in paths], step))
        self.tags.append(tuple(tags))

    def _log_config(self, config):
        """Record the config."""
        self.configs.append(dict(config))

    def _log_checkpoint(self, index, path, step, score):
        """Record the checkpoint."""
        self.checkpoints.append((index, step, score))

    def _log_text(self, name, text):
        """Record a stored text file."""
        self.texts.append((name, text))

    def _close(self, status):
        """Record how the run ended."""
        self.closes.append(status)

    def series(self, key: str) -> list[tuple[int, float]]:
        """Return a key's `(step, value)` points.

        Args:
            key: Metric key to read.
        """
        return [(step, m[key]) for m, step in self.logs if key in m]

    def keys(self) -> set[str]:
        """Return every metric key logged."""
        return {key for m, _ in self.logs for key in m}


class FailAt:
    """Stand in for a failure or a cancellation partway through a run."""

    def __init__(self, steps: int, error: type = C3liRunError):
        """Constructor.

        Args:
            steps: Training steps to allow before raising.
            error: What is raised.
        """
        self.steps = steps
        self.error = error
        self.seen = 0

    def on(self, event):
        """Raise once enough steps have gone by.

        Args:
            event: What the runner just did.

        Raises:
            C3liRunError: Once the run has taken `steps` steps.
        """
        if event.type is EventType.STEP_ENDED:
            self.seen += 1
            if self.seen >= self.steps:
                raise self.error("stopped")
        return Signal.GO


def rows() -> TensorDataset:
    """Build eight samples, the same every time."""
    generator = torch.Generator().manual_seed(0)
    return TensorDataset(
        torch.rand(8, 2, generator=generator), torch.rand(8, 1, generator=generator)
    )


def fitted(passes: int = 3) -> Program:
    """Build a program that trains a pass at a time and scores after each.

    Args:
        passes: Passes over the data; two steps each.
    """
    torch.manual_seed(1)
    fit = Train(
        "fit", data=rows(), batch_size=4, loss=nn.MSELoss(), optim=OptimSpec.sgd(lr=0.1)
    )
    check = Eval("check", data=rows(), batch_size=4, metrics=[MSE()])
    return Program(
        [Phase.each_pass(passes, fit, check)], provide={"model": nn.Linear(2, 1)}
    )


def run(program: Program, *hooks, **kwargs) -> None:
    """Run a program on the CPU with the given hooks.

    Args:
        program: What to run.
        hooks: The run's hooks.
        kwargs: Passed on to `Runner`.
    """
    Runner(program, hooks=list(hooks), device="cpu", **kwargs).run()


def test_a_run_logs_its_losses_rates_and_scores():
    """Training steps, each pass and each score land on one step axis."""
    tracker = Recording()
    run(fitted(), tracker)
    assert [step for step, _ in tracker.series("fit/loss")] == [1, 2, 3, 4, 5, 6]
    assert {value for _, value in tracker.series("fit/lr")} == {0.1}
    assert [step for step, _ in tracker.series("check/mse")] == [2, 4, 6]
    assert tracker.series("epoch") == [(2, 1), (4, 2), (6, 3)]


def test_every_thins_the_step_metrics_alone():
    """Scores and passes are already sparse, so only steps are skipped."""
    tracker = Recording(every=2)
    run(fitted(), tracker)
    assert [step for step, _ in tracker.series("fit/loss")] == [2, 4, 6]
    assert len(tracker.series("check/mse")) == 3
    assert len(tracker.series("epoch")) == 3


def test_timing_logs_each_pass_s_duration_and_throughput():
    """Eight samples a pass, so the rate is eight over the duration."""
    timed, untimed = Recording(), Recording(timing=False)
    run(fitted(), timed, untimed)
    seconds = dict(timed.series("fit/pass_seconds"))
    rates = dict(timed.series("fit/samples_per_second"))
    assert list(seconds) == list(rates) == [2, 4, 6]
    assert all(rates[s] == pytest.approx(8 / seconds[s]) for s in seconds)
    assert not {"fit/pass_seconds", "fit/samples_per_second"} & untimed.keys()


def test_memory_is_the_pass_s_peak_where_the_device_reports_one(monkeypatch):
    """The counter is reset as each pass begins and read as it ends."""

    class Counter:
        """A device memory counter peaking at 5 MiB each pass."""

        resets = 0

        def reset_peak_memory_stats(self, device):
            """Start a new peak."""
            Counter.resets += 1

        def max_memory_allocated(self, device):
            """Report the peak."""
            return 5 * 2**20

    monkeypatch.setattr(trackers, "_memory_api", lambda device: Counter())
    tracker, unmeasured = Recording(), Recording(memory=False)
    run(fitted(), tracker, unmeasured)
    assert [mb for _, mb in tracker.series("fit/peak_memory_mb")] == [5.0, 5.0, 5.0]
    assert Counter.resets >= 3
    assert "fit/peak_memory_mb" not in unmeasured.keys()


def test_the_cpu_reports_no_peak_memory():
    """There is no counter to read, so nothing is logged rather than a zero."""
    tracker = Recording()
    run(fitted(), tracker)
    assert "fit/peak_memory_mb" not in tracker.keys()


def test_an_eval_only_program_logs_at_step_zero():
    """Without training there is no step to place the scores at but the first."""
    tracker = Recording()
    program = Program(
        [Eval("check", data=rows(), batch_size=4, metrics=[MSE()])],
        provide={"model": nn.Linear(2, 1)},
    )
    run(program, tracker)
    assert [step for step, _ in tracker.series("check/mse")] == [0]


def test_a_fresh_run_opens_under_its_names_and_logs_its_config_once():
    """The runner names the run; the tracker passes the names on."""
    tracker = Recording(config={"width": 8, "dtype": torch.float32})
    program = fitted()
    run(program, tracker, seed=7, project="p", experiment="e", run_name="r")
    names = {"project": "p", "experiment": "e", "run_name": "r"}
    assert tracker.opens == [(names, None, False)]
    (config,) = tracker.configs
    assert config["config"] == {"width": 8, "dtype": "float32"}
    assert config["runner"]["seed"] == 7
    assert config["program"] == stage_signature(program)
    assert config["stages"]["program/0:repeat/0:epoch/0:fit"]["batch_size"] == 4
    assert config["models"]["model"] == {
        "cls": "Linear",
        "kwargs": None,
        "parameters": 3,
        "trainable": 3,
    }
    assert config["environment"]["torch"] == torch.__version__
    assert config["environment"]["device"] == "cpu"
    assert tracker.closes == ["finished"]


@pytest.mark.parametrize(
    ("option", "sections"),
    [
        ("runner", {"runner"}),
        ("stages", {"program", "stages"}),
        ("model", {"models"}),
        ("environment", {"environment"}),
    ],
)
def test_each_config_section_can_be_left_out(option, sections):
    """Switching one off removes exactly its own sections."""
    full, without = Recording(), Recording(**{option: False})
    run(fitted(), full, without)
    assert set(full.configs[0]) - set(without.configs[0]) == sections


def test_your_config_is_logged_whenever_it_is_given():
    """Passing it is the request; nothing is sent in its place when it is not."""
    given, absent = Recording(log="minimal", config={"lr": 0.1}), Recording()
    run(fitted(), given, absent)
    assert given.configs == [{"config": {"lr": 0.1}}]
    assert "config" not in absent.configs[0]


def test_the_minimal_preset_logs_metrics_and_how_the_run_ended():
    """No config, no pass details, no files: the curves and the outcome."""
    tracker = Recording(log="minimal")
    run(fitted(), tracker)
    assert tracker.configs == []
    assert "epoch" not in tracker.keys()
    assert "fit/pass_seconds" not in tracker.keys()
    assert {"fit/loss", "check/mse"} <= tracker.keys()
    assert tracker.closes == ["finished"]


def test_a_run_with_nothing_to_log_never_opens():
    """A stage ending without numbers makes no backend call."""
    tracker = Recording()
    run(Program([Call("noop", fn=lambda ctx: None)]), tracker)
    assert (tracker.opens, tracker.logs, tracker.configs) == ([], [], [])
    assert tracker.closes == []


def exporting(tmp_path) -> Program:
    """Build a program writing one artifact of every kind.

    Args:
        tmp_path: Directory the files are written to.
    """
    return Program(
        [
            ImageExport("plots", path=tmp_path / "img.png", source="images"),
            ArchiveExport("samples", path=tmp_path / "samples.h5", source="images"),
            WeightsExport("weights", path=tmp_path / "w.safetensors"),
        ],
        provide={"images": torch.rand(2, 1, 8, 8), "model": nn.Linear(2, 1)},
    )


def test_artifacts_arrive_by_kind_under_their_stage(tmp_path):
    """Each exporter's files reach the tracker with what they hold."""
    pytest.importorskip("matplotlib")
    tracker = Recording(artifacts=True)
    run(exporting(tmp_path), tracker)
    assert tracker.files == [
        ("image", "plots", ["img_0.png", "img_1.png"], 0),
        ("tensor", "samples", ["samples.h5"], 0),
        ("weights", "weights", ["w.safetensors"], 0),
    ]


@pytest.mark.parametrize(
    ("options", "kinds"),
    [
        ({}, ["image", "tensor"]),
        ({"artifacts": ("image",)}, ["image"]),
        ({"artifacts": False}, []),
        ({"log": "full"}, ["image", "tensor", "weights"]),
        ({"artifacts": ("checkpoint",)}, []),
    ],
)
def test_artifacts_selects_the_kinds_uploaded(tmp_path, options, kinds):
    """Weights are left out unless asked for, as every upload of them adds up."""
    pytest.importorskip("matplotlib")
    tracker = Recording(**options)
    run(exporting(tmp_path), tracker)
    assert [kind for kind, *_ in tracker.files] == kinds


def test_checkpoints_are_recorded_with_their_step_and_score(tmp_path):
    """The backend shows where the resumable states are, and how they scored."""
    tracker, unrecorded = Recording(), Recording(checkpoints=False)
    hook = Checkpointer(every=1, unit="epoch", monitor="mse")
    run(fitted(), hook, tracker, unrecorded, store=tmp_path / "run")
    steps = [step for _, step, _ in tracker.checkpoints]
    assert steps == sorted(steps) and steps[0] == 2
    scored = [score for *_, score in tracker.checkpoints if score is not None]
    assert scored and scored[0]["monitor"] == "mse"
    assert unrecorded.checkpoints == []


@pytest.mark.parametrize(
    ("error", "status"), [(C3liRunError, "failed"), (C3liRunCancelled, "cancelled")]
)
def test_the_backend_is_told_how_the_run_ended(error, status):
    """A cancelled run did not fail, so it is not reported as failed."""
    tracker = Recording()
    with pytest.raises(error):
        run(fitted(), tracker, FailAt(3, error))
    assert tracker.closes == [status]


def test_a_failing_backend_is_dropped_and_the_run_goes_on():
    """A tracker outage must not cost a training run."""

    class Down(Recording):
        """A backend that cannot be reached."""

        def _log(self, metrics, step):
            """Fail to reach the backend."""
            raise ConnectionError("unreachable")

    with pytest.warns(UserWarning, match="dropping it"):
        run(fitted(), Down())


def test_a_checkpoint_taken_before_any_log_names_the_backend_run(tmp_path):
    """A checkpointer ahead of the tracker still records which run to continue."""
    store = tmp_path / "run"
    tracker = Recording()
    run(fitted(), Checkpointer(every=1, unit="step"), tracker, store=store)
    first = CheckpointStore(store).load(1)
    assert first.hooks["Recording#0"] == {"run_id": "run-1"}


def test_a_resumed_run_reopens_its_backend_run_and_continues_the_axis(tmp_path):
    """Neither a second backend run nor a second config: the same run goes on."""
    store = tmp_path / "run"
    first = Recording()
    with pytest.raises(C3liRunError):
        run(
            fitted(),
            Checkpointer(every=1, unit="step"),
            first,
            FailAt(3),
            store=store,
            seed=5,
        )
    resumed = CheckpointStore(store).load("last").counts["step"]
    second = Recording(new_id="run-2")
    run(
        fitted(),
        Checkpointer(every=1, unit="step"),
        second,
        store=store,
        seed=5,
        resume="last",
    )
    ((names, run_id, was_resumed),) = second.opens
    assert (run_id, was_resumed) == ("run-1", True)
    assert names["run_name"] == first.opens[0][0]["run_name"]
    assert second.configs == []
    steps = [step for step, _ in second.series("fit/loss")]
    assert steps == list(range(resumed + 1, 7))


def test_a_policy_resolves_from_a_preset_and_options():
    """Options change fields on top of the preset they are given with."""
    assert TrackingPolicy.resolve() == TrackingPolicy()
    assert TrackingPolicy.resolve("full").kinds == {
        "image",
        "tensor",
        "weights",
        "checkpoint",
    }
    minimal = TrackingPolicy.resolve("minimal", timing=True)
    assert (minimal.timing, minimal.runner) == (True, False)
    shared = TrackingPolicy(every=10)
    assert Recording(log=shared).log == shared
    assert Recording(log=shared, memory=False).log.every == 10


def test_a_tracker_shows_how_it_was_configured():
    """What was changed, not every field of the policy."""
    assert repr(Recording()) == "Recording()"
    assert repr(Recording(log="full", every=10)) == "Recording(log='full', every=10)"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"every": 0}, "every"),
        ({"artifacts": ("video",)}, "artifact kind"),
        ({"checkpoint_uploads": "first"}, "upload mode"),
        ({"memroy": False}, "tracking option"),
        ({"log": "everything"}, "tracking preset"),
    ],
)
def test_a_tracker_refuses_what_it_cannot_honour(kwargs, match):
    """Typos fail at construction, not silently log nothing."""
    with pytest.raises(ValueError, match=match):
        Recording(**kwargs)


def uploads(store, *kinds, **options) -> Recording:
    """Train with a scoring checkpointer and return what the tracker uploaded.

    Args:
        store: Directory the checkpoints are written to.
        kinds: Artifact kinds to upload besides checkpoints.
        options: Further `TrackingPolicy` fields.
    """
    hook = Checkpointer(every=1, unit="epoch", monitor=options.pop("monitor", "mse"))
    tracker = Recording(artifacts=(*kinds, "checkpoint"), **options)
    run(fitted(), hook, tracker, store=store)
    return tracker


def test_a_checkpointer_announces_each_checkpoint_it_writes(tmp_path):
    """Every file of the directory, and whether it is the best so far."""
    seen: list[dict] = []

    class Exports:
        """Records `EXPORTED` payloads."""

        def on(self, event):
            """Record a checkpoint's announcement.

            Args:
                event: What the runner just did.
            """
            if event.type is EventType.EXPORTED:
                seen.append(dict(event.payload))
            return Signal.GO

    hook = Checkpointer(every=1, unit="epoch", monitor="mse")
    run(fitted(), hook, Exports(), store=tmp_path / "run")
    assert seen and all(p["kind"] == "checkpoint" for p in seen)
    first = seen[0]
    names = {Path(f).name for f in first["files"]}
    assert {"state.safetensors", "model.safetensors"} <= names
    assert any(p["best"] for p in seen)


def test_checkpoints_are_not_uploaded_by_default(tmp_path):
    """Optimizer state makes them large, so they are opt-in like weights."""
    tracker = Recording()
    run(fitted(), Checkpointer(every=1, unit="epoch"), tracker, store=tmp_path)
    assert tracker.files == []


def test_all_uploads_each_checkpoint_as_it_is_written(tmp_path):
    """One upload per checkpoint recorded, at the step it was taken."""
    tracker = uploads(tmp_path, checkpoint_uploads="all")
    assert [step for *_, step in tracker.files] == [
        s for _, s, _ in tracker.checkpoints
    ]


def test_best_uploads_each_new_best_and_tags_it(tmp_path):
    """Only improvements go up, and the backend is told which they are."""
    tracker = uploads(tmp_path)
    assert tracker.files and all(tags == ("best",) for tags in tracker.tags)


def test_best_without_a_monitor_uploads_the_latest_as_the_run_ends(tmp_path):
    """Nothing is ever best, so nothing would go up; the latest one does."""
    tracker = Recording(artifacts=("checkpoint",))
    run(fitted(), Checkpointer(every=1, unit="epoch"), tracker, store=tmp_path)
    assert [step for *_, step in tracker.files] == [6]
    assert tracker.tags == [()]


def test_last_uploads_one_checkpoint_as_the_run_ends(tmp_path):
    """However many were written, one goes up: the latest."""
    tracker = uploads(tmp_path, checkpoint_uploads="last")
    assert [step for *_, step in tracker.files] == [6]


def test_a_pruned_latest_checkpoint_is_reported_not_uploaded(tmp_path):
    """The final write prunes what the tracker kept back, if it runs first."""
    tracker = Recording(artifacts=("checkpoint",), checkpoint_uploads="last")
    hook = Checkpointer(every=1, unit="epoch", keep=1)
    with pytest.warns(UserWarning, match="pruned before it could be uploaded"):
        run(fitted(), hook, tracker, store=tmp_path)
    assert tracker.files == []


def committed(root):
    """Make a repository with one committed file, and return it.

    Args:
        root: Directory the repository is made in.
    """
    git = pytest.importorskip("git")
    author = git.Actor("Test", "test@example.com")
    repo = git.Repo.init(root)
    (root / "a.txt").write_text("one\n")
    repo.index.add(["a.txt"])
    repo.index.commit("init", author=author, committer=author)
    return repo


def test_the_repository_s_commit_and_dirtiness_are_logged(tmp_path, monkeypatch):
    """A clean repository: the commit identifies the code, and no patch is stored."""
    repo = committed(tmp_path)
    monkeypatch.chdir(tmp_path)
    tracker = Recording()
    run(fitted(), tracker)
    git = tracker.configs[0]["environment"]["git"]
    assert git == {"commit": repo.head.commit.hexsha, "dirty": False}
    assert tracker.texts == []


def test_a_dirty_repository_s_diff_is_stored_as_a_patch(tmp_path, monkeypatch):
    """The commit alone would not say what ran; the patch does."""
    committed(tmp_path)
    (tmp_path / "a.txt").write_text("one\ntwo\n")
    monkeypatch.chdir(tmp_path)
    tracker = Recording()
    run(fitted(), tracker)
    assert tracker.configs[0]["environment"]["git"]["dirty"] is True
    ((name, patch),) = tracker.texts
    assert name == "git.patch" and "+two" in patch


def test_outside_a_repository_no_git_state_is_logged(tmp_path, monkeypatch):
    """Nothing known, so nothing logged, rather than a commit of None."""
    monkeypatch.chdir(tmp_path)
    tracker = Recording()
    run(fitted(), tracker)
    assert "git" not in tracker.configs[0]["environment"]


def test_without_gitpython_the_commit_alone_is_logged(tmp_path, monkeypatch):
    """Read from the repository's files; the warning says what is missing."""
    sha = "a" * 40
    (tmp_path / ".git" / "refs" / "heads").mkdir(parents=True)
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / ".git" / "refs" / "heads" / "main").write_text(sha + "\n")
    monkeypatch.setitem(sys.modules, "git", None)
    monkeypatch.chdir(tmp_path)
    tracker = Recording()
    with pytest.warns(UserWarning, match=r"chuchichaestli\[git\]"):
        run(fitted(), tracker)
    assert tracker.configs[0]["environment"]["git"] == {"commit": sha}
    assert tracker.texts == []


def test_without_gitpython_outside_a_repository_nothing_is_missed(
    tmp_path, monkeypatch, recwarn
):
    """No repository, so nothing is lost and there is nothing to warn about."""
    monkeypatch.setitem(sys.modules, "git", None)
    monkeypatch.chdir(tmp_path)
    tracker = Recording()
    run(fitted(), tracker)
    assert "git" not in tracker.configs[0]["environment"]
    assert not [w for w in recwarn if "chuchichaestli[git]" in str(w.message)]


def test_no_git_state_is_read_without_the_environment_section(tmp_path, monkeypatch):
    """Switched off means not looked at, the patch included."""
    committed(tmp_path)
    (tmp_path / "a.txt").write_text("changed\n")
    monkeypatch.chdir(tmp_path)
    tracker = Recording(environment=False)
    run(fitted(), tracker)
    assert tracker.texts == []
