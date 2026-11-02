# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for logging a run to MLflow."""

import sys
import types
from collections import namedtuple

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from chuchichaestli.metrics import MSE
from chuchichaestli.programs import (
    Checkpointer,
    Eval,
    EventType,
    MLflow,
    Phase,
    Program,
    Runner,
    Signal,
    Train,
)
from chuchichaestli.programs.events import C3liRunCancelled, C3liRunError
from chuchichaestli.training import OptimSpec

Metric = namedtuple("Metric", "key value timestamp step")
Param = namedtuple("Param", "key value")


class Client:
    """An `MlflowClient` that records what it is asked to do."""

    made: list["Client"] = []

    def __init__(self, tracking_uri=None):
        """Constructor.

        Args:
            tracking_uri: Where runs would be tracked.
        """
        self.tracking_uri = tracking_uri
        self.experiments: list[tuple[str, str | None]] = []
        self.runs: list[dict] = []
        self.metric_batches: list[list[Metric]] = []
        self.param_batches: list[list[Param]] = []
        self.artifacts: list[tuple[str, str]] = []
        self.tags: dict[str, str] = {}
        self.updates: list[tuple[str, str]] = []
        self.terminated: list[str] = []
        Client.made.append(self)

    def get_experiment_by_name(self, name):
        """Find nothing, so the experiment is created."""
        return None

    def create_experiment(self, name, artifact_location=None):
        """Record the experiment."""
        self.experiments.append((name, artifact_location))
        return "7"

    def create_run(self, experiment_id, run_name=None, tags=None):
        """Record the run."""
        self.runs.append({"experiment": experiment_id, "name": run_name, "tags": tags})
        return types.SimpleNamespace(info=types.SimpleNamespace(run_id="run-1"))

    def update_run(self, run_id, status=None):
        """Record a status change."""
        self.updates.append((run_id, status))

    def log_batch(self, run_id, metrics=(), params=()):
        """Record a batch."""
        if metrics:
            self.metric_batches.append(list(metrics))
        if params:
            self.param_batches.append(list(params))

    def log_artifact(self, run_id, local_path, artifact_path=None):
        """Record an artifact."""
        self.artifacts.append((artifact_path, local_path))

    def log_text(self, run_id, text, artifact_file):
        """Record a text artifact."""
        self.artifacts.append((artifact_file, text))

    def set_tag(self, run_id, key, value):
        """Record a tag."""
        self.tags[key] = value

    def set_terminated(self, run_id, status=None):
        """Record the end."""
        self.terminated.append(status)


@pytest.fixture
def clients(monkeypatch):
    """Replace `mlflow` with recording modules, and return the clients made."""
    Client.made = []
    registry = types.ModuleType("mlflow.tracking.context.registry")
    registry.resolve_tags = lambda tags: {**tags, "mlflow.user": "someone"}
    modules = {
        "mlflow": types.ModuleType("mlflow"),
        "mlflow.tracking": types.SimpleNamespace(MlflowClient=Client),
        "mlflow.tracking.context": types.ModuleType("mlflow.tracking.context"),
        "mlflow.tracking.context.registry": registry,
        "mlflow.entities": types.SimpleNamespace(Metric=Metric, Param=Param),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    return Client.made


def rows() -> TensorDataset:
    """Build eight samples, the same every time."""
    generator = torch.Generator().manual_seed(0)
    return TensorDataset(
        torch.rand(8, 2, generator=generator), torch.rand(8, 1, generator=generator)
    )


def fitted(passes: int = 3) -> Program:
    """Build a program training a pass at a time, scoring after each.

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


def test_by_default_runs_live_in_one_sqlite_file_in_the_store(tmp_path, clients):
    """Runs, metrics and params in one database; files beside it."""
    store = tmp_path / "run"
    run(fitted(1), MLflow(), store=store)
    client = clients[0]
    assert client.tracking_uri == f"sqlite:///{store.resolve() / 'mlflow.db'}"
    assert client.experiments == [
        ("chuchichaestli", (store.resolve() / "mlartifacts").as_uri())
    ]


def test_a_given_or_configured_tracking_uri_is_used_as_it_is(clients, monkeypatch):
    """So any store plugin serves it, and no local store is needed."""
    assert MLflow().needs_store and not MLflow(tracking_uri="http://x").needs_store
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://server:5000")
    tracker = MLflow()
    assert not tracker.needs_store
    run(fitted(1), tracker)
    assert clients[0].tracking_uri == "http://server:5000"
    assert clients[0].experiments == [("chuchichaestli", None)]


@pytest.mark.parametrize(
    ("names", "experiment"),
    [
        ({"project": "p", "experiment": "e"}, "e"),
        ({"project": "p"}, "p"),
        ({}, "chuchichaestli"),
    ],
)
def test_the_experiment_falls_back_from_experiment_to_project(
    tmp_path, clients, names, experiment
):
    """MLflow has no project level, so the project is a tag and a fallback."""
    run(fitted(1), MLflow(tags={"team": "t"}), store=tmp_path, run_name="r", **names)
    client = clients[0]
    assert client.experiments[0][0] == experiment
    (created,) = client.runs
    assert created["name"] == "r"
    assert (
        created["tags"]["team"] == "t" and created["tags"]["mlflow.user"] == "someone"
    )
    assert created["tags"].get("project") == names.get("project")


def test_numbers_are_sent_in_batches_not_one_call_per_step(tmp_path, clients):
    """A remote server is spared a request per step; nothing is lost at the end."""
    run(fitted(), MLflow(), store=tmp_path)
    batches = clients[0].metric_batches
    assert len(batches) == 1
    steps = [m.step for m in batches[0] if m.key == "fit/loss"]
    assert steps == [1, 2, 3, 4, 5, 6]


def test_numbers_are_sent_as_they_come_once_a_batch_is_due(
    tmp_path, clients, monkeypatch
):
    """With no wait between batches, each log call is its own batch."""
    monkeypatch.setattr(MLflow, "FLUSH_SECONDS", 0.0)
    run(fitted(1), MLflow(), store=tmp_path)
    assert len(clients[0].metric_batches) > 1


def test_the_config_becomes_parameters_in_batches_within_limits(tmp_path, clients):
    """Flattened, as strings, at most 6000 characters, 100 to a call."""
    config = {f"k{i}": i for i in range(150)} | {"long": "x" * 7000}
    run(fitted(1), MLflow(config=config), store=tmp_path, seed=3)
    params = [p for batch in clients[0].param_batches for p in batch]
    assert all(len(batch) <= 100 for batch in clients[0].param_batches)
    by_key = {p.key: p.value for p in params}
    assert by_key["config/k5"] == "5" and by_key["runner/seed"] == "3"
    assert len(by_key["config/long"]) == 6000


def test_checkpoints_land_under_their_kind_and_step_and_the_best_is_tagged(
    tmp_path, clients
):
    """MLflow shows the files by folder; a tag points at the best one."""
    tracker = MLflow(artifacts=("checkpoint",))
    run(
        fitted(),
        Checkpointer(every=1, unit="epoch", monitor="mse"),
        tracker,
        store=tmp_path,
    )
    client = clients[0]
    folders = {folder for folder, _ in client.artifacts}
    assert client.tags["best_checkpoint"] in folders
    assert all(
        f.startswith("checkpoint/checkpoint/step_") for f in folders - {"git.patch"}
    )
    assert '"step": 6' in client.tags["checkpoint"]


def test_a_resumed_run_is_marked_running_again_not_created(tmp_path, clients):
    """The same MLflow run goes on, without a second set of parameters."""
    store = tmp_path / "store"
    with pytest.raises(C3liRunError):
        run(
            fitted(),
            Checkpointer(every=1, unit="step"),
            MLflow(),
            FailAt(3),
            store=store,
            seed=5,
        )
    run(
        fitted(),
        Checkpointer(every=1, unit="step"),
        MLflow(),
        store=store,
        seed=5,
        resume="last",
    )
    second = clients[1]
    assert second.updates == [("run-1", "RUNNING")]
    assert second.runs == [] and second.param_batches == []


@pytest.mark.parametrize(
    ("error", "status"),
    [(None, "FINISHED"), (C3liRunError, "FAILED"), (C3liRunCancelled, "KILLED")],
)
def test_the_run_ends_with_mlflow_s_status_for_how_it_ended(
    tmp_path, clients, error, status
):
    """A cancelled run is killed, not failed."""
    hooks = [MLflow()] + ([FailAt(2, error)] if error else [])
    if error:
        with pytest.raises(error):
            run(fitted(), *hooks, store=tmp_path)
    else:
        run(fitted(), *hooks, store=tmp_path)
    assert clients[0].terminated == [status]


def test_without_mlflow_the_tracker_is_dropped_naming_the_extra(tmp_path, monkeypatch):
    """The run goes on; the warning says what to install."""
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with pytest.warns(UserWarning, match=r"chuchichaestli\[mlflow\]"):
        run(fitted(1), MLflow(), store=tmp_path)


def test_a_run_reads_back_from_mlflow_s_own_database(tmp_path, monkeypatch):
    """Against the real thing: the SQLite store holds the losses at their steps."""
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    run(fitted(), MLflow(config={"lr": 0.1}), store=tmp_path, run_name="r")
    client = mlflow.tracking.MlflowClient(f"sqlite:///{tmp_path / 'mlflow.db'}")
    (found,) = client.search_runs(
        [client.get_experiment_by_name("chuchichaestli").experiment_id]
    )
    history = client.get_metric_history(found.info.run_id, "fit/loss")
    assert sorted(m.step for m in history) == [1, 2, 3, 4, 5, 6]
    assert found.info.status == "FINISHED" and found.data.params["config/lr"] == "0.1"
