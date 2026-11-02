# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for logging a run to TensorBoard."""

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
    ImageExport,
    Phase,
    Program,
    Runner,
    TensorBoard,
    Train,
)
from chuchichaestli.programs.events import C3liRunError
from chuchichaestli.training import OptimSpec


class Writer:
    """A `SummaryWriter` that records what it is asked to write."""

    made: list["Writer"] = []

    def __init__(self, log_dir=None, purge_step=None):
        """Constructor.

        Args:
            log_dir: Where the events would go.
            purge_step: First step to drop on reopening.
        """
        self.log_dir, self.purge_step = log_dir, purge_step
        self.scalars: list[tuple[str, float, int]] = []
        self.images: list[tuple[str, tuple[int, ...], int]] = []
        self.texts: list[tuple[str, str]] = []
        self.hparams: list[dict] = []
        self.closed = False
        Writer.made.append(self)

    def add_scalar(self, key, value, step):
        """Record a scalar."""
        self.scalars.append((key, value, step))

    def add_image(self, key, image, step):
        """Record an image by its shape."""
        self.images.append((key, tuple(image.shape), step))

    def add_text(self, key, text, step=None):
        """Record a text."""
        self.texts.append((key, text))

    def add_hparams(self, hparams, metrics, run_name=None):
        """Record hyperparameters, and where they would go."""
        assert run_name == "."
        self.hparams.append(dict(hparams))

    def close(self):
        """Record the close."""
        self.closed = True


@pytest.fixture
def writers(monkeypatch):
    """Replace TensorBoard's writer with a recording one, and return those made."""
    Writer.made = []
    module = types.SimpleNamespace(SummaryWriter=Writer)
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", module)
    return Writer.made


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


def test_each_run_has_its_own_directory_under_its_names(tmp_path, writers):
    """TensorBoard lists runs by subdirectory, so one directory holds a project."""
    run(fitted(1), TensorBoard(log_dir=tmp_path), project="p", run_name="r")
    assert writers[0].log_dir == str(tmp_path / "p" / "r")
    assert writers[0].purge_step is None


def test_without_a_directory_the_run_s_store_holds_the_events(tmp_path, writers):
    """The store is the run's home; the events go beside its checkpoints."""
    run(fitted(1), TensorBoard(), store=tmp_path / "run", run_name="r")
    assert writers[0].log_dir == str(tmp_path / "run" / "tensorboard" / "r")


def test_without_a_directory_or_a_store_the_run_refuses_to_start():
    """There is nowhere to write, which is better found before training."""
    assert TensorBoard().needs_store and not TensorBoard(log_dir="x").needs_store
    with pytest.raises(Exception, match="store"):
        run(fitted(1), TensorBoard())


def test_numbers_become_scalars_on_the_step_axis(tmp_path, writers):
    """The losses, the rate and the scores, each a series of its own."""
    run(fitted(), TensorBoard(log_dir=tmp_path))
    scalars = writers[0].scalars
    assert [s for k, _, s in scalars if k == "fit/loss"] == [1, 2, 3, 4, 5, 6]
    assert [s for k, _, s in scalars if k == "check/mse"] == [2, 4, 6]


def test_the_config_becomes_hyperparameters_and_text(tmp_path, writers):
    """Scalars go to the HParams tab, comparable across runs; the rest as text."""
    run(fitted(1), TensorBoard(log_dir=tmp_path, config={"lr": 0.1}), seed=3)
    writer = writers[0]
    (hparams,) = writer.hparams
    assert hparams["config/lr"] == 0.1 and hparams["runner/seed"] == 3
    assert all(isinstance(v, (bool, int, float, str)) for v in hparams.values())
    assert any(key == "config" and '"lr": 0.1' in text for key, text in writer.texts)


def test_images_are_drawn_and_other_kinds_skipped_with_one_warning(tmp_path, writers):
    """TensorBoard stores no files, so it shows what it can draw and says so."""
    pytest.importorskip("matplotlib")
    after = (
        ImageExport(
            "plots", path=tmp_path / "out" / "img_{epoch}.png", source="images"
        ),
        ArchiveExport("save", path=tmp_path / "out" / "s_{epoch}.h5", source="images"),
    )
    with pytest.warns(UserWarning, match="'tensor' artifacts are not logged") as seen:
        run(fitted(2, *after), TensorBoard(log_dir=tmp_path / "tb"))
    images = writers[0].images
    assert [(key, step) for key, _, step in images] == [
        ("plots/0", 2),
        ("plots/1", 2),
        ("plots/0", 4),
        ("plots/1", 4),
    ]
    assert images[0][1][0] in (1, 3)
    assert sum("not logged" in str(w.message) for w in seen) == 1


def test_a_resumed_run_reopens_its_directory_and_purges_later_steps(tmp_path, writers):
    """Steps past the checkpoint were logged by the interrupted run; they go."""
    store = tmp_path / "run"

    class FailAt:
        """Stop the run after three steps."""

        seen = 0

        def on(self, event):
            """Fail on the third step.

            Args:
                event: What the runner just did.
            """
            from chuchichaestli.programs import EventType, Signal

            if event.type is EventType.STEP_ENDED:
                FailAt.seen += 1
                if FailAt.seen == 3:
                    raise C3liRunError("stopped")
            return Signal.GO

    with pytest.raises(C3liRunError):
        run(
            fitted(),
            Checkpointer(every=1, unit="step"),
            TensorBoard(),
            FailAt(),
            store=store,
            seed=5,
        )
    run(
        fitted(),
        Checkpointer(every=1, unit="step"),
        TensorBoard(),
        store=store,
        seed=5,
        resume="last",
    )
    first, second = writers
    assert second.log_dir == first.log_dir
    assert second.purge_step is not None and second.purge_step <= 3
    assert second.hparams == []


def test_the_run_ends_with_its_status_and_closed_files(tmp_path, writers):
    """The status is the last thing written, then the writer lets go."""
    run(fitted(1), TensorBoard(log_dir=tmp_path))
    assert writers[0].texts[-1] == ("status", "finished") and writers[0].closed


def test_without_tensorboard_the_tracker_is_dropped_naming_the_extra(
    tmp_path, monkeypatch
):
    """The run goes on; the warning says what to install."""
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)
    with pytest.warns(UserWarning, match=r"chuchichaestli\[tensorboard\]"):
        run(fitted(1), TensorBoard(log_dir=tmp_path))


def test_a_run_reads_back_from_tensorboard_s_own_files(tmp_path):
    """Against the real thing: the event files hold the losses at their steps."""
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    run(fitted(), TensorBoard(log_dir=tmp_path), run_name="r")
    events = EventAccumulator(str(tmp_path / "r"))
    events.Reload()
    assert [e.step for e in events.Scalars("fit/loss")] == [1, 2, 3, 4, 5, 6]
    assert "fit/grad_norm" in events.Tags()["scalars"]
