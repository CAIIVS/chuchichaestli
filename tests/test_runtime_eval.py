# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the stages that read a model without changing it."""

import h5py
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from chuchichaestli.metrics import MSE, PSNR
from chuchichaestli.runtime import (
    Call,
    Context,
    Eval,
    Predict,
    Program,
    Runtime,
    Train,
)


def linear(fill: float = 0.5) -> nn.Module:
    """Build a one-layer model with a known weight.

    Args:
        fill: Value every weight starts at.
    """
    model = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(fill)
    return model


def ramp(n: int = 8) -> TensorDataset:
    """Build a small paired dataset.

    Args:
        n: Number of samples.
    """
    x = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) / (n * 3)
    return TensorDataset(x, torch.zeros(n, 1))


def test_eval_leaves_the_model_untouched():
    """Metrics accumulate without a gradient or an update in sight."""
    model = linear()
    before = model.weight.detach().clone()
    Runtime(
        Eval("probe", model=model, data=ramp(), batch_size=4, metrics=[MSE()]),
        hooks=(),
        device="cpu",
    ).run()
    assert torch.equal(model.weight, before)


def test_eval_publishes_each_metric_for_later_siblings():
    """`ctx["probe/mse"]` is what a later `When` reads."""
    seen: dict[str, float] = {}
    program = Program(
        provide={"model": linear()},
        stages=[
            Eval("probe", data=ramp(), batch_size=4, metrics=[MSE(), PSNR()]),
            Call(
                "read",
                fn=lambda ctx: seen.update(
                    mse=ctx["probe/mse"], psnr=ctx["probe/psnr"]
                ),
            ),
        ],
    )
    Runtime(program, hooks=(), device="cpu").run()
    assert sorted(seen) == ["mse", "psnr"]
    assert isinstance(seen["mse"], float)


def test_metrics_may_be_named():
    """A mapping keys the published names instead of the class name."""
    stage = Eval("probe", model=linear(), data=ramp(), metrics={"recon": MSE()})
    assert sorted(stage.metrics) == ["recon"]


def test_eval_needs_something_to_measure():
    """A stage with no metric would publish nothing."""
    with pytest.raises(ValueError, match="no metric"):
        Eval("probe", model=linear(), data=ramp())


def test_metric_state_rides_in_the_stage_state():
    """A resumed eval keeps what it had already accumulated."""
    stage = Eval("probe", model=linear(), data=ramp(), batch_size=4, metrics=[MSE()])
    ctx = Context("t")
    stage.enter(ctx)
    stage.execute(ctx)
    partial = stage.state_dict()
    assert "mse" in partial["metrics"]

    resumed = Eval("probe", model=linear(), data=ramp(), batch_size=4, metrics=[MSE()])
    resumed.enter(Context("t"))
    resumed.load_state_dict(partial)
    assert torch.equal(resumed.metrics["mse"].aggregate, stage.metrics["mse"].aggregate)


def test_weights_ema_reads_the_average_a_train_published():
    """The two evals disagree, which is what makes the swap observable."""
    torch.manual_seed(0)
    x = torch.randn(16, 3)
    data = TensorDataset(x, x @ torch.tensor([[1.0], [2.0], [-1.0]]))
    program = Program(
        provide={"model": nn.Linear(3, 1, bias=False)},
        stages=[
            Train(
                "fit",
                data=data,
                batch_size=4,
                loss=nn.MSELoss(),
                epochs=5,
                lr=0.1,
                ema=0.5,
            ),
            Eval("live", data=data, batch_size=4, metrics=[MSE()]),
            Eval("avg", data=data, batch_size=4, metrics=[MSE()], weights="ema"),
        ],
    )
    Runtime(program, hooks=(), device="cpu").run()
    live = program.stages[1].metrics["mse"].compute()
    averaged = program.stages[2].metrics["mse"].compute()
    assert live != averaged


def test_an_unknown_weights_choice_is_refused():
    """Only the model or its average can be read."""
    with pytest.raises(ValueError, match="weights"):
        Eval("probe", model=linear(), data=ramp(), metrics=[MSE()], weights="swa")


def test_predict_collects_every_output():
    """One prediction per sample, in order."""
    stage = Predict("out", model=linear(), data=ramp(8), batch_size=4)
    Runtime(stage, hooks=(), device="cpu").run()
    assert tuple(stage.predictions.shape) == (8, 1)


def test_predict_streams_to_an_appendable_archive(tmp_path):
    """Batches reach the file as they are produced, not at the end.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    archive = tmp_path / "out.h5"
    stage = Predict("out", model=linear(), data=ramp(8), batch_size=4, archive=archive)
    Runtime(stage, hooks=(), device="cpu").run()
    with h5py.File(archive) as handle:
        assert handle["data"].shape == (8, 1)
    assert stage.predictions is None


def test_predict_still_writes_a_format_it_cannot_append_to(tmp_path):
    """The fallback buffers, and says so.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    archive = tmp_path / "out.npy"
    stage = Predict("out", model=linear(), data=ramp(8), batch_size=4, archive=archive)
    with pytest.warns(UserWarning, match="cannot be appended to"):
        Runtime(stage, hooks=(), device="cpu").run()
    assert np.load(archive).shape == (8, 1)


def test_predict_publishes_the_archive_it_wrote(tmp_path):
    """A later sibling learns where the predictions went.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    seen: dict[str, object] = {}
    archive = tmp_path / "out.h5"
    program = Program(
        provide={"model": linear()},
        stages=[
            Predict("out", data=ramp(8), batch_size=4, archive=archive),
            Call("read", fn=lambda ctx: seen.update(p=ctx["out/archive"])),
        ],
    )
    Runtime(program, hooks=(), device="cpu").run()
    assert seen["p"] == archive


def test_predict_publishes_what_it_produced():
    """A later sibling can read the predictions without a file."""
    seen: dict[str, torch.Tensor] = {}
    program = Program(
        provide={"model": linear()},
        stages=[
            Predict("out", data=ramp(8), batch_size=4),
            Call("read", fn=lambda ctx: seen.update(p=ctx["out/predictions"])),
        ],
    )
    Runtime(program, hooks=(), device="cpu").run()
    assert tuple(seen["p"].shape) == (8, 1)
