# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the stages that read a model without changing it."""

import h5py
import warnings

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from chuchichaestli.metrics import MSE, PSNR
from chuchichaestli.data.archive import ARCHIVES, read_archive
from chuchichaestli.runtime import (
    Phase,
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
        Eval("probe", model=linear(), data=ramp(), metrics=[MSE()], weights="ewa")


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
    archive = tmp_path / "out.npz"
    stage = Predict("out", model=linear(), data=ramp(8), batch_size=4, archive=archive)
    with pytest.warns(UserWarning, match="cannot be appended to"):
        Runtime(stage, hooks=(), device="cpu").run()
    assert np.load(archive)["data"].shape == (8, 1)


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


@pytest.mark.parametrize("suffix", sorted(ARCHIVES))
def test_predict_streams_every_format_that_can_append(suffix, tmp_path):
    """Nothing is held back, so a large prediction run stays bounded.

    Args:
        suffix: Extension under test.
        tmp_path: Directory pytest gives the test.
    """
    archive = tmp_path / f"out{suffix}"
    stage = Predict("out", model=linear(), data=ramp(8), batch_size=4, archive=archive)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Runtime(stage, hooks=(), device="cpu").run()
    assert stage._writer is None
    assert len(torch.cat(list(read_archive(archive, "data")))) == 8


class TwoArgs(nn.Module):
    """A model called with more than the samples."""

    def forward(self, first, second):
        """Return the two inputs combined, so both are seen to arrive.

        Args:
            first: The samples.
            second: Whatever else the model is given.
        """
        return first + second.reshape(-1, 1)


def test_eval_calls_a_model_with_every_input_it_names(tmp_path):
    """A model taking a timestep or a condition needs no wrapper.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    samples = torch.ones(4, 1)
    extra = torch.full((4,), 2.0)
    stage = Eval(
        "probe",
        model=TwoArgs(),
        data=TensorDataset(samples, extra, samples * 3),
        batch_size=4,
        inputs=("x", "t"),
        metrics=[MSE()],
    )
    Runtime(Program(stages=[stage]), hooks=(), device="cpu").run()
    assert float(stage.metrics["mse"].compute()) == pytest.approx(0.0)


def test_predict_calls_a_model_with_every_input_it_names(tmp_path):
    """The same reading serves both inference stages.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    samples = torch.ones(4, 1)
    extra = torch.full((4,), 2.0)
    stage = Predict(
        "out",
        model=TwoArgs(),
        data=TensorDataset(samples, extra),
        batch_size=4,
        inputs=("x", "t"),
    )
    Runtime(Program(stages=[stage]), hooks=(), device="cpu").run()
    assert torch.equal(stage.predictions, torch.full((4, 1), 3.0))


class Denoiser(nn.Module):
    """Stands in for a noise-prediction model."""

    def forward(self, noised, steps):
        """Return a prediction shaped like the target.

        Args:
            noised: The condition and the noised sample together.
            steps: Which step of the schedule.
        """
        return noised[:, :1] * 0.0


def test_sample_draws_from_a_process_and_pairs_the_truth():
    """A `Predict` reads the model once; a process walks a whole schedule."""
    from chuchichaestli.diffusion.processes import DDPM
    from chuchichaestli.runtime import FromProcess, Predict

    torch.manual_seed(0)
    coarse, fine = torch.rand(4, 1, 8, 8), torch.rand(4, 1, 8, 8)
    stage = Predict(
        "draw",
        model=Denoiser(),
        draw=FromProcess(DDPM(num_timesteps=3, device="cpu")),
        data=TensorDataset(coarse, fine),
        batch_size=4,
        targets="y",
    )
    Runtime(Program(stages=[stage]), hooks=(), device="cpu").run()
    assert tuple(stage.predictions.shape) == (4, 1, 8, 8)


def test_sample_draws_from_a_latent_with_no_process_at_all():
    """A generator network is handed noise, not a schedule to walk."""
    from chuchichaestli.runtime import FromLatent, Predict

    generator = nn.Sequential(nn.Linear(3, 5))
    stage = Predict(
        "draw",
        model=generator,
        draw=FromLatent((3,), draws=6),
        data=TensorDataset(torch.zeros(1, 1)),
        batch_size=1,
        inputs=None,
    )
    Runtime(Program(stages=[stage]), hooks=(), device="cpu").run()
    assert tuple(stage.predictions.shape) == (6, 5)


def test_the_same_position_draws_the_same_latents():
    """The stage's own stream is what makes a drawn sample reproducible."""
    from chuchichaestli.runtime import FromLatent, Predict

    def drawn():
        """Return what one run of the stage produced."""
        torch.manual_seed(7)
        stage = Predict(
            "draw",
            model=nn.Identity(),
            draw=FromLatent((4,), draws=3),
            data=TensorDataset(torch.zeros(1, 1)),
            batch_size=1,
            inputs=None,
        )
        Runtime(Program(stages=[stage]), seed=11, hooks=(), device="cpu").run()
        return stage.predictions

    assert torch.equal(drawn(), drawn())


def test_eval_scores_what_a_batch_already_holds():
    """Data an earlier stage produced needs no model run over it again."""
    drawn = torch.zeros(4, 1)
    truth = torch.ones(4, 1)
    stage = Eval(
        "score",
        model=None,
        data=TensorDataset(drawn, truth),
        batch_size=4,
        metrics=[MSE()],
    )
    Runtime(Program(stages=[stage]), hooks=(), device="cpu").run()
    assert float(stage.metrics["mse"].compute()) == pytest.approx(1.0)


def test_a_stage_declares_what_it_leaves_behind():
    """A plan naming a binding nothing publishes is refused before any compute."""
    from chuchichaestli.runtime import FromLatent, Predict
    from chuchichaestli.runtime.runtime import C3liProgramError

    def drawing(targets):
        """Build a sampling stage, with or without a truth to pair.

        Args:
            targets: Key the ground truth is read from, or `None`.
        """
        return Predict(
            "sample",
            model=nn.Identity(),
            draw=FromLatent((2,), draws=4),
            data=TensorDataset(torch.zeros(1, 1), torch.zeros(1, 2)),
            batch_size=1,
            inputs=None,
            targets=targets,
        )

    assert drawing("y").provides == ("sample/predictions", "sample/pairs")
    assert drawing(None).provides == ("sample/predictions",)

    scoring = Eval(
        "score",
        model=None,
        data="sample/pairs",
        batch_size=4,
        metrics=[MSE()],
        requires=("sample/pairs",),
    )
    Runtime(Program(stages=[drawing("y"), scoring]), hooks=()).check()
    with pytest.raises(C3liProgramError, match="sample/pairs"):
        Runtime(Program(stages=[drawing(None), scoring]), hooks=()).check()


def test_an_unconditioned_draw_pairs_with_a_reference():
    """Nothing conditions it, so there is no truth to read per sample."""
    from chuchichaestli.runtime import FromLatent, Predict

    real = torch.rand(6, 2)
    drawing = Predict(
        "sample",
        model=nn.Linear(3, 2),
        draw=FromLatent((3,), draws=6),
        data=TensorDataset(torch.zeros(1, 1)),
        batch_size=1,
        inputs=None,
        reference=real,
    )
    scoring = Eval(
        "score", model=None, data="sample/pairs", batch_size=6, metrics=[MSE()]
    )
    Runtime(Program(stages=[Phase("run", [drawing, scoring])]), hooks=()).run()
    assert drawing.provides == ("sample/predictions", "sample/pairs")
    assert float(scoring.metrics["mse"].compute()) > 0


def test_a_draw_pairs_with_one_thing_or_the_other():
    """Reading a truth per batch and naming a reference set would conflict."""
    from chuchichaestli.runtime import FromLatent, Predict

    with pytest.raises(ValueError, match="not both"):
        Predict(
            "sample",
            draw=FromLatent((3,)),
            data=TensorDataset(torch.zeros(1, 1)),
            targets="y",
            reference=torch.zeros(1, 2),
        )


def test_a_reference_that_does_not_line_up_says_so():
    """Silently scoring six draws against four would be worse."""
    from chuchichaestli.runtime import FromLatent, Predict

    drawing = Predict(
        "sample",
        model=nn.Linear(3, 2),
        draw=FromLatent((3,), draws=6),
        data=TensorDataset(torch.zeros(1, 1)),
        batch_size=1,
        inputs=None,
        reference=torch.rand(4, 2),
    )
    with pytest.raises(ValueError, match="generated 6 but has 4"):
        Runtime(Program(stages=[drawing]), hooks=()).run()


class Conditioned(nn.Module):
    """Returns the condition it was handed, so pairing is observable."""

    def forward(self, latent, condition):
        """Return the condition unchanged.

        Args:
            latent: Noise the draw started from.
            condition: What the draw was conditioned on.
        """
        return condition


def test_the_truth_is_the_column_it_was_named_not_the_first():
    """A sequence batch maps names by position, so the pair must line up."""
    from chuchichaestli.runtime import FromLatent, Predict

    condition, truth = torch.zeros(4, 2), torch.ones(4, 2)
    stage = Predict(
        "draw",
        model=Conditioned(),
        draw=FromLatent((2,)),
        data=TensorDataset(condition, truth),
        batch_size=4,
        targets="y",
    )
    program = Program(stages=[Phase("run", [stage])])
    Runtime(program, hooks=()).run()
    assert torch.equal(stage.predictions, condition), "the draw saw the condition"
    assert torch.equal(
        torch.stack([held.cpu() for held in stage._truths]).reshape(4, 2), truth
    ), "the truth must be the named column, not the first"


def test_what_an_eval_measured_rides_in_its_closing_event():
    """Published bindings are for later stages; the event is for the log."""
    import io

    from chuchichaestli.runtime import Console

    stream = io.StringIO()
    stage = Eval(
        "score",
        model=None,
        data=TensorDataset(torch.zeros(4, 1), torch.ones(4, 1)),
        batch_size=4,
        metrics=[MSE()],
    )
    Runtime(
        Program(stages=[stage]),
        hooks=[Console(every=1, stream=stream, color=False)],
        device="cpu",
    ).run()
    assert stage.summary() == {"mse": pytest.approx(1.0)}
    assert "mse=1" in stream.getvalue()


def test_a_metric_is_given_the_target_before_the_prediction():
    """`EvalMetric.update(data, prediction)`, or an asymmetric metric lies."""
    seen: list[tuple[float, float]] = []

    class Order(nn.Module):
        """Record the two tensors a metric is handed, in order."""

        name = "order"

        def update(self, data, prediction, update_range=True):
            """Note which tensor arrived first.

            Args:
                data: Observed data aka target.
                prediction: Predicted data aka inferred target.
                update_range: Unused, part of the metric interface.
            """
            seen.append((float(data.flatten()[0]), float(prediction.flatten()[0])))

        def compute(self):
            """Return a value, as a metric must."""
            return torch.tensor(0.0)

        def reset(self):
            """Discard what was recorded."""
            seen.clear()

    pairs = TensorDataset(torch.full((4, 1), 1.0), torch.full((4, 1), 2.0))
    Runtime(
        Program(
            [Eval("probe", model=None, data=pairs, batch_size=4, metrics=[Order()])]
        ),
        hooks=(),
        device="cpu",
    ).run()
    assert seen == [(2.0, 1.0)]


def test_an_archive_creates_the_directories_it_needs(tmp_path):
    """An appendable format opens its file at once, before any batch arrives."""
    program = Program(
        provide={"model": linear()},
        stages=[
            Predict(
                "p",
                model="model",
                data=ramp(),
                batch_size=4,
                archive=tmp_path / "nested" / "under" / "out.h5",
            )
        ],
    )
    Runtime(program, hooks=(), device="cpu").run()
    assert (tmp_path / "nested" / "under" / "out.h5").is_file()
