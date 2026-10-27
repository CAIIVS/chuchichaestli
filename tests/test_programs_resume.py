# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests that a resumed run is indistinguishable from an uninterrupted one."""

import json
import os
import signal
import warnings

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from chuchichaestli.programs import (
    Alternating,
    Call,
    Checkpointer,
    CheckpointStore,
    DataManager,
    EarlyStop,
    Eval,
    EventType,
    GracefulStop,
    Jsonl,
    Phase,
    Predict,
    Program,
    DiscriminatorAdv,
    GeneratorAdv,
    Repeat,
    Runner,
    Signal,
    Train,
    WeightsExport,
)
from chuchichaestli.metrics import MSE
from chuchichaestli.models.unet import UNet
from chuchichaestli.programs.runner import C3liRunError
from chuchichaestli.training import OptimSpec, Term

SAMPLES = 8
BATCH = 2
STEPS_PER_EPOCH = SAMPLES // BATCH


def ramp(n: int = SAMPLES) -> TensorDataset:
    """Build a dataset of small images whose values count upwards.

    Args:
        n: Number of samples.
    """
    x = torch.arange(n * 64, dtype=torch.float32).reshape(n, 1, 8, 8) / (n * 64)
    return TensorDataset(x, x.flip(-1))


def model() -> nn.Module:
    """Build a model whose initial weights are the same every time.

    Convolutions and normalisation buffers are what a resume has to carry;
    a single dense layer would not show a divergence in either.
    """
    torch.manual_seed(1234)
    return UNet(
        dimensions=2,
        in_channels=1,
        out_channels=1,
        n_channels=8,
        down_block_types=("DownBlock",),
        up_block_types=("UpBlock",),
        block_out_channel_mults=(1,),
        res_groups=4,
    )


def trainer(
    epochs: int = 2, workers: int = 0, optim: OptimSpec | None = None, **kwargs
) -> Train:
    """Build the stage under test.

    Args:
        epochs: Passes over the data to make.
        workers: Dataloader worker processes.
        optim: Optimizer and schedule; an exponential decay by default.
        kwargs: Passed on to `Train`.
    """
    return Train(
        "fit",
        data=DataManager(ramp(), batch_size=BATCH, num_workers=workers),
        loss=nn.MSELoss(),
        epochs=epochs,
        optim=optim or OptimSpec.sgd(lr=0.1).with_exponential_schedule(0.9),
        **kwargs,
    )


def plain(epochs: int = 2, workers: int = 0, **kwargs) -> Program:
    """Build a program of one training stage.

    Args:
        epochs: Passes over the data to make.
        workers: Dataloader worker processes.
        kwargs: Passed on to `Train`.
    """
    return Program(
        provide={"model": model()}, stages=[trainer(epochs, workers, **kwargs)]
    )


def adversarial() -> Program:
    """Build a program whose stage steps two models in named groups."""
    torch.manual_seed(7)
    gen, disc = nn.Linear(3, 3, bias=False), nn.Linear(3, 1, bias=False)
    stage = Train(
        "gan",
        data=TensorDataset(torch.zeros(8, 3) + 0.5, torch.zeros(8, 1)),
        batch_size=2,
        epochs=2,
        objective=[
            Term("g", GeneratorAdv(), groups=("gen",)),
            Term("d", DiscriminatorAdv(), groups=("disc",)),
        ],
        update=Alternating(("disc", "gen")),
        optim={
            "disc": OptimSpec.sgd(lr=0.05, params="disc"),
            "gen": OptimSpec.sgd(lr=0.05, params="model"),
        },
    )
    return Program(provide={"model": gen, "disc": disc}, stages=[stage])


def bindings(program: Program, *names: str) -> dict:
    """Return a copy of every tensor in the named bindings.

    Args:
        program: Program holding the bindings.
        names: Which bindings to read.
    """
    return {
        f"{name}.{key}": tensor.detach().clone()
        for name in names
        for key, tensor in program.provide[name].state_dict().items()
    }


def repeated(times: int = 2) -> Program:
    """Build a program whose training stage sits inside a `Repeat`.

    Args:
        times: How many times the child runs.
    """
    return Program(
        provide={"model": model()},
        stages=[Repeat(times, trainer(epochs=1), name="cycle")],
    )


def side_trainer() -> Train:
    """Build a training stage for the `side` binding, to follow the first."""
    return Train(
        "side",
        model="side",
        data=DataManager(ramp(), batch_size=BATCH),
        loss=nn.MSELoss(),
        epochs=1,
        optim=OptimSpec.sgd(lr=0.1).with_exponential_schedule(0.9),
    )


def cycles(times: int = 1) -> Program:
    """Build a program repeating two training stages, one after the other.

    Args:
        times: How many times the pair runs.
    """
    return Program(
        provide={"model": model(), "side": model()},
        stages=[Repeat(times, Phase("cycle", [trainer(epochs=1), side_trainer()]))],
    )


def probed(seen: dict) -> Program:
    """Build a program whose last stage reads what earlier ones published.

    Args:
        seen: Receives the score and the averaged weights it read.
    """
    return Program(
        provide={"model": model(), "side": model()},
        stages=[
            trainer(epochs=1, ema=0.9),
            Eval(
                "probe",
                data=ramp(),
                batch_size=BATCH,
                weights="ema",
                metrics=[MSE()],
            ),
            side_trainer(),
            Call(
                "read",
                fn=lambda ctx: seen.update(
                    mse=ctx["probe/mse"], ema=ctx["model/ema"].state_dict()
                ),
            ),
        ],
    )


def weights(program: Program) -> dict[str, torch.Tensor]:
    """Return a copy of every trained tensor, buffers included.

    Args:
        program: Program holding the model binding.
    """
    return {
        name: tensor.detach().clone()
        for name, tensor in program.provide["model"].state_dict().items()
    }


def identical(one: dict, other: dict) -> bool:
    """Whether two sets of tensors match to the bit.

    Args:
        one: Tensors from one run.
        other: Tensors from the other.
    """
    return one.keys() == other.keys() and all(
        torch.equal(tensor, other[name]) for name, tensor in one.items()
    )


def work_trace(path) -> list[dict]:
    """Return every step the run reported taking, whole records.

    An event carries no timestamp, so the counters and the payload are
    compared along with the rest. Only what brackets the work is left out:
    an interrupted run is bracketed twice, re-entering its stage and
    reopening its pass, so the `run.*`, `stage.*` and `epoch.*` pairs
    legitimately occur once more than in a run that was never stopped.

    Args:
        path: JSONL file the `Jsonl` hook wrote.
    """
    return [
        event
        for event in map(json.loads, path.read_text().splitlines())
        if event["type"].startswith("step.")
    ]


class CancelAt:
    """Stand in for a signal arriving partway through a run."""

    def __init__(self, steps: int):
        """Constructor.

        Args:
            steps: Optimizer steps to allow before cancelling.
        """
        self.steps = steps
        self.seen = 0

    def on(self, event):
        """Cancel once enough steps have gone by.

        Args:
            event: What the runner just did.

        Raises:
            C3liRunError: Once the run has taken `steps` steps.
        """
        if event.type is EventType.STEP_ENDED:
            self.seen += 1
            if self.seen >= self.steps:
                raise C3liRunError("cancelled")
        return Signal.GO


class SignalAt:
    """Send this process a SIGTERM partway into a step."""

    def __init__(self, steps: int):
        """Constructor.

        Args:
            steps: Optimizer steps to allow before signalling.
        """
        self.steps = steps
        self.seen = 0

    def on(self, event):
        """Signal once enough steps have gone by.

        Args:
            event: What the runner just did.
        """
        if event.type is EventType.STEP_ENDED:
            self.seen += 1
            if self.seen == self.steps:
                os.kill(os.getpid(), signal.SIGTERM)
        return Signal.GO


def uninterrupted(program: Program, trace=None) -> Program:
    """Run a program from start to finish.

    Args:
        program: What to run.
        trace: Directory to record the trace into, if anywhere. This run
            keeps no checkpoints, so the hook is told where to write.
    """
    hooks = [Jsonl("trace.jsonl", store=trace)] if trace else []
    Runner(program, seed=42, hooks=hooks).run()
    return program


def interrupted(
    program: Program,
    store,
    at: int,
    trace=None,
    format="safetensors",
    graceful=False,
) -> Program:
    """Run a program until it is cancelled partway through.

    Args:
        program: What to run.
        store: Where checkpoints and the trace are written.
        at: Optimizer step to cancel on.
        trace: Name to record the trace under, if anywhere.
        format: File format the checkpoints are written in.
        graceful: Stop as a signal would, once the step is whole, rather than
            failing partway into it.
    """
    hooks = [Checkpointer(every=1, unit="step", format=format)]
    if trace:
        hooks.append(Jsonl(trace))
    hooks += [GracefulStop(), SignalAt(at)] if graceful else [CancelAt(at)]
    with pytest.raises(C3liRunError):
        Runner(program, seed=42, store=store, hooks=hooks).run()
    return program


def resumed(program: Program, store, trace=None) -> Program:
    """Carry a program on from its last checkpoint.

    Args:
        program: A fresh copy of what was interrupted.
        store: Where checkpoints were written.
        trace: Name to record the trace under, if anywhere.
    """
    hooks = [Jsonl(trace)] if trace else []
    Runner(program, seed=42, store=store, resume="last", hooks=hooks).run()
    return program


def optimizer_state(program: Program) -> dict:
    """Return the training stage's optimizer and scheduler state.

    Args:
        program: Program whose first stage is the trainer.
    """
    stage = program.stages[0]
    stage = stage.stages[0] if hasattr(stage, "stages") else stage
    return {
        k: v for k, v in stage.state_dict().items() if k.startswith(("optim", "sched"))
    }


@pytest.mark.parametrize("at", [3, 4, 5])
def test_a_resumed_run_ends_where_an_uninterrupted_one_does(tmp_path, at):
    """The headline guarantee: mid-pass, on a pass boundary, and after one."""
    store = tmp_path / "run"
    whole = uninterrupted(plain())
    interrupted(plain(), store, at=at)
    carried = resumed(plain(), store)
    assert identical(weights(whole), weights(carried))


def test_an_unseeded_resume_takes_the_seed_of_the_run_it_resumes(tmp_path):
    """Not torch's, which would draw the rest of the run from other streams."""
    store = tmp_path / "run"
    whole = uninterrupted(plain())
    interrupted(plain(), store, at=5)
    torch.manual_seed(0)
    carried = plain()
    runner = Runner(carried, store=store, resume="last", hooks=[])
    runner.run()
    assert runner.seed == 42
    assert identical(weights(whole), weights(carried))


def test_the_optimizer_and_schedule_come_back_too(tmp_path):
    """Matching weights would mean little if the state behind them differed."""
    store = tmp_path / "run"
    whole = uninterrupted(plain())
    interrupted(plain(), store, at=5)
    carried = resumed(plain(), store)
    assert optimizer_state(whole) == optimizer_state(carried)


@pytest.mark.parametrize("at", [2, 3, 5])
def test_a_warmup_resumes_on_either_side_of_its_handover(tmp_path, at):
    """Inside the warmup, on its last step, and in the schedule after it."""
    optim = (
        OptimSpec.sgd(lr=0.1)
        .with_exponential_schedule(0.9, interval="step")
        .with_warmup("linear", 3, start_factor=0.1)
    )
    store = tmp_path / "run"
    whole = uninterrupted(plain(optim=optim))
    interrupted(plain(optim=optim), store, at=at)
    carried = resumed(plain(optim=optim), store)
    assert identical(weights(whole), weights(carried))
    assert optimizer_state(whole) == optimizer_state(carried)


def test_early_stopping_picks_up_its_patience_where_it_left_off(tmp_path):
    """A fresh patience would let the resumed run train past the plateau."""

    def watching():
        # the step count never improves, so every step after the first waits
        return EarlyStop("total", mode="max", patience=1000)

    store = tmp_path / "run"
    whole = watching()
    Runner(plain(), seed=42, hooks=[whole]).run()
    with pytest.raises(C3liRunError):
        Runner(
            plain(),
            seed=42,
            store=store,
            hooks=[watching(), Checkpointer(every=1, unit="step"), CancelAt(5)],
        ).run()
    carried = watching()
    Runner(plain(), seed=42, store=store, resume="last", hooks=[carried]).run()
    assert whole.waited == 2 * STEPS_PER_EPOCH - 1
    assert (carried.best, carried.waited) == (whole.best, whole.waited)


def test_a_hook_finds_its_state_though_others_were_added(tmp_path):
    """State follows what a hook is, not where it sits in the list."""
    store = tmp_path / "run"
    whole = EarlyStop("total", mode="max", patience=1000)
    Runner(plain(), seed=42, hooks=[whole]).run()
    with pytest.raises(C3liRunError):
        Runner(
            plain(),
            seed=42,
            store=store,
            hooks=[
                EarlyStop("total", mode="max", patience=1000),
                Checkpointer(every=1, unit="step"),
                CancelAt(5),
            ],
        ).run()
    carried = EarlyStop("total", mode="max", patience=1000)
    Runner(
        plain(), seed=42, store=store, resume="last", hooks=[CancelAt(99), carried]
    ).run()
    assert (carried.best, carried.waited) == (whole.best, whole.waited)


def test_early_stopping_on_another_value_starts_afresh(tmp_path):
    """Another value's best would mean nothing, so it is reported instead."""
    store = tmp_path / "run"
    with pytest.raises(C3liRunError):
        Runner(
            plain(),
            seed=42,
            store=store,
            hooks=[
                EarlyStop("total", mode="max", patience=1000),
                Checkpointer(every=1, unit="step"),
                CancelAt(5),
            ],
        ).run()
    # a flag is never a measurement, so only restored state could move `waited`
    other = EarlyStop("trains", patience=1000)
    with pytest.warns(UserWarning, match="EarlyStop:total#0"):
        Runner(plain(), seed=42, store=store, resume="last", hooks=[other]).run()
    assert other.waited == 0


def test_a_sparser_checkpointer_keeps_its_rhythm_across_a_resume(tmp_path):
    """Counting from zero again would shift every later checkpoint."""

    def every_third(store):
        return Checkpointer(every=3, unit="step", store=store, at_end=False)

    def steps_written(store):
        read = CheckpointStore(store)
        return [read.load(path).progress.global_step for path in read.checkpoints()]

    whole = tmp_path / "whole"
    Runner(plain(), seed=42, hooks=[every_third(whole)]).run()
    store, sparse = tmp_path / "run", tmp_path / "sparse"
    with pytest.raises(C3liRunError):
        Runner(
            plain(),
            seed=42,
            store=store,
            hooks=[
                every_third(sparse),
                Checkpointer(every=1, unit="step", at_end=False),
                CancelAt(5),
            ],
        ).run()
    Runner(
        plain(),
        seed=42,
        store=store,
        resume="last",
        hooks=[every_third(sparse), Checkpointer(every=1, unit="step", at_end=False)],
    ).run()
    assert steps_written(sparse) == steps_written(whole) == [3, 6]


def test_leaving_a_hook_out_of_the_resumed_run_is_not_reported(tmp_path):
    """Resuming without the checkpointer is a choice, not a mistake."""
    store = tmp_path / "run"
    interrupted(plain(), store, at=5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resumed(plain(), store)
    assert not [w for w in caught if "No hook in this run" in str(w.message)]


def test_a_run_failing_partway_into_a_step_keeps_its_last_clean_checkpoint(
    tmp_path,
):
    """A final write there would replace it with a state no resume can trust."""
    store = tmp_path / "run"
    interrupted(plain(), store, at=3)
    read = CheckpointStore(store)
    written = [read.load(path) for path in read.checkpoints()]
    assert [(c.index, c.progress.global_step) for c in written] == [(1, 1), (2, 2)]


def test_a_graceful_stop_still_writes_its_final_checkpoint(tmp_path):
    """It waits for the step to finish, so the state it leaves is whole."""
    store = tmp_path / "run"
    with pytest.raises(C3liRunError, match="cancelled by SIGTERM"):
        Runner(
            plain(),
            seed=42,
            store=store,
            hooks=[Checkpointer(every=100, unit="step"), GracefulStop(), SignalAt(3)],
        ).run()
    read = CheckpointStore(store)
    written = [read.load(path) for path in read.checkpoints()]
    assert [(c.index, c.progress.global_step) for c in written] == [(3, 3)]


def test_a_torch_format_checkpoint_resumes_too(tmp_path):
    """The runner reads whichever format the checkpointer was told to write."""
    store = tmp_path / "run"
    whole = uninterrupted(plain())
    interrupted(plain(), store, at=5, format="torch")
    carried = resumed(plain(), store)
    assert identical(weights(whole), weights(carried))
    assert optimizer_state(whole) == optimizer_state(carried)


@pytest.mark.parametrize("workers", [0, 2])
def test_the_result_does_not_depend_on_the_worker_count(tmp_path, workers):
    """Workers change who reads the data, not what the run computes."""
    store = tmp_path / f"run{workers}"
    whole = uninterrupted(plain(workers=workers))
    interrupted(plain(workers=workers), store, at=5)
    carried = resumed(plain(workers=workers), store)
    assert identical(weights(whole), weights(carried))


def test_a_run_resumes_inside_a_nested_repeat(tmp_path):
    """The stage that was live is found however deep it sits."""
    store = tmp_path / "run"
    whole = uninterrupted(repeated())
    interrupted(repeated(), store, at=5)
    carried = resumed(repeated(), store)
    assert identical(weights(whole), weights(carried))


def test_the_resumed_trace_matches_the_uninterrupted_one(tmp_path):
    """What the run reports must not say it was interrupted either."""
    store = tmp_path / "run"
    uninterrupted(plain(), trace=tmp_path / "alone")
    interrupted(plain(), store, at=5, trace="first.jsonl", graceful=True)
    resumed(plain(), store, trace="second.jsonl")

    carried = work_trace(store / "first.jsonl") + work_trace(store / "second.jsonl")
    assert carried == work_trace(tmp_path / "alone" / "trace.jsonl")


def test_a_cancelled_run_keeps_the_stage_it_was_in(tmp_path):
    """Its final checkpoint is the one a resume has to work from."""
    store = tmp_path / "run"
    interrupted(plain(), store, at=4)
    last = sorted(p for p in store.iterdir() if p.is_dir())[-1]
    state = json.loads((last / "manifest.json").read_text())["state"]["program"]
    assert state["child"] is not None
    assert state["index"] == 0


def test_a_resumed_stage_is_entered_again(tmp_path):
    """It rebuilds the optimizer and objective that were never checkpointed."""
    store = tmp_path / "run"
    interrupted(plain(), store, at=4)
    program = plain()
    began = []

    class Watching:
        """Records the stages the run enters."""

        def on(self, event):
            """Note a stage beginning.

            Args:
                event: What the runner just did.
            """
            if event.type is EventType.STAGE_BEGAN:
                began.append(event.path)
            return Signal.GO

    Runner(program, seed=42, store=store, resume="last", hooks=[Watching()]).run()
    assert "program/0:fit" in began
    assert program.stages[0].update.optimizers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_resumed_run_matches_bitwise_on_a_gpu(tmp_path):
    """Kernels have to be pinned down too: the same inputs are not enough."""
    store = tmp_path / "run"

    def run(program, **kwargs):
        """Run a program on the GPU with every source of drift shut off.

        Args:
            program: What to run.
            kwargs: Passed on to the runner.
        """
        return Runner(
            program, seed=42, device="cuda", backends="strict", **kwargs
        ).run()

    whole = plain()
    run(whole, hooks=[])
    with pytest.raises(C3liRunError):
        run(
            plain(),
            store=store,
            hooks=[Checkpointer(every=1, unit="step"), CancelAt(5)],
        )
    carried = plain()
    run(carried, store=store, resume="last", hooks=[])
    assert identical(weights(whole), weights(carried))


def test_a_reopened_pass_reports_the_epoch_it_resumed_into(tmp_path):
    """The trace is the oracle here, so its counters have to be right."""
    store = tmp_path / "run"
    interrupted(plain(), store, at=6, graceful=True)
    resumed(plain(), store, trace="second.jsonl")
    trace = store / "second.jsonl"
    opened = [
        event
        for event in map(json.loads, trace.read_text().splitlines())
        if event["type"] == EventType.EPOCH_BEGAN.value
    ]
    assert opened[0]["payload"]["epoch"] == 1
    assert opened[0]["progress"]["epoch"] == 1
    assert opened[0]["progress"]["step"] == 2


def test_a_resumed_run_matches_when_a_step_is_micro_batched(tmp_path):
    """A step spanning several batches is the unit a checkpoint falls between."""
    store = tmp_path / "run"
    whole = uninterrupted(plain(accumulate=2))
    interrupted(plain(accumulate=2), store, at=2)
    carried = resumed(plain(accumulate=2), store)
    assert identical(weights(whole), weights(carried))


def test_a_moving_average_survives_the_interruption(tmp_path):
    """The average is stage state, so it has to come back with the rest."""
    store = tmp_path / "run"
    whole = uninterrupted(plain(ema=0.9))
    interrupted(plain(ema=0.9), store, at=5)
    carried = resumed(plain(ema=0.9), store)
    assert identical(weights(whole), weights(carried))
    assert identical(
        dict(whole.stages[0]._ema["model"].state_dict()),
        dict(carried.stages[0]._ema["model"].state_dict()),
    )


def test_every_update_group_comes_back(tmp_path):
    """Two optimizers mean two sets of state, and neither may be dropped."""
    store = tmp_path / "run"
    whole = uninterrupted(adversarial())
    interrupted(adversarial(), store, at=5)
    carried = resumed(adversarial(), store)
    assert identical(
        bindings(whole, "model", "disc"), bindings(carried, "model", "disc")
    )


def test_a_run_resumes_from_where_the_hook_was_told_to_write(tmp_path):
    """The store is a directory, so the two ways of naming it agree."""
    elsewhere = tmp_path / "elsewhere"
    whole = uninterrupted(plain())
    with pytest.raises(C3liRunError):
        Runner(
            plain(),
            seed=42,
            hooks=[Checkpointer(every=1, unit="step", store=elsewhere), CancelAt(5)],
        ).run()
    carried = plain()
    Runner(carried, seed=42, store=elsewhere, resume="last", hooks=[]).run()
    assert identical(weights(whole), weights(carried))


def test_naming_the_same_store_twice_writes_it_once(tmp_path):
    """A hook pointed at the run's own store is not a second writer."""
    store = tmp_path / "run"
    whole = uninterrupted(plain())
    with pytest.raises(C3liRunError):
        Runner(
            plain(),
            seed=42,
            store=store,
            hooks=[Checkpointer(every=1, unit="step", store=store), CancelAt(5)],
        ).run()
    written = sorted(p.name for p in store.glob("ckpt_*"))
    assert written == sorted(set(written))
    carried = plain()
    Runner(carried, seed=42, store=store, resume="last", hooks=[]).run()
    assert identical(weights(whole), weights(carried))


def test_a_run_resumes_before_its_second_training_stage(tmp_path):
    """A stage not yet entered has recorded nothing, which loads as nothing."""
    store = tmp_path / "run"
    whole = uninterrupted(cycles())
    interrupted(cycles(), store, at=2)
    carried = resumed(cycles(), store)
    assert identical(
        bindings(whole, "model", "side"), bindings(carried, "model", "side")
    )


def test_a_run_resumes_twice_before_a_stage_runs_again(tmp_path):
    """A stage not yet entered since the first resume records what it carries."""
    store = tmp_path / "run"
    whole = uninterrupted(cycles(2))
    interrupted(cycles(2), store, at=5)
    with pytest.raises(C3liRunError):
        Runner(
            cycles(2),
            seed=42,
            store=store,
            resume="last",
            hooks=[Checkpointer(every=1, unit="step"), CancelAt(1)],
        ).run()
    carried = resumed(cycles(2), store)
    assert identical(
        bindings(whole, "model", "side"), bindings(carried, "model", "side")
    )


@pytest.mark.parametrize(("at", "stage"), [(5, 1), (9, 2)])
def test_what_a_finished_stage_published_survives_a_resume(tmp_path, at, stage):
    """Mid-evaluation and in a later stage, both read what came before."""
    store = tmp_path / "run"
    whole, carried = {}, {}
    uninterrupted(probed(whole))
    interrupted(probed({}), store, at=at, graceful=True)
    last = sorted(p for p in store.iterdir() if p.is_dir())[-1]
    state = json.loads((last / "manifest.json").read_text())["state"]["program"]
    assert state["index"] == stage
    resumed(probed(carried), store)
    assert carried["mse"] == whole["mse"]
    assert identical(dict(whole["ema"]), dict(carried["ema"]))


def predicted(seen: dict, archive=None) -> Program:
    """Build a program whose last stage reads what a finished Predict published.

    Args:
        seen: Receives the predictions and truths it read.
        archive: File the predictions are written to, or `None` to keep them.
    """

    def read(ctx):
        pairs = ctx["sample/pairs"]
        rows = [pairs[index] for index in range(len(pairs))]
        seen["predicted"] = torch.stack([row[0] for row in rows])
        seen["truths"] = torch.stack([row[1] for row in rows])

    return Program(
        provide={"model": model(), "side": model()},
        stages=[
            trainer(epochs=1),
            Predict(
                "sample",
                data=ramp(),
                batch_size=BATCH,
                inputs=(0,),
                targets=1,
                archive=archive,
            ),
            side_trainer(),
            Call("read", fn=read),
        ],
    )


@pytest.mark.parametrize("archived", [False, True], ids=["in_memory", "archive"])
def test_what_a_finished_predict_published_survives_a_resume(tmp_path, archived):
    """A later stage still reads the predictions and their truths."""

    def archive(name):
        return tmp_path / name / "sample.safetensors" if archived else None

    store, whole, carried = tmp_path / "run", {}, {}
    Runner(predicted(whole, archive("whole")), seed=42, hooks=[]).run()
    interrupted(predicted({}, archive("carried")), store, at=9, graceful=True)
    last = sorted(p for p in store.iterdir() if p.is_dir())[-1]
    state = json.loads((last / "manifest.json").read_text())["state"]["program"]
    assert state["index"] == 2
    resumed(predicted(carried, archive("carried")), store)
    assert torch.equal(carried["predicted"], whole["predicted"])
    assert torch.equal(carried["truths"], whole["truths"])


def counted(seen: list, export=None) -> Program:
    """Build a program that trains, then notes the run's counts.

    Args:
        seen: List the counts are appended to.
        export: Directory to export numbered weights into, if any.
    """
    stages = [
        trainer(epochs=3),
        Call("count", fn=lambda ctx: seen.append(ctx.counts())),
    ]
    if export is not None:
        stages.append(
            WeightsExport(
                "save", path=export / "w_{epoch}_{step}_{advance}.safetensors"
            )
        )
    return Program(provide={"model": model()}, stages=stages)


@pytest.mark.parametrize("at", [3, 5, 8])
def test_a_resumed_run_counts_on_from_where_it_stopped(tmp_path, at):
    """The run's step, epoch and advance counts must not restart at zero."""
    whole: list = []
    uninterrupted(counted(whole))
    carried: list = []
    store = tmp_path / "run"
    interrupted(counted([]), store, at=at)
    resumed(counted(carried), store)
    assert carried == whole
    assert whole[0]["step"] == 3 * STEPS_PER_EPOCH and whole[0]["epoch"] == 3


def test_a_resumed_export_numbers_its_file_as_an_uninterrupted_one(tmp_path):
    """Restarted counters would overwrite the files of earlier passes."""
    first, second = tmp_path / "whole", tmp_path / "carried"
    uninterrupted(counted([], first))
    store = tmp_path / "run"
    interrupted(counted([], second), store, at=5)
    resumed(counted([], second), store)
    assert sorted(p.name for p in second.iterdir()) == sorted(
        p.name for p in first.iterdir()
    )
