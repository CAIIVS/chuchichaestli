# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests that a resumed run is indistinguishable from an uninterrupted one."""

import json

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from chuchichaestli.runtime import (
    Alternating,
    Checkpointer,
    DataManager,
    EventType,
    Jsonl,
    Program,
    DiscriminatorAdv,
    GeneratorAdv,
    Repeat,
    Runtime,
    Signal,
    Train,
)
from chuchichaestli.models.unet import UNet
from chuchichaestli.runtime.runtime import C3liRuntimeError
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


def trainer(epochs: int = 2, workers: int = 0, **kwargs) -> Train:
    """Build the stage under test.

    Args:
        epochs: Passes over the data to make.
        workers: Dataloader worker processes.
        kwargs: Passed on to `Train`.
    """
    return Train(
        "fit",
        data=DataManager(ramp(), batch_size=BATCH, num_workers=workers),
        loss=nn.MSELoss(),
        epochs=epochs,
        optim=OptimSpec.sgd(lr=0.1).with_exponential_schedule(0.9),
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
            event: What the runtime just did.

        Raises:
            C3liRuntimeError: Once the run has taken `steps` steps.
        """
        if event.type is EventType.STEP_ENDED:
            self.seen += 1
            if self.seen >= self.steps:
                raise C3liRuntimeError("cancelled")
        return Signal.GO


def uninterrupted(program: Program, trace=None) -> Program:
    """Run a program from start to finish.

    Args:
        program: What to run.
        trace: Where to write the event trace, if anywhere.
    """
    hooks = [Jsonl(trace)] if trace else []
    Runtime(program, seed=42, hooks=hooks).run()
    return program


def interrupted(program: Program, store, at: int, trace=None) -> Program:
    """Run a program until it is cancelled partway through.

    Args:
        program: What to run.
        store: Where checkpoints are written.
        at: Optimizer step to cancel on.
        trace: Where to write the event trace, if anywhere.
    """
    hooks = [Checkpointer(every=1, unit="step")]
    if trace:
        hooks.append(Jsonl(trace))
    hooks.append(CancelAt(at))
    with pytest.raises(C3liRuntimeError):
        Runtime(program, seed=42, store=store, hooks=hooks).run()
    return program


def resumed(program: Program, store, trace=None) -> Program:
    """Carry a program on from its last checkpoint.

    Args:
        program: A fresh copy of what was interrupted.
        store: Where checkpoints were written.
        trace: Where to write the event trace, if anywhere.
    """
    hooks = [Jsonl(trace)] if trace else []
    Runtime(program, seed=42, store=store, resume="last", hooks=hooks).run()
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


def test_the_optimizer_and_schedule_come_back_too(tmp_path):
    """Matching weights would mean little if the state behind them differed."""
    store = tmp_path / "run"
    whole = uninterrupted(plain())
    interrupted(plain(), store, at=5)
    carried = resumed(plain(), store)
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
    uninterrupted(plain(), trace=tmp_path / "whole.jsonl")
    interrupted(plain(), store, at=5, trace=tmp_path / "first.jsonl")
    resumed(plain(), store, trace=tmp_path / "second.jsonl")

    carried = work_trace(tmp_path / "first.jsonl") + work_trace(
        tmp_path / "second.jsonl"
    )
    assert carried == work_trace(tmp_path / "whole.jsonl")


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
                event: What the runtime just did.
            """
            if event.type is EventType.STAGE_BEGAN:
                began.append(event.path)
            return Signal.GO

    Runtime(program, seed=42, store=store, resume="last", hooks=[Watching()]).run()
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
            kwargs: Passed on to the runtime.
        """
        return Runtime(
            program, seed=42, device="cuda", backends="strict", **kwargs
        ).run()

    whole = plain()
    run(whole, hooks=[])
    with pytest.raises(C3liRuntimeError):
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
    interrupted(plain(), store, at=6)
    trace = tmp_path / "second.jsonl"
    resumed(plain(), store, trace=trace)
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
