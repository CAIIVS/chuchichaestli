# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for running across several processes without deadlocking one."""

import os

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import TensorDataset

from chuchichaestli.programs import (
    ArchiveExport,
    Call,
    Checkpointer,
    Eval,
    Predict,
    DataManager,
    Ddp,
    EventType,
    Jsonl,
    Program,
    Repeat,
    Runner,
    Signal,
    Train,
)
from chuchichaestli.programs.events import C3liRunError
from chuchichaestli.data.archive import read_archive
from chuchichaestli.metrics import MSE
from chuchichaestli.training import OptimSpec

WORLD = 2
JOIN_TIMEOUT = 120


def ramp(n: int = 8) -> TensorDataset:
    """Build a dataset whose inputs count upwards.

    Args:
        n: Number of samples.
    """
    x = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) / 10
    return TensorDataset(x, x[:, :1])


def model() -> nn.Module:
    """Build a model whose initial weights are the same every time."""
    torch.manual_seed(1234)
    return nn.Linear(3, 1, bias=False)


def under_ddp(worker, *args):
    """Run a worker on every rank and return what each one reported.

    The join is bounded so a rank that never arrives fails the test rather
    than hanging the suite, which is the failure this file exists to catch.

    Args:
        worker: Callable taking `(rank, world, results, *args)`.
        args: Extra arguments passed on to the worker.

    Raises:
        TimeoutError: If a rank has not finished in time.
    """
    manager = mp.Manager()
    results = manager.dict()
    context = mp.spawn(worker, args=(WORLD, results, *args), nprocs=WORLD, join=False)
    waited = 0
    while not context.join(timeout=5):
        waited += 5
        if waited >= JOIN_TIMEOUT:
            for process in context.processes:
                process.terminate()
            raise TimeoutError("a rank never finished; the run deadlocked")
    return dict(results)


def join_group(rank: int, world: int, port: int, device: str = "cpu") -> Ddp:
    """Put this process into the run's process group.

    Args:
        rank: Index of this process.
        world: Number of processes taking part.
        port: Rendezvous port the ranks agree on.
        device: Device this rank computes on.
    """
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    return Ddp(backend="gloo", device=device)


def free_port() -> int:
    """Return a port nothing is listening on."""
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def stage(topology, **kwargs) -> Train:
    """Build a training stage reading its data through the topology.

    Args:
        topology: Process layout the run executes across.
        kwargs: Passed on to `Train`.
    """
    return Train(
        "fit",
        data=DataManager(ramp(), batch_size=2),
        loss=nn.MSELoss(),
        epochs=2,
        optim=OptimSpec.sgd(lr=0.1),
        **kwargs,
    )


def _trains(rank, world, results, port):
    """Train on every rank and report the weights each one ends with.

    Args:
        rank: Index of this process.
        world: Number of processes taking part.
        results: Shared mapping the ranks report into.
        port: Rendezvous port.
    """
    topology = join_group(rank, world, port)
    try:
        program = Program(provide={"model": model()}, stages=[stage(topology)])
        Runner(program, seed=42, topology=topology, hooks=[]).run()
        weights = topology.state_of(program.provide["model"])
        results[rank] = {k: v.tolist() for k, v in weights.items()}
    finally:
        topology.close()


def _checkpoints(rank, world, results, port, store):
    """Train with a checkpointer and report what this rank wrote.

    Args:
        rank: Index of this process.
        world: Number of processes taking part.
        results: Shared mapping the ranks report into.
        port: Rendezvous port.
        store: Where checkpoints are written.
    """
    topology = join_group(rank, world, port)
    try:
        program = Program(provide={"model": model()}, stages=[stage(topology)])
        wrote = Counting()
        Runner(
            program,
            seed=42,
            topology=topology,
            store=store,
            hooks=[Checkpointer(every=1, unit="epoch"), wrote],
        ).run()
        results[rank] = (topology.is_main, wrote.count)
    finally:
        topology.close()


def _traces(rank, world, results, port, store):
    """Train while recording a trace into a store every rank shares.

    Args:
        rank: Index of this process.
        world: Number of processes taking part.
        results: Shared mapping the ranks report into.
        port: Rendezvous port.
        store: Where the trace is written.
    """
    topology = join_group(rank, world, port)
    try:
        program = Program(provide={"model": model()}, stages=[stage(topology)])
        Runner(program, seed=42, topology=topology, store=store, hooks=[Jsonl()]).run()
    finally:
        topology.close()


def _repeats(rank, world, results, port):
    """Repeat a stage holding its model, and report how it was wrapped.

    Args:
        rank: Index of this process.
        world: Number of processes taking part.
        results: Shared mapping the ranks report into.
        port: Rendezvous port.
    """
    topology = join_group(rank, world, port)
    try:
        fit = stage(topology, model=model())
        Runner(Program([Repeat(2, fit)]), seed=42, topology=topology).run()
        results[rank] = (type(fit.model).__name__, type(fit.model.module).__name__)
    finally:
        topology.close()


class Counting:
    """Counts the checkpoints this process reported writing."""

    def __init__(self):
        """Constructor."""
        self.count = 0

    def on(self, event):
        """Note a checkpoint this rank wrote.

        Args:
            event: What the runner just did.
        """
        if event.type is EventType.CHECKPOINT:
            self.count += 1
        return Signal.GO


class BreakOnMain:
    """Stop the run, but decide it on rank 0 only."""

    def __init__(self, topology, after: int):
        """Constructor.

        Args:
            topology: Process layout, for deciding who may stop the run.
            after: Steps to allow before stopping.
        """
        self.topology = topology
        self.after = after
        self.seen = 0

    def on(self, event):
        """Stop once enough steps have gone by, on the main rank only.

        Args:
            event: What the runner just did.
        """
        if event.type is EventType.STEP_ENDED and self.topology.is_main:
            self.seen += 1
            if self.seen >= self.after:
                return Signal.BREAK
        return Signal.GO


def _breaks_on_one_rank(rank, world, results, port):
    """Let only rank 0 ask to stop, and report where each rank ended.

    Args:
        rank: Index of this process.
        world: Number of processes taking part.
        results: Shared mapping the ranks report into.
        port: Rendezvous port.
    """
    topology = join_group(rank, world, port)
    try:
        trainer = stage(topology)
        program = Program(provide={"model": model()}, stages=[trainer])
        Runner(
            program,
            seed=42,
            topology=topology,
            hooks=[BreakOnMain(topology, after=2)],
        ).run()
        results[rank] = trainer.progress().global_step
    finally:
        topology.close()


class AbortOffMain:
    """Raise on a rank that is not the one others watch for decisions."""

    def __init__(self, topology, after: int):
        """Constructor.

        Args:
            topology: Process layout, for deciding who raises.
            after: Steps to allow before raising.
        """
        self.topology = topology
        self.after = after
        self.seen = 0

    def on(self, event):
        """Raise once enough steps have gone by, off the main rank.

        Args:
            event: What the runner just did.

        Raises:
            C3liRunError: On the non-main rank, once `after` is reached.
        """
        if event.type is EventType.STEP_ENDED and not self.topology.is_main:
            self.seen += 1
            if self.seen >= self.after:
                raise C3liRunError("cancelled on a follower")
        return Signal.GO


def _aborts_off_main(rank, world, results, port):
    """Raise on the follower and report whether every rank unwound.

    Args:
        rank: Index of this process.
        world: Number of processes taking part.
        results: Shared mapping the ranks report into.
        port: Rendezvous port.
    """
    topology = join_group(rank, world, port)
    try:
        program = Program(provide={"model": model()}, stages=[stage(topology)])
        try:
            Runner(
                program,
                seed=42,
                topology=topology,
                hooks=[AbortOffMain(topology, after=2)],
            ).run()
            results[rank] = "finished"
        except C3liRunError as failure:
            results[rank] = f"raised: {failure}"
    finally:
        topology.close()


def _shards(rank, world, results, port, samples):
    """Report how many batches this rank was dealt.

    Args:
        rank: Index of this process.
        world: Number of processes taking part.
        results: Shared mapping the ranks report into.
        port: Rendezvous port.
        samples: Size of the dataset to deal out.
    """
    from chuchichaestli.programs.context import Context

    topology = join_group(rank, world, port)
    try:
        manager = DataManager(ramp(samples), batch_size=3)
        ctx = Context("program", 42, topology=topology, device=topology.device)
        results[rank] = len(manager.sharded_plan(ctx, epoch=0))
    finally:
        topology.close()


def _evaluates(rank, world, results, port, samples=8):
    """Evaluate across ranks and report what each one published.

    Args:
        rank: Index of this process.
        world: Number of processes taking part.
        results: Shared mapping the ranks report into.
        port: Rendezvous port.
        samples: Size of the dataset evaluated.
    """
    topology = join_group(rank, world, port)
    try:
        probe = Eval(
            "probe", model=model(), data=ramp(samples), batch_size=2, metrics=[MSE()]
        )
        program = Program(stages=[probe])
        Runner(program, seed=42, topology=topology, hooks=[]).run()
        metric = probe.metrics["mse"]
        results[rank] = (
            float(metric.compute()),
            float(metric.n_observations),
            bool(metric.is_nan),
        )
    finally:
        topology.close()


def test_every_rank_publishes_the_same_metric():
    """A later `When` reading one must decide the same way everywhere."""
    reported = under_ddp(_evaluates, free_port())
    assert reported[0] == reported[1]


def test_a_reduced_metric_saw_the_whole_dataset():
    """Each rank sees its shard, so an unreduced count would be short."""
    reported = under_ddp(_evaluates, free_port())
    alone = Eval("probe", model=model(), data=ramp(), batch_size=2, metrics=[MSE()])
    Runner(Program(stages=[alone]), seed=42, hooks=[]).run()
    assert reported[0][1] == float(alone.metrics["mse"].n_observations)
    assert reported[0][0] == pytest.approx(float(alone.metrics["mse"].compute()))


def test_a_reduced_metric_scores_a_repeated_batch_once():
    """Nine samples in batches of two pad one rank's share with a repeat."""
    reported = under_ddp(_evaluates, free_port(), 9)
    alone = Eval("probe", model=model(), data=ramp(9), batch_size=2, metrics=[MSE()])
    Runner(Program(stages=[alone]), seed=42, hooks=[]).run()
    assert reported[0][1] == float(alone.metrics["mse"].n_observations) == 9.0
    assert reported[0][0] == pytest.approx(float(alone.metrics["mse"].compute()))


def predicted_then_scored() -> tuple[Predict, Eval, Program]:
    """Build a program scoring what an earlier stage predicted, in memory."""
    guess = Predict(
        "guess", model=model(), data=ramp(9), batch_size=2, inputs=("x",), targets="y"
    )
    score = Eval("score", model=None, data="guess/pairs", batch_size=2, metrics=[MSE()])
    return guess, score, Program(stages=[guess, score])


def _scores_predictions(rank, world, results, port):
    """Predict, then score the pairs, and report what each rank ended with.

    Args:
        rank: Index of this process.
        world: Number of processes taking part.
        results: Shared mapping the ranks report into.
        port: Rendezvous port.
    """
    topology = join_group(rank, world, port)
    try:
        guess, score, program = predicted_then_scored()
        Runner(program, seed=42, topology=topology, hooks=[]).run()
        metric = score.metrics["mse"]
        results[rank] = (
            guess.predictions.tolist(),
            float(metric.n_observations),
            float(metric.compute()),
        )
    finally:
        topology.close()


def test_a_later_stage_scores_every_rank_s_predictions():
    """Each rank keeps only its share, which a later stage would split again."""
    from chuchichaestli.programs.topology import Local

    reported = under_ddp(_scores_predictions, free_port())
    guess, score, program = predicted_then_scored()
    Runner(program, seed=42, topology=Local(device="cpu"), hooks=[]).run()
    metric = score.metrics["mse"]
    for predictions, observed, value in reported.values():
        assert predictions == guess.predictions.tolist()
        assert observed == float(metric.n_observations) == 9.0
        assert value == pytest.approx(float(metric.compute()))


def _weighs_uneven_shards(rank, world, results, port, balance):
    """Take one step over shards of different sizes and report the weights.

    Six samples in batches of four leave rank 0 with four and rank 1 with
    two, which is the case the balancing exists for.

    Args:
        rank: Index of this process.
        world: Number of processes taking part.
        results: Shared mapping the ranks report into.
        port: Rendezvous port.
        balance: Whether a batch is weighed against every rank's share.
    """
    topology = join_group(rank, world, port)
    try:
        trainer = Train(
            "fit",
            data=DataManager(ramp(6), batch_size=4, balance_ranks=balance),
            loss=nn.MSELoss(),
            steps=1,
            optim=OptimSpec.sgd(lr=1.0),
        )
        program = Program(provide={"model": model()}, stages=[trainer])
        Runner(program, seed=42, topology=topology, hooks=[]).run()
        weights = topology.state_of(program.provide["model"])
        results[rank] = weights["weight"].tolist()
    finally:
        topology.close()


def one_step_over_everything() -> list:
    """Return the weights one process reaches seeing all six samples at once."""
    trainer = Train(
        "fit",
        data=DataManager(ramp(6), batch_size=6),
        loss=nn.MSELoss(),
        steps=1,
        optim=OptimSpec.sgd(lr=1.0),
    )
    program = Program(provide={"model": model()}, stages=[trainer])
    Runner(program, seed=42, hooks=[]).run()
    return program.provide["model"].weight.detach().tolist()


def test_balancing_ranks_weighs_a_shard_by_what_it_holds():
    """Averaged gradients would otherwise count a short shard in full."""
    balanced = under_ddp(_weighs_uneven_shards, free_port(), True)
    alone = torch.tensor(one_step_over_everything())
    assert torch.allclose(torch.tensor(balanced[0]), alone, atol=1e-6)


def test_without_balancing_a_short_shard_counts_in_full():
    """The correction is off by default, so the mis-weighting is observable."""
    plain_shards = under_ddp(_weighs_uneven_shards, free_port(), False)
    alone = torch.tensor(one_step_over_everything())
    assert not torch.allclose(torch.tensor(plain_shards[0]), alone, atol=1e-6)


def _predicts(rank, world, results, port, archive, merge):
    """Predict on every rank and report what this one wrote.

    Args:
        rank: Index of this process.
        world: Number of processes taking part.
        results: Shared mapping the ranks report into.
        port: Rendezvous port.
        archive: File the predictions are written to.
        merge: Whether the shards are joined when the run ends.
    """
    topology = join_group(rank, world, port)
    try:
        sampler = Predict(
            "sample",
            model=model(),
            data=ramp(8),
            batch_size=2,
            archive=archive,
            merge=merge,
        )
        Runner(Program(stages=[sampler]), seed=42, topology=topology, hooks=[]).run()
        results[rank] = str(sampler._written)
    finally:
        topology.close()


def _predicts_pairs(rank, world, results, port, archive, samples=8):
    """Predict with targets on every rank and report the pairs this one reads.

    Args:
        rank: Index of this process.
        world: Number of processes taking part.
        results: Shared mapping the ranks report into.
        port: Rendezvous port.
        archive: File the predictions are written to.
        samples: Size of the dataset predicted.
    """
    topology = join_group(rank, world, port)
    try:
        first_input = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            first_input.weight.copy_(torch.tensor([[1.0, 0.0, 0.0]]))
        store = {}
        program = Program(
            stages=[
                Predict(
                    "sample",
                    model=first_input,
                    data=ramp(samples),
                    batch_size=2,
                    targets="y",
                    archive=archive,
                ),
                Call("read", fn=lambda ctx: store.update(pairs=ctx["sample/pairs"])),
            ]
        )
        Runner(program, seed=42, topology=topology, hooks=[]).run()
        pairs = store["pairs"]
        results[rank] = [
            (pairs[index][0].tolist(), pairs[index][1].tolist())
            for index in range(len(pairs))
        ]
    finally:
        topology.close()


def test_archived_pairs_stay_aligned_across_ranks(tmp_path):
    """Each shard's predictions and targets are joined in the same rank order."""
    archive = tmp_path / "out.h5"
    reported = under_ddp(_predicts_pairs, free_port(), str(archive))
    assert sorted(p.name for p in tmp_path.glob("*.h5")) == [
        "out.h5",
        "out.targets.h5",
    ]
    for rows in reported.values():
        assert len(rows) == 8
        assert all(y == truth for y, truth in rows)
    assert sorted(truth for _, truth in reported[0]) == ramp(8).tensors[1].tolist()


def test_archived_pairs_hold_each_sample_once(tmp_path):
    """A batch repeated to even out the shares must not land in the archive."""
    archive = tmp_path / "out.h5"
    reported = under_ddp(_predicts_pairs, free_port(), str(archive), 9)
    for rows in reported.values():
        assert len(rows) == 9
        assert all(y == truth for y, truth in rows)
    assert sorted(truth for _, truth in reported[0]) == ramp(9).tensors[1].tolist()


def _archive_exports(rank, world, results, port, path):
    """Export a binding on every rank and report how the run ended.

    Args:
        rank: Index of this process.
        world: Number of processes taking part.
        results: Shared mapping the ranks report into.
        port: Rendezvous port.
        path: File the export would land on.
    """
    topology = join_group(rank, world, port)
    try:
        program = Program(
            stages=[ArchiveExport("save", path=path, source="samples")],
            provide={"samples": torch.zeros(4, 2)},
        )
        try:
            Runner(program, seed=42, topology=topology, hooks=[]).run()
            results[rank] = "finished"
        except Exception as failure:
            results[rank] = f"raised: {failure}"
    finally:
        topology.close()


def test_an_archive_export_refuses_a_run_of_several_processes(tmp_path):
    """Rank 0 alone would write only its own share of a prediction."""
    reported = under_ddp(_archive_exports, free_port(), str(tmp_path / "out.h5"))
    assert all("Predict(archive=...)" in outcome for outcome in reported.values())
    assert not list(tmp_path.glob("*.h5"))


def test_the_shards_are_joined_into_one_file(tmp_path):
    """Ranks see different data, and the run should leave one file, not many."""
    archive = tmp_path / "out.h5"
    reported = under_ddp(_predicts, free_port(), str(archive), True)
    assert sorted(p.name for p in tmp_path.glob("*.h5")) == ["out.h5"]
    assert set(reported.values()) == {str(archive)}
    rows = torch.cat(list(read_archive(archive, "data")))
    assert len(rows) == 8


def test_the_shards_can_be_left_apart(tmp_path):
    """A large output need not be funnelled through one process."""
    archive = tmp_path / "out.h5"
    under_ddp(_predicts, free_port(), str(archive), False)
    assert sorted(p.name for p in tmp_path.glob("*.h5")) == [
        "out.rank0.h5",
        "out.rank1.h5",
    ]
    rows = sum(
        len(torch.cat(list(read_archive(shard, "data"))))
        for shard in tmp_path.glob("*.h5")
    )
    assert rows == 8


def test_every_rank_agrees_on_the_weights():
    """Gradients are averaged, so the replicas must not drift apart."""
    reported = under_ddp(_trains, free_port())
    assert len(reported) == WORLD
    assert reported[0].keys() == reported[1].keys()
    for key, value in reported[0].items():
        assert torch.equal(torch.tensor(value), torch.tensor(reported[1][key]))


def test_the_captured_weights_carry_no_wrapper_prefix():
    """A checkpoint written distributed has to load into a single process."""
    reported = under_ddp(_trains, free_port())
    assert sorted(reported[0]) == ["weight"]


def test_only_the_main_rank_writes_checkpoints(tmp_path):
    """Two ranks writing the same directory would race and corrupt it."""
    store = tmp_path / "run"
    reported = under_ddp(_checkpoints, free_port(), str(store))
    main, written = reported[0]
    follower, quiet = reported[1]
    assert main and written > 0
    assert not follower and quiet == 0
    assert sorted(p.name for p in (store).glob("ckpt_*"))


def test_only_the_main_rank_writes_the_trace(tmp_path):
    """Every rank appending to one file would duplicate and interleave events."""
    store = tmp_path / "run"
    under_ddp(_traces, free_port(), str(store))
    events = Jsonl.read(store / "trace.jsonl")
    assert [e.type for e in events].count(EventType.RUN_BEGAN) == 1
    assert events[-1].type is EventType.RUN_ENDED


def test_a_repeated_stage_wraps_its_model_once():
    """A replica of a replica would reduce every gradient twice."""
    reported = under_ddp(_repeats, free_port())
    assert set(reported.values()) == {("DistributedDataParallel", "Linear")}


def test_a_stop_decided_on_one_rank_ends_them_all():
    """The others would otherwise block forever on the next collective."""
    reported = under_ddp(_breaks_on_one_rank, free_port())
    assert reported[0] == reported[1]


def test_an_abort_on_a_follower_unwinds_every_rank():
    """Raising is not self-synchronizing, so it is broadcast like any signal."""
    reported = under_ddp(_aborts_off_main, free_port())
    assert all(str(outcome).startswith("raised") for outcome in reported.values())


@pytest.mark.parametrize("samples", [8, 9, 10])
def test_every_rank_is_dealt_the_same_number_of_batches(samples):
    """An uneven share is a deadlock, not a rounding error."""
    reported = under_ddp(_shards, free_port(), samples)
    assert reported[0] == reported[1]


def test_a_topology_unwraps_what_it_wrapped():
    """`unwrap` is `wrap`'s inverse, so each layout undoes its own."""
    from torch.nn.parallel import DistributedDataParallel
    from chuchichaestli.programs.topology import Local

    plain = nn.Linear(2, 1)
    local = Local()
    assert local.unwrap(local.wrap(plain)) is plain

    replica = DistributedDataParallel.__new__(DistributedDataParallel)
    object.__setattr__(replica, "_modules", {"module": plain})
    assert local.unwrap(replica) is replica


def _export(rank, world, results, port, directory):
    """Export a wrapped model's weights from a distributed run.

    Args:
        rank: Index of this process.
        world: Number of processes taking part.
        results: Shared mapping the ranks report into.
        port: Rendezvous port.
        directory: Where the weights are written.
    """
    from pathlib import Path
    from chuchichaestli.programs import Program, Runner, WeightsExport
    from chuchichaestli.utils.io import read_state

    topology = join_group(rank, world, port)
    try:
        program = Program(
            provide={"model": nn.Linear(3, 2)},
            stages=[WeightsExport("out", path=Path(directory) / "w.safetensors")],
        )
        Runner(program, hooks=[], topology=topology).run()
        if rank == 0:
            results["keys"] = sorted(read_state(Path(directory) / "w.safetensors"))
    finally:
        topology.close()


def test_an_export_from_a_replica_loads_into_a_plain_module(tmp_path):
    """A `module.` prefix makes the weights unloadable outside a DDP run."""
    reported = under_ddp(_export, free_port(), str(tmp_path))
    assert reported["keys"] == ["bias", "weight"]


def _unbound_ddp(device: torch.device) -> Ddp:
    """Return a `Ddp` pinned to a device without joining a process group.

    Args:
        device: Device the rank computes on.
    """
    topology = Ddp.__new__(Ddp)
    topology.device = device
    return topology


def test_a_rank_makes_its_own_device_the_process_default(monkeypatch):
    """A collective given no device uses the current one, so it must be ours."""
    from types import SimpleNamespace

    pinned: list[int] = []
    monkeypatch.setattr(
        torch,
        "accelerator",
        SimpleNamespace(set_device_index=pinned.append),
        raising=False,
    )
    _unbound_ddp(torch.device("cuda", 3))._pin_process()
    assert pinned == [3]


def test_a_rank_pins_through_the_backend_without_an_accelerator(monkeypatch):
    """Torch below 2.6 has no accelerator module, but every backend pins."""
    monkeypatch.delattr(torch, "accelerator", raising=False)
    pinned: list[int] = []
    monkeypatch.setattr(torch.cuda, "set_device", pinned.append)
    _unbound_ddp(torch.device("cuda", 2))._pin_process()
    assert pinned == [2]


def test_a_cpu_rank_pins_no_device(monkeypatch):
    """There is no cpu index to meet on, so pinning would fail rather than help."""
    from types import SimpleNamespace

    pinned: list[int] = []
    monkeypatch.setattr(
        torch,
        "accelerator",
        SimpleNamespace(set_device_index=pinned.append),
        raising=False,
    )
    _unbound_ddp(torch.device("cpu"))._pin_process()
    assert pinned == []


def test_a_rank_pins_its_device_before_joining_the_group(monkeypatch):
    """`nccl` binds to the current device as it initializes, so pinning precedes it."""
    import torch.distributed as dist

    order: list[str] = []
    monkeypatch.setattr(Ddp, "_pin_process", lambda self: order.append("pin"))
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        dist, "init_process_group", lambda **kwargs: order.append("init")
    )
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    Ddp(device="cpu")
    assert order == ["pin", "init"]
