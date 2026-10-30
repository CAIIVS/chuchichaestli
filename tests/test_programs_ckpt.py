# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for checkpointing a run and carrying on from one."""

import json

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from chuchichaestli.metrics import MSE
from chuchichaestli.programs import (
    C3liCheckpointError,
    C3liProgramError,
    Call,
    Checkpointer,
    CheckpointStore,
    EventType,
    Jsonl,
    Local,
    Program,
    Runner,
    Signal,
    unpack_tree,
    stage_signature,
    merge_tree,
)
from chuchichaestli.models.spec import ModelSpec, ModelSpecMixin, load_model, read_spec
from chuchichaestli.programs.stages import Eval, Phase, StageBlock, Train
from chuchichaestli.training import OptimSpec
from chuchichaestli.programs.ckpt import SCHEMA_VERSION
from chuchichaestli.programs.serialize import writable_spec
from chuchichaestli.utils.io import read_state


class StopAfter:
    """Break the run once a number of stage advances have gone by."""

    def __init__(self, advances: int):
        """Constructor.

        Args:
            advances: How many advances to allow before breaking.
        """
        self.advances = advances
        self.seen = 0

    def on(self, event):
        """Count advances and break when enough have happened.

        Args:
            event: What the runner just did.
        """
        if event.type is EventType.STAGE_ADVANCED:
            self.seen += 1
            if self.seen >= self.advances:
                return Signal.BREAK
        return Signal.GO


def recorder(ran, names="abcd"):
    """Build a program of calls that each note they ran.

    Args:
        ran: List the calls append their name to.
        names: One call per character.
    """
    return Program([Call(n, fn=lambda ctx, n=n: ran.append(n)) for n in names])


def stage_trace(path, root="program"):
    """Return the trace's per-stage events as `(type, path)` pairs.

    The root's own pair is dropped, every run bracketing the program it ran.

    Args:
        path: JSONL file the `Jsonl` hook wrote.
        root: Path of the program itself.
    """
    events = [json.loads(line) for line in path.read_text().splitlines()]
    return [
        (e["type"], e["path"])
        for e in events
        if e["type"].startswith("stage.") and e["path"] != root
    ]


def test_flatten_splits_tensors_from_everything_else():
    """A tensor file holds tensors; the manifest holds the shape around them."""
    state = {"a": torch.ones(2), "b": {"c": 3, "d": torch.zeros(1)}, "e": "text"}
    tensors, skeleton = unpack_tree(state)
    assert sorted(tensors) == ["a", "b/d"]
    assert skeleton["b"]["c"] == 3
    assert skeleton["e"] == "text"
    assert json.dumps(skeleton)


def test_flatten_round_trips_a_state_tree():
    """What comes back has to be what went in."""
    state = {"a": torch.arange(3), "b": [1, {"c": torch.ones(2)}], "d": None}
    restored = merge_tree(*unpack_tree(state))
    assert torch.equal(restored["a"], state["a"])
    assert torch.equal(restored["b"][1]["c"], state["b"][1]["c"])
    assert restored["d"] is None


def test_an_optimizers_integer_keys_survive():
    """Stringified indices do not raise; they silently orphan the momentum.

    Torch maps saved integer ids onto the new parameters, so a string key
    matches nothing and the entry is kept unattached: every parameter resumes
    with its moments back at zero.
    """
    model = nn.Linear(3, 3)
    optimizer = torch.optim.Adam(model.parameters())
    for _ in range(3):
        optimizer.zero_grad()
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()
    saved = optimizer.state_dict()

    restored = merge_tree(*unpack_tree(saved))
    assert all(isinstance(key, int) for key in restored["state"])

    fresh = nn.Linear(3, 3)
    reloaded = torch.optim.Adam(fresh.parameters())
    reloaded.load_state_dict(restored)
    carried = [p for p in fresh.parameters() if "exp_avg" in reloaded.state.get(p, {})]
    assert len(carried) == len(list(fresh.parameters()))
    assert all(reloaded.state[p]["step"].item() == 3 for p in carried)


def test_stringified_optimizer_keys_lose_the_momentum_silently():
    """The failure the pair encoding exists to prevent, pinned."""
    model = nn.Linear(3, 3)
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.zero_grad()
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    saved = optimizer.state_dict()

    fresh = nn.Linear(3, 3)
    victim = torch.optim.Adam(fresh.parameters())
    victim.load_state_dict(
        {**saved, "state": {str(k): v for k, v in saved["state"].items()}}
    )
    assert not [p for p in fresh.parameters() if p in victim.state]


def test_a_reference_to_a_missing_tensor_is_refused():
    """Better to say the file is short than to hand back a hole."""
    _, skeleton = unpack_tree({"a": torch.ones(2)})
    with pytest.raises(C3liCheckpointError, match="does not hold"):
        merge_tree({}, skeleton)


def test_the_tree_shape_names_every_stage():
    """The shape is what a resume compares against."""
    shape = stage_signature(recorder([], "ab"))
    assert shape[0] == ["program", "Program"]
    assert ["program/0:a", "Call"] in shape


def test_a_store_rejects_an_unusable_configuration(tmp_path):
    """Both mistakes are worth catching before a run starts."""
    with pytest.raises(ValueError, match="at least one checkpoint"):
        CheckpointStore(tmp_path, keep=0)
    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        CheckpointStore(tmp_path, format="pickle")


def test_a_checkpoint_round_trips_through_the_store(tmp_path):
    """Program state and binding state both have to come back."""
    model = nn.Linear(2, 2)
    program = recorder([], "ab")
    store = CheckpointStore(tmp_path)
    written = store.save(
        index=7,
        program=program,
        bindings={"model": model},
        topology=Local(device="cpu"),
        seed=11,
    )
    assert written is not None

    read = store.load("last")
    assert read.index == 7
    assert read.seed == 11
    assert read.version == SCHEMA_VERSION
    assert torch.equal(read.bindings["model"]["weight"], model.weight)


def test_resume_accepts_a_directory_or_its_manifest(tmp_path):
    """Tab completion tends to land on the file rather than the directory."""
    store = CheckpointStore(tmp_path)
    written = store.save(
        index=1, program=recorder([], "a"), bindings={}, topology=Local(device="cpu")
    )
    assert store.resolve(written.path) == written.path
    assert store.resolve(written.path / store.manifest) == written.path


def test_an_interrupted_checkpoint_is_passed_over(tmp_path):
    """A directory with no readable manifest was never finished."""
    store = CheckpointStore(tmp_path)
    good = store.save(
        index=1, program=recorder([], "a"), bindings={}, topology=Local(device="cpu")
    )
    broken = store.directory_for(2)
    broken.mkdir()
    (broken / "state.safetensors").write_bytes(b"")

    assert store.checkpoints() == [good.path]
    assert store.resolve("last") == good.path


def explode(path, payload):
    """Fail to write a manifest.

    Args:
        path: File to write.
        payload: What to serialize.
    """
    raise OSError("disk full")


def test_the_manifest_lands_after_the_state(tmp_path, monkeypatch):
    """A crash between the two must leave no checkpoint, not a stale pairing."""
    store = CheckpointStore(tmp_path)
    monkeypatch.setattr(CheckpointStore, "_write_json", staticmethod(explode))
    with pytest.raises(OSError):
        store.save(
            index=1,
            program=recorder([], "a"),
            bindings={},
            topology=Local(device="cpu"),
        )
    assert not store.directory_for(1).exists()
    assert store.checkpoints() == []


def test_an_unfinished_write_leaves_no_scratch_files(tmp_path, monkeypatch):
    """The staged directory goes, whatever failed inside it."""
    store = CheckpointStore(tmp_path)
    monkeypatch.setattr(
        "chuchichaestli.programs.ckpt.write_state",
        lambda *a, **k: (_ for _ in ()).throw(OSError("nope")),
    )
    with pytest.raises(OSError):
        store.save(
            index=1,
            program=recorder([], "a"),
            bindings={},
            topology=Local(device="cpu"),
        )
    assert list(tmp_path.iterdir()) == []


def test_a_failed_rewrite_leaves_the_checkpoint_it_would_replace(tmp_path, monkeypatch):
    """A final save at the index of the last one must not half-replace it."""
    store = CheckpointStore(tmp_path)
    save = dict(
        index=1, program=recorder([], "a"), bindings={}, topology=Local(device="cpu")
    )
    store.save(seed=1, **save)
    with monkeypatch.context() as patched:
        patched.setattr(CheckpointStore, "_write_json", staticmethod(explode))
        with pytest.raises(OSError):
            store.save(seed=2, **save)
    assert store.load(1).seed == 1
    store.save(seed=2, **save)
    assert store.load(1).seed == 2
    assert [p.name for p in tmp_path.iterdir()] == [store.directory_for(1).name]


def test_keep_retains_only_the_newest(tmp_path):
    """Long runs must not fill the disk with every step they passed."""
    store = CheckpointStore(tmp_path, keep=2)
    for index in (1, 2, 3, 4):
        store.save(
            index=index,
            program=recorder([], "a"),
            bindings={},
            topology=Local(device="cpu"),
        )
    assert [p.name for p in store.checkpoints()] == [
        store.directory_for(3).name,
        store.directory_for(4).name,
    ]


def test_a_changed_stage_signature_is_refused(tmp_path):
    """Reordered stages misalign every key the checkpoint holds."""
    store = CheckpointStore(tmp_path)
    checkpoint = store.load(
        store.save(
            index=1,
            program=recorder([], "ab"),
            bindings={},
            topology=Local(device="cpu"),
        ).path
    )
    with pytest.raises(C3liCheckpointError, match="different stage signature"):
        store.restore(
            checkpoint,
            program=recorder([], "abc"),
            bindings={},
            topology=Local(device="cpu"),
        )
    store.restore(
        checkpoint,
        program=recorder([], "abc"),
        bindings={},
        topology=Local(device="cpu"),
        allow_signature_change=True,
    )


def test_a_manifest_from_a_later_schema_is_refused(tmp_path):
    """Guessing at a format this version does not know would be worse."""
    store = CheckpointStore(tmp_path)
    written = store.save(
        index=1, program=recorder([], "a"), bindings={}, topology=Local(device="cpu")
    )
    manifest_path = written.path / store.manifest
    manifest = json.loads(manifest_path.read_text())
    manifest["version"] = SCHEMA_VERSION + 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(C3liCheckpointError, match="understands at most"):
        store.load(written.path)


def test_a_manifest_that_does_not_name_its_state_file_still_loads(tmp_path):
    """The shared tensor file is found by suffix when the manifest is silent."""
    written = CheckpointStore(tmp_path, format="torch").save(
        index=1, program=recorder([], "a"), bindings={}, topology=Local(device="cpu")
    )
    store = CheckpointStore(tmp_path)
    manifest_path = written.path / store.manifest
    manifest = json.loads(manifest_path.read_text())
    del manifest["state_file"]
    manifest_path.write_text(json.dumps(manifest))
    assert store.load(written.path).rng is not None


def scored(store, index, value, monitor="score", mode="min"):
    """Write an empty checkpoint ranked by a monitored value.

    Args:
        store: Store to write into.
        index: Where in the run it is taken.
        value: The monitored value it records.
        monitor: Name of what was monitored.
        mode: Whether smaller or larger is better.
    """
    return store.save(
        index=index,
        program=recorder([], "a"),
        bindings={},
        topology=Local(device="cpu"),
        score={"monitor": monitor, "mode": mode, "value": value},
    )


def reporter(values, monitor="score"):
    """Build a program whose calls each report one monitored value.

    Args:
        values: One call per value, reporting it in order.
        monitor: Payload key the values are reported under.
    """
    return Program(
        [
            Call(
                f"r{i}",
                fn=lambda ctx, v=v: ctx.emit(
                    EventType.STEP_ENDED, trains=False, **{monitor: v}
                ),
            )
            for i, v in enumerate(values)
        ]
    )


def test_best_needs_a_scored_checkpoint(tmp_path):
    """Nothing can be ranked when no writer monitored anything."""
    with pytest.raises(C3liCheckpointError, match="records a score"):
        CheckpointStore(tmp_path).resolve("best")


@pytest.mark.parametrize("mode, expected", [("min", 2), ("max", 3)], ids=["min", "max"])
def test_best_follows_the_mode(tmp_path, mode, expected):
    """Smaller wins under min, larger under max."""
    store = CheckpointStore(tmp_path)
    for index, value in enumerate([3.0, 1.0, 5.0, 2.0], start=1):
        scored(store, index, value, mode=mode)
    assert store.resolve("best") == store.directory_for(expected)
    assert store.load("best").score == {
        "monitor": "score",
        "mode": mode,
        "value": {"min": 1.0, "max": 5.0}[mode],
    }


def test_best_goes_to_the_newer_of_a_tie(tmp_path):
    """The newer checkpoint has trained at least as long."""
    store = CheckpointStore(tmp_path)
    for index, value in enumerate([2.0, 1.0, 1.0], start=1):
        scored(store, index, value)
    assert store.resolve("best") == store.directory_for(3)


def test_best_ignores_a_nan_score(tmp_path):
    """A diverged evaluation is not the best one."""
    store = CheckpointStore(tmp_path)
    scored(store, 1, float("nan"))
    scored(store, 2, 4.0)
    assert store.resolve("best") == store.directory_for(2)


def test_best_refuses_scores_of_different_monitors(tmp_path):
    """A loss and an accuracy cannot be ranked against each other."""
    store = CheckpointStore(tmp_path)
    scored(store, 1, 0.5, monitor="loss")
    scored(store, 2, 0.9, monitor="accuracy", mode="max")
    with pytest.raises(C3liCheckpointError, match="different monitors"):
        store.resolve("best")


def test_keep_never_prunes_the_best(tmp_path):
    """Pruning the oldest must not delete the checkpoint worth keeping."""
    store = CheckpointStore(tmp_path, keep=1)
    for index, value in enumerate([3.0, 1.0, 2.0, 5.0], start=1):
        scored(store, index, value)
    assert store.checkpoints() == [store.directory_for(2), store.directory_for(4)]


def test_a_monitored_improvement_writes_a_checkpoint(tmp_path):
    """Only values that beat the best so far are worth a write."""
    store = tmp_path / "run"
    hook = Checkpointer(at_end=False, monitor="score")
    Runner(reporter([3.0, 1.0, 2.0, 0.5, 4.0]), store=store, hooks=[hook]).run()
    written = CheckpointStore(store)
    assert [written.load_index(p) for p in written.checkpoints()] == [1, 2, 4]
    assert written.load("best").score["value"] == 0.5


def test_a_periodic_checkpoint_records_the_value_reported_with_it(tmp_path):
    """Every write in the boundary after a report is scored by it."""
    store = tmp_path / "run"
    hook = Checkpointer(unit="advance", keep=1, monitor="score")
    Runner(reporter([3.0, 1.0, 2.0, 5.0]), store=store, hooks=[hook]).run()
    written = CheckpointStore(store)
    assert [written.load(p).score["value"] for p in written.checkpoints()] == [
        1.0,
        5.0,
    ]


def test_a_checkpoint_written_after_training_moved_on_carries_no_score(tmp_path):
    """A value describes the weights it was measured on, not later ones."""
    store = tmp_path / "run"
    program = Program(
        [
            Call("report", fn=lambda ctx: ctx.emit(EventType.STEP_ENDED, score=1.0)),
            Call("later", fn=lambda ctx: None),
        ]
    )
    hook = Checkpointer(unit="advance", every=2, at_end=False, monitor="score")
    Runner(program, store=store, hooks=[hook]).run()
    written = CheckpointStore(store)
    assert [written.load(p).score for p in written.checkpoints()] == [
        {"monitor": "score", "mode": "min", "value": 1.0}
    ]


def test_the_best_value_survives_a_resume(tmp_path):
    """A resumed run must not count a worse value as an improvement."""
    store = tmp_path / "run"
    values = [3.0, 1.0, 2.0, 2.5, 0.5]
    Runner(
        reporter(values),
        store=store,
        hooks=[Checkpointer(monitor="score"), StopAfter(3)],
    ).run()
    Runner(
        reporter(values),
        store=store,
        resume="last",
        hooks=[Checkpointer(at_end=False, monitor="score")],
    ).run()
    written = CheckpointStore(store)
    scores = {
        written.load_index(p): (written.load(p).score or {}).get("value")
        for p in written.checkpoints()
    }
    assert scores == {1: 3.0, 2: 1.0, 3: None, 5: 0.5}


def test_the_best_checkpoint_holds_the_weights_it_was_scored_on(tmp_path):
    """Ranking is only useful if the best checkpoint is the evaluated model."""
    rows = TensorDataset(torch.rand(8, 2), torch.rand(8, 1))
    fit = Train(
        "fit", data=rows, batch_size=4, loss=nn.MSELoss(), optim=OptimSpec.sgd(lr=0.5)
    )
    check = Eval("check", data=rows, batch_size=4, metrics=[MSE()])
    seen: list[tuple[float, dict[str, torch.Tensor]]] = []

    def snapshot(ctx):
        state = ctx["model"].state_dict()
        seen.append((ctx["check/mse"], {k: v.clone() for k, v in state.items()}))

    program = Program(
        [Phase.each_pass(6, fit, check, Call("snapshot", fn=snapshot))],
        provide={"model": nn.Linear(2, 1)},
    )
    store = tmp_path / "run"
    hook = Checkpointer(keep=1, monitor="mse")
    Runner(program, store=store, device="cpu", hooks=[hook]).run()
    value, expected = min(seen, key=lambda pair: pair[0])
    best = CheckpointStore(store).load("best")
    assert best.score["value"] == pytest.approx(value)
    for key, tensor in expected.items():
        assert torch.equal(best.bindings["model"][key], tensor)


def test_a_checkpointer_refuses_an_unknown_mode():
    """A typo must fail at construction, not rank checkpoints backwards."""
    with pytest.raises(ValueError, match="mode"):
        Checkpointer(monitor="score", mode="lowest")


def test_resume_continues_where_the_run_stopped(tmp_path):
    """The headline: no stage runs twice and none is skipped."""
    store = tmp_path / "run"
    first: list[str] = []
    Runner(
        recorder(first),
        store=store,
        hooks=[Checkpointer(every=1, unit="advance"), StopAfter(2)],
    ).run()
    second: list[str] = []
    Runner(recorder(second), store=store, resume="last", hooks=[]).run()

    whole: list[str] = []
    Runner(recorder(whole), hooks=[]).run()
    assert first + second == whole == ["a", "b", "c", "d"]


def test_the_resumed_trace_matches_the_uninterrupted_one(tmp_path):
    """A run that was interrupted must be indistinguishable from one that was not."""
    store = tmp_path / "run"
    Runner(
        recorder([]),
        store=store,
        hooks=[
            Checkpointer(every=1, unit="advance"),
            Jsonl("first.jsonl"),
            StopAfter(2),
        ],
    ).run()
    Runner(
        recorder([]),
        store=store,
        resume="last",
        hooks=[Jsonl("second.jsonl")],
    ).run()
    Runner(recorder([]), hooks=[Jsonl("whole.jsonl")], store=store).run()

    resumed = stage_trace(store / "first.jsonl") + stage_trace(store / "second.jsonl")
    assert resumed == stage_trace(store / "whole.jsonl")


def test_a_binding_is_restored_from_the_checkpoint(tmp_path):
    """Weights are what a checkpoint is mostly for."""
    store = tmp_path / "run"
    trained = nn.Linear(3, 3)
    with torch.no_grad():
        trained.weight.fill_(0.5)
    program = Program([Call("a", fn=lambda ctx: None)], provide={"model": trained})
    Runner(program, store=store, hooks=[Checkpointer(every=1, unit="advance")]).run()

    fresh = nn.Linear(3, 3)
    resumed = Program([Call("a", fn=lambda ctx: None)], provide={"model": fresh})
    Runner(
        resumed, store=store, resume="last", hooks=[], allow_signature_change=True
    ).run()
    assert torch.equal(fresh.weight, trained.weight)


def test_the_ambient_rng_state_comes_back(tmp_path):
    """Code that ignores an explicit generator still has to resume."""
    store = CheckpointStore(tmp_path)
    torch.manual_seed(1234)
    checkpoint = store.load(
        store.save(
            index=1,
            program=recorder([], "a"),
            bindings={},
            topology=Local(device="cpu"),
        ).path
    )
    expected = torch.rand(4)

    from chuchichaestli.utils.rng import restore_rng_state

    restore_rng_state(dict(checkpoint.rng))
    assert torch.equal(torch.rand(4), expected)


def test_check_rejects_a_resume_with_nothing_to_resume_from(tmp_path):
    """A plan error belongs before any compute, not on the first read."""
    runner = Runner(recorder([]), store=tmp_path / "run", resume="last", hooks=[])
    with pytest.raises(C3liProgramError, match="No complete checkpoint"):
        runner.check()


def test_check_rejects_a_checkpointer_with_nowhere_to_write():
    """The hook says it needs a store; the pre-flight enforces it."""
    runner = Runner(recorder([]), hooks=[Checkpointer()])
    with pytest.raises(C3liProgramError, match="needs a store"):
        runner.check()


def test_a_detached_checkpointer_says_so():
    """Used outside a run it has no program to write down."""
    from chuchichaestli.programs.events import C3liRunError, Event

    with pytest.raises(C3liRunError, match="never attached"):
        Checkpointer().save(Event(EventType.RUN_ENDED, "program"))


def test_a_checkpoint_announces_itself(tmp_path):
    """`Console` and `Jsonl` both render checkpoint events."""
    seen: list[str] = []

    class Watch:
        """Note the path of every checkpoint event."""

        def on(self, event):
            """Record checkpoint events.

            Args:
                event: What the runner just did.
            """
            if event.type is EventType.CHECKPOINT:
                seen.append(event.payload["path"])
            return Signal.GO

    Runner(
        recorder([], "ab"),
        store=tmp_path / "run",
        hooks=[Checkpointer(every=1, unit="advance"), Watch()],
    ).run()
    assert seen and all("ckpt_" in path for path in seen)


class Tiny(ModelSpecMixin, nn.Module):
    """A model that records how it was built."""

    def __init__(self, width: int = 4):
        """Constructor.

        Args:
            width: Width of the linear layer.
        """
        super().__init__()
        self.lin = nn.Linear(width, width)


class Opaque(ModelSpecMixin, nn.Module):
    """A model built from a component that records nothing."""

    def __init__(self, inner: nn.Module):
        """Constructor.

        Args:
            inner: Component to wrap.
        """
        super().__init__()
        self.inner = inner


def test_a_spec_is_only_offered_for_models_that_record_one():
    """An optimizer has no architecture to describe."""
    assert writable_spec(Tiny()) is not None
    assert writable_spec(nn.Linear(2, 2)) is None
    assert writable_spec(torch.optim.Adam(nn.Linear(2, 2).parameters())) is None


def test_a_spec_past_the_limit_is_left_out_with_a_warning(monkeypatch):
    """The weights are still written; only the spec is dropped."""
    monkeypatch.setattr(ModelSpec, "json_limit", 10)
    with pytest.warns(UserWarning, match="json_limit"):
        assert writable_spec(Tiny()) is None


def test_a_model_binding_records_its_spec(tmp_path):
    """Weights alone do not say which architecture built them."""
    store = CheckpointStore(tmp_path)
    model = Tiny(width=6)
    written = store.save(
        index=1,
        program=recorder([], "a"),
        bindings={"model": model, "optim": torch.optim.Adam(model.parameters())},
        topology=Local(device="cpu"),
    )
    manifest = json.loads((written.path / store.manifest).read_text())
    assert sorted(manifest["weights"]) == ["model"]
    assert read_spec(written.path / "model.safetensors").kwargs["width"] == 6


def test_a_checkpoint_rebuilds_its_model(tmp_path):
    """With a spec beside the weights, nothing else is needed."""
    store = CheckpointStore(tmp_path)
    model = Tiny(width=6)
    store.save(
        index=1,
        program=recorder([], "a"),
        bindings={"model": model},
        topology=Local(device="cpu"),
    )
    rebuilt = store.load("last").build("model")
    assert isinstance(rebuilt, Tiny)
    assert rebuilt.lin.in_features == 6
    assert torch.equal(rebuilt.lin.weight, model.lin.weight)


def test_building_a_binding_with_no_spec_says_so(tmp_path):
    """The refusal names what was recorded instead."""
    store = CheckpointStore(tmp_path)
    model = Tiny()
    store.save(
        index=1,
        program=recorder([], "a"),
        bindings={"model": model, "optim": torch.optim.Adam(model.parameters())},
        topology=Local(device="cpu"),
    )
    with pytest.raises(C3liCheckpointError, match="No model spec for binding 'optim'"):
        store.load("last").build("optim")


def test_an_unrenderable_spec_warns_but_keeps_the_weights(tmp_path):
    """A spec is a bonus; losing the checkpoint over one would not be."""
    store = CheckpointStore(tmp_path)
    model = Opaque(nn.Linear(3, 3))
    with pytest.warns(UserWarning, match="Not recording a model spec"):
        written = store.save(
            index=1,
            program=recorder([], "a"),
            bindings={"model": model},
            topology=Local(device="cpu"),
        )
    assert written.specs == {}
    assert torch.equal(
        store.load("last").bindings["model"]["inner.weight"], model.inner.weight
    )


def test_a_run_records_the_spec_of_what_it_provides(tmp_path):
    """The whole path, from Runner down."""
    store = tmp_path / "run"
    model = Tiny(width=5)
    program = Program([Call("a", fn=lambda ctx: None)], provide={"model": model})
    Runner(program, store=store, hooks=[Checkpointer(every=1, unit="advance")]).run()

    rebuilt = CheckpointStore(store).load("last").build("model")
    assert rebuilt.lin.in_features == 5
    assert torch.equal(rebuilt.lin.weight, model.lin.weight.detach().cpu())


class Epochs(StageBlock):
    """A stage that reports a number of epochs, a few units apart."""

    def __init__(self, name, epochs, units_per_epoch=2):
        """Constructor.

        Args:
            name: Identifies the stage within its parent.
            epochs: How many epochs to report.
            units_per_epoch: Units of work each epoch takes.
        """
        super().__init__(name)
        self.epochs = epochs
        self.units_per_epoch = units_per_epoch

    def core(self, ctx):
        """Nothing to do beyond the counting in `execute`.

        Args:
            ctx: Execution context for this entry.
        """

    def execute(self, ctx):
        """Take one unit, ending an epoch every few of them.

        Args:
            ctx: Execution context for this entry.
        """
        self._progress = self._progress.next_step()
        ctx.progress = self._progress
        if self._progress.step % self.units_per_epoch == 0:
            self._progress = self._progress.next_epoch()
            ctx.progress = self._progress
            ctx.emit(EventType.EPOCH_ENDED)
        done = self._progress.epoch >= self.epochs
        return Signal.DONE if done else Signal.GO


def test_an_unknown_unit_names_the_alternatives():
    """The message follows the package's usual shape."""
    with pytest.raises(ValueError, match="Unsupported checkpoint unit"):
        Checkpointer(unit="forever")


def test_the_unit_selects_which_event_is_counted():
    """Epoch and step events arrive once the loop stages report them."""
    assert Checkpointer(unit="advance").trigger is EventType.STAGE_ADVANCED
    assert Checkpointer(unit="epoch").trigger is EventType.EPOCH_ENDED
    assert Checkpointer(unit="step").trigger is EventType.STEP_ENDED


def test_counting_epochs_writes_one_checkpoint_per_epoch(tmp_path):
    """Three epochs, two units each, one checkpoint apiece."""
    store = tmp_path / "run"
    program = Program([Epochs("train", epochs=3), Call("after", fn=lambda ctx: None)])
    Runner(
        program, store=store, hooks=[Checkpointer(every=1, unit="epoch", at_end=False)]
    ).run()
    assert len(CheckpointStore(store).checkpoints()) == 3


def test_an_epoch_that_ends_the_run_is_still_checkpointed(tmp_path):
    """The last epoch has no advance after it to be written at."""
    store = tmp_path / "run"
    program = Program([Epochs("train", epochs=1, units_per_epoch=2)])
    Runner(
        program, store=store, hooks=[Checkpointer(every=1, unit="epoch", at_end=False)]
    ).run()
    assert len(CheckpointStore(store).checkpoints()) == 1


def test_a_resume_after_an_epoch_checkpoint_starts_clean(tmp_path):
    """A checkpoint written mid-`execute` would re-run the stage it was in."""
    store = tmp_path / "run"
    ran: list[str] = []
    program = Program(
        [Epochs("train", epochs=2), Call("after", fn=lambda ctx: ran.append("after"))]
    )
    Runner(
        program, store=store, hooks=[Checkpointer(every=1, unit="epoch", at_end=False)]
    ).run()

    checkpoint = CheckpointStore(store).load("last")
    assert checkpoint.program["entered"] is False
    assert checkpoint.program["child"] is None


def test_a_model_is_written_beside_the_run_state_not_inside_it(tmp_path):
    """The weights are their own artifact."""
    store = CheckpointStore(tmp_path)
    model = Tiny(width=6)
    written = store.save(
        index=1,
        program=recorder([], "a"),
        bindings={"model": model, "optim": torch.optim.Adam(model.parameters())},
        topology=Local(device="cpu"),
    )
    assert sorted(p.name for p in written.path.iterdir()) == [
        "manifest.json",
        "model.safetensors",
        "state.safetensors",
    ]
    assert set(read_state(written.path / "model.safetensors")) == set(
        model.state_dict()
    )
    assert not any(
        "model" in key for key in read_state(written.path / "state.safetensors")
    )


def test_an_optimizer_stays_with_the_run_state(tmp_path):
    """It is needed to resume, not to ship a model."""
    store = CheckpointStore(tmp_path)
    model = Tiny()
    optimizer = torch.optim.Adam(model.parameters())
    model.lin(torch.ones(1, 4)).sum().backward()
    optimizer.step()
    written = store.save(
        index=1,
        program=recorder([], "a"),
        bindings={"model": model, "optim": optimizer},
        topology=Local(device="cpu"),
    )
    tensors = read_state(written.path / "state.safetensors")
    assert any(key.startswith("bindings/optim/") for key in tensors)
    assert sorted(written.weights) == ["model"]


def test_a_model_file_opens_on_its_own(tmp_path):
    """No store, no manifest: the file says what built it."""
    store = CheckpointStore(tmp_path)
    model = Tiny(width=6)
    written = store.save(
        index=1,
        program=recorder([], "a"),
        bindings={"model": model},
        topology=Local(device="cpu"),
    )
    rebuilt = load_model(written.weights["model"])
    assert isinstance(rebuilt, Tiny)
    assert torch.equal(rebuilt.lin.weight, model.lin.weight)


def test_a_binding_name_that_is_not_a_filename_is_made_into_one(tmp_path):
    """`Train(ema=...)` publishes its shadow as `model/ema`."""
    store = CheckpointStore(tmp_path)
    written = store.save(
        index=1,
        program=recorder([], "a"),
        bindings={"model/ema": Tiny(), "model-ema": Tiny()},
        topology=Local(device="cpu"),
    )
    names = sorted(path.name for path in written.weights.values())
    assert names == ["model-ema-2.safetensors", "model-ema.safetensors"]
    assert all(path.is_file() for path in written.weights.values())
    assert sorted(store.load("last").bindings) == ["model-ema", "model/ema"]


def test_a_model_without_a_spec_still_gets_its_own_file(tmp_path):
    """Separation does not depend on the model recording anything."""
    store = CheckpointStore(tmp_path)
    written = store.save(
        index=1,
        program=recorder([], "a"),
        bindings={"model": nn.Linear(3, 3)},
        topology=Local(device="cpu"),
    )
    assert written.weights["model"].is_file()
    assert written.specs == {}
    assert read_spec(written.weights["model"]) is None


def test_the_directory_prefix_is_the_stores_to_choose(tmp_path):
    """The number counts units of work, so the name should not claim a unit."""
    assert CheckpointStore(tmp_path).directory_for(3).name.startswith("ckpt_")
    assert CheckpointStore(tmp_path, prefix="epoch-").directory_for(3).name == (
        "epoch-" + CheckpointStore(tmp_path).directory_for(3).name.removeprefix("ckpt_")
    )
    with pytest.raises(ValueError, match="non-empty prefix"):
        CheckpointStore(tmp_path, prefix="")


def test_reading_ignores_the_prefix(tmp_path):
    """A run resumed after the prefix changed must still find its checkpoints."""
    written = CheckpointStore(tmp_path, prefix="epoch-").save(
        index=1, program=recorder([], "a"), bindings={}, topology=Local(device="cpu")
    )
    assert CheckpointStore(tmp_path).checkpoints() == [written.path]
    assert CheckpointStore(tmp_path).resolve("last") == written.path


def test_the_prefix_reaches_the_store_through_the_hook(tmp_path):
    """Otherwise the store keyword would be unreachable from a run."""
    store = tmp_path / "run"
    Runner(
        recorder([], "ab"),
        store=store,
        hooks=[Checkpointer(every=1, unit="advance", prefix="snap-")],
    ).run()
    assert all(p.name.startswith("snap-") for p in store.iterdir())


def test_a_checkpoint_records_the_counters_of_what_triggered_it(tmp_path):
    """The boundary event carries the program's counters, not the stage's."""
    store = tmp_path / "run"
    program = Program([Epochs("train", epochs=3), Call("after", fn=lambda ctx: None)])
    Runner(
        program, store=store, hooks=[Checkpointer(every=1, unit="epoch", at_end=False)]
    ).run()
    epochs = [
        c.progress.epoch
        for c in map(CheckpointStore(store).load, CheckpointStore(store).checkpoints())
    ]
    assert epochs == [1, 2, 3]


def test_the_layout_names_are_the_stores_to_choose(tmp_path):
    """Directory prefix and both file names are the store's, not the module's."""
    store = CheckpointStore(
        tmp_path, prefix="snap_", manifest="meta.json", state_key="run"
    )
    written = store.save(
        index=1, program=recorder([], "a"), bindings={}, topology=Local(device="cpu")
    )
    assert written.path.name == store.directory_for(1).name
    assert written.path.name.startswith("snap_")
    assert sorted(p.name for p in written.path.iterdir()) == [
        "meta.json",
        "run.safetensors",
    ]
    assert store.load("last").index == 1


@pytest.mark.parametrize(
    "kwargs",
    [{"prefix": ""}, {"manifest": ""}, {"state_key": ""}],
    ids=lambda k: next(iter(k)),
)
def test_an_empty_layout_name_is_refused(tmp_path, kwargs):
    """Each would break the store in its own quiet way."""
    with pytest.raises(ValueError, match="non-empty"):
        CheckpointStore(tmp_path, **kwargs)


def test_build_and_load_model_agree(tmp_path):
    """Two entry points to the same thing: from a loaded checkpoint, or a path."""
    store = CheckpointStore(tmp_path)
    model = Tiny(width=6)
    store.save(
        index=1,
        program=recorder([], "a"),
        bindings={"model": model},
        topology=Local(device="cpu"),
    )
    checkpoint = store.load("last")
    from_state = checkpoint.build("model")
    from_path = load_model(checkpoint.weights["model"])
    assert type(from_state) is type(from_path)
    assert torch.equal(from_state.lin.weight, from_path.lin.weight)


def test_build_takes_strict_like_load_model(tmp_path):
    """The same knob, since it is the same load_state_dict underneath."""
    store = CheckpointStore(tmp_path)
    store.save(
        index=1,
        program=recorder([], "a"),
        bindings={"model": Tiny()},
        topology=Local(device="cpu"),
    )
    checkpoint = store.load("last")
    checkpoint.bindings["model"]["stray"] = torch.ones(2)
    with pytest.raises(RuntimeError, match="(?i)unexpected key"):
        checkpoint.build("model")
    assert checkpoint.build("model", strict=False) is not None


def test_a_checkpoint_records_what_triggered_it(tmp_path):
    """The counters alone cannot say which stage reported them."""
    store = tmp_path / "run"
    program = Program([Epochs("pretrain", epochs=2), Epochs("finetune", epochs=2)])
    Runner(
        program, store=store, hooks=[Checkpointer(every=1, unit="epoch", at_end=False)]
    ).run()

    read = CheckpointStore(store)
    taken = [read.load(p) for p in read.checkpoints()]
    assert [c.progress.epoch for c in taken] == [1, 2, 1, 2]
    assert [c.at for c in taken] == [
        "program/0:pretrain",
        "program/0:pretrain",
        "program/1:finetune",
        "program/1:finetune",
    ]
    assert {c.unit for c in taken} == {"epoch"}


def test_the_index_stays_unique_where_the_epoch_does_not(tmp_path):
    """Two stages both report epoch 1, so the epoch cannot name a directory."""
    store = tmp_path / "run"
    Runner(
        Program([Epochs("pretrain", epochs=2), Epochs("finetune", epochs=2)]),
        store=store,
        hooks=[Checkpointer(every=1, unit="epoch", at_end=False)],
    ).run()
    read = CheckpointStore(store)
    indices = [read.load(p).index for p in read.checkpoints()]
    assert len(set(indices)) == len(indices)
    assert indices == sorted(indices)


def test_the_unit_and_trigger_are_absent_when_nobody_says(tmp_path):
    """A direct save need not know about units at all."""
    store = CheckpointStore(tmp_path)
    store.save(
        index=1, program=recorder([], "a"), bindings={}, topology=Local(device="cpu")
    )
    checkpoint = store.load("last")
    assert checkpoint.unit is None
    assert checkpoint.at is None


def written_at(store, indices):
    """Write one checkpoint per index and return the store.

    Args:
        store: Where to write.
        indices: Indices to write at.
    """
    for index in indices:
        store.save(
            index=index,
            program=recorder([], "a"),
            bindings={},
            topology=Local(device="cpu"),
        )
    return store


def test_first_and_last_pick_the_ends(tmp_path):
    """The oldest kept checkpoint and the newest."""
    store = written_at(CheckpointStore(tmp_path), (2, 5, 9))
    assert store.resolve("first") == store.directory_for(2)
    assert store.resolve("last") == store.directory_for(9)


@pytest.mark.parametrize(
    ("name", "index"), [("last~0", 9), ("last~1", 5), ("last~2", 2)]
)
def test_counting_back_from_the_last(tmp_path, name, index):
    """A run that diverged resumes from before the damage."""
    store = written_at(CheckpointStore(tmp_path), (2, 5, 9))
    assert store.resolve(name) == store.directory_for(index)


def test_counting_back_too_far_says_how_many_there_are(tmp_path):
    """Better than silently landing on the oldest."""
    store = written_at(CheckpointStore(tmp_path), (2, 5))
    with pytest.raises(C3liCheckpointError, match="holds only 2"):
        store.resolve("last~2")


def test_an_index_addresses_one_checkpoint(tmp_path):
    """The counterpart to directory_for, and unambiguous because it is an int."""
    store = written_at(CheckpointStore(tmp_path), (2, 5, 9))
    assert store.resolve(5) == store.directory_for(5)
    assert store.load(5).index == 5


def test_an_index_that_was_never_kept_lists_the_ones_that_were(tmp_path):
    """Pruning makes this easy to hit."""
    store = written_at(CheckpointStore(tmp_path), (2, 5))
    with pytest.raises(C3liCheckpointError, match=r"Kept: \[2, 5\]"):
        store.resolve(7)


def test_a_positional_name_needs_something_to_point_at(tmp_path):
    """An empty store cannot answer any of them."""
    store = CheckpointStore(tmp_path)
    for name in ("first", "last", "last~1"):
        with pytest.raises(C3liCheckpointError, match="No complete checkpoint"):
            store.resolve(name)


def test_a_run_resumes_from_an_earlier_checkpoint(tmp_path):
    """The whole point of counting back: skip what went wrong."""
    store = tmp_path / "run"
    first: list[str] = []
    Runner(
        recorder(first),
        store=store,
        hooks=[Checkpointer(every=1, unit="advance")],
    ).run()

    again: list[str] = []
    Runner(recorder(again), store=store, resume="last~2", hooks=[]).run()
    assert again == ["c", "d"]


def test_directory_for_does_not_create_by_default(tmp_path):
    """A lookup must not leave an empty directory behind."""
    store = CheckpointStore(tmp_path)
    directory = store.directory_for(3)
    assert not directory.exists()
    with pytest.raises(C3liCheckpointError, match="No checkpoint at index 3"):
        store.resolve(3)
    assert not directory.exists()


def test_directory_for_creates_on_request(tmp_path):
    """Saving wants the directory there, parents included."""
    store = CheckpointStore(tmp_path / "deep" / "run")
    directory = store.directory_for(3, create=True)
    assert directory.is_dir()
    assert store.directory_for(3, create=True) == directory


def weight_files(tmp_path, names):
    """Return the filenames a set of model bindings is written to.

    Args:
        tmp_path: Where the store lives.
        names: Binding names to save a small model under.
    """
    store = CheckpointStore(tmp_path)
    written = store.save(
        index=1,
        program=recorder([], "a"),
        bindings={name: Tiny() for name in names},
        topology=Local(device="cpu"),
    )
    return {name: path.name for name, path in written.weights.items()}


def test_a_long_binding_name_is_truncated_not_fatal(tmp_path):
    """The filesystem's limit must not end a training run."""
    files = weight_files(tmp_path, ["x" * 300])
    assert len(files["x" * 300]) <= 120
    assert CheckpointStore(tmp_path).load("last").bindings["x" * 300]


def test_names_differing_only_in_case_get_separate_files(tmp_path):
    """They would be one file on a case-insensitive filesystem."""
    files = weight_files(tmp_path, ["Model", "model"])
    assert len({name.lower() for name in files.values()}) == 2


def test_a_name_with_nothing_usable_left_falls_back(tmp_path):
    """Sanitizing must not produce a hidden file or a row of dashes."""
    files = weight_files(tmp_path, [".", "..", "模型"])
    assert all(not name.startswith(".") for name in files.values())
    assert len(set(files.values())) == 3


def test_a_binding_name_cannot_escape_the_checkpoint(tmp_path):
    """Separators are sanitized away, so the file stays put."""
    store = CheckpointStore(tmp_path)
    written = store.save(
        index=1,
        program=recorder([], "a"),
        bindings={"../../etc/passwd": Tiny()},
        topology=Local(device="cpu"),
    )
    path = written.weights["../../etc/passwd"]
    assert path.parent == written.path
    assert path.is_file()


def test_a_checkpointer_can_be_told_where_to_write(tmp_path):
    """Its own store wins over the run's, which stays untouched.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    elsewhere = tmp_path / "elsewhere"
    Runner(
        recorder([]),
        store=tmp_path / "store",
        hooks=[Checkpointer(every=1, unit="advance", store=elsewhere)],
    ).run()
    assert sorted(p.name for p in elsewhere.glob("ckpt_*"))
    assert not sorted((tmp_path / "store").glob("ckpt_*"))


def test_a_checkpointer_told_where_to_go_needs_no_run_store(tmp_path):
    """Only a hook relying on the run's store forces one to be given.

    Args:
        tmp_path: Directory pytest gives the test.
    """
    Runner(
        recorder([]), hooks=[Checkpointer(every=1, unit="advance", store=tmp_path)]
    ).run()
    assert sorted(p.name for p in tmp_path.glob("ckpt_*"))


def test_an_epoch_interval_counts_only_training_passes(tmp_path):
    """An inference pass is not an epoch, or the interval silently shortens."""
    import torch.nn as nn
    from torch.utils.data import TensorDataset
    from chuchichaestli.programs.stages import Phase, Predict, Train

    seen = []

    class Watch:
        """Note every checkpoint event."""

        def on(self, event):
            """Record checkpoint events.

            Args:
                event: What the runner just did.
            """
            if event.type is EventType.CHECKPOINT:
                seen.append(event.payload["path"])
            return Signal.GO

    rows = TensorDataset(torch.rand(4, 2), torch.rand(4, 1))
    fit = Train("fit", data=rows, batch_size=4, loss=nn.MSELoss())
    Runner(
        Program(
            provide={"model": nn.Linear(2, 1)},
            stages=[
                Phase.each_pass(
                    4,
                    fit,
                    Predict("sample", data=rows, batch_size=4, inputs="x", targets="y"),
                )
            ],
        ),
        store=tmp_path / "run",
        hooks=[Checkpointer(every=2, unit="epoch", at_end=False), Watch()],
    ).run()
    assert len(seen) == 2


def test_a_loop_keeps_its_step_count_across_a_resume(tmp_path):
    """The one counter that says how much work a stage did, in all."""
    import torch.nn as nn
    from torch.utils.data import TensorDataset
    from chuchichaestli.programs.stages import Phase, Train

    rows = TensorDataset(torch.rand(4, 2), torch.rand(4, 1))

    def program(fit):
        return Program(
            provide={"model": nn.Linear(2, 1)},
            stages=[Phase.each_pass(4, fit, name="pass")],
        )

    store = tmp_path / "run"
    stopped = Train("fit", data=rows, batch_size=2, loss=nn.MSELoss())
    Runner(
        program(stopped),
        store=store,
        hooks=[Checkpointer(every=1, unit="advance"), StopAfter(5)],
    ).run()

    carried = Train("fit", data=rows, batch_size=2, loss=nn.MSELoss())
    Runner(program(carried), store=store, resume="last", hooks=[]).run()

    whole = Train("fit", data=rows, batch_size=2, loss=nn.MSELoss())
    Runner(program(whole), hooks=[]).run()
    assert carried.total_steps == whole.total_steps


def test_a_resume_refuses_a_checkpoint_seeded_differently(tmp_path):
    """Derived randomness diverges, so the bit-for-bit guarantee would not hold."""
    import torch.nn as nn
    from torch.utils.data import TensorDataset
    from chuchichaestli.programs.stages import Train

    rows = TensorDataset(torch.rand(16, 2), torch.rand(16, 1))

    def program():
        return Program(
            provide={"model": nn.Linear(2, 1)},
            stages=[Train("fit", data=rows, batch_size=4, epochs=4, loss=nn.MSELoss())],
        )

    store = tmp_path / "run"
    Runner(
        program(),
        seed=42,
        store=store,
        hooks=[Checkpointer(every=1, unit="epoch")],
    ).run()
    Runner(program(), seed=42, store=store, resume="last~2", hooks=[]).run()
    with pytest.raises(C3liCheckpointError, match="seeded with 42"):
        Runner(program(), seed=7, store=store, resume="last~2", hooks=[]).run()


def test_a_model_given_to_a_stage_is_checkpointed(tmp_path):
    """Otherwise a resume silently carries on from fresh weights."""
    import torch.nn as nn
    from torch.utils.data import TensorDataset
    from chuchichaestli.programs.stages import Train

    rows = TensorDataset(torch.rand(16, 2), torch.rand(16, 1))

    def stage():
        torch.manual_seed(0)
        return Train(
            "fit",
            model=nn.Linear(2, 1),
            data=rows,
            batch_size=4,
            epochs=4,
            loss=nn.MSELoss(),
        )

    store = tmp_path / "run"
    whole = stage()
    Runner(
        Program([whole]),
        seed=1,
        store=store,
        hooks=[Checkpointer(every=1, unit="epoch")],
    ).run()
    carried = stage()
    Runner(Program([carried]), seed=1, store=store, resume="last~2", hooks=[]).run()
    expected = whole.model.state_dict()
    actual = carried.model.state_dict()
    assert all(torch.equal(v, actual[k]) for k, v in expected.items())


def nested_program(tmp_path):
    """Build a program whose trained model a nested phase provides.

    Args:
        tmp_path: Directory the run writes under.
    """
    from torch.utils.data import TensorDataset
    from chuchichaestli.programs import DataManager, Phase, Train

    x = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    inner, outer = nn.Linear(3, 1, bias=False), nn.Linear(3, 1, bias=False)
    stage = Train(
        "fit",
        model="inner",
        data=DataManager(TensorDataset(x, torch.zeros(8, 1)), batch_size=4),
        loss=nn.MSELoss(),
        epochs=4,
        lr=0.5,
    )
    program = Program(
        provide={"outer": outer},
        stages=[Phase("nested", [stage], provide={"inner": inner})],
    )
    return program, inner


def test_a_nested_phase_provides_state_that_is_checkpointed(tmp_path):
    """Bindings are recorded run-wide, not only those the root can see."""
    store = tmp_path / "run"
    program, _ = nested_program(tmp_path)
    Runner(program, store=store, hooks=[Checkpointer(every=1)], device="cpu").run()
    recorded = sorted(CheckpointStore(store).load("last").bindings)
    assert recorded == ["inner", "outer"]


def test_a_nested_phase_provides_state_that_is_restored(tmp_path):
    """Weights the root cannot see would otherwise resume untrained."""
    store = tmp_path / "run"
    torch.manual_seed(0)
    whole, trained = nested_program(tmp_path)
    Runner(whole, store=store, hooks=[Checkpointer(every=1)], device="cpu").run()

    torch.manual_seed(0)
    again, resumed = nested_program(tmp_path)
    Runner(again, store=store, resume="last~2", hooks=[], device="cpu").run()
    assert torch.allclose(trained.weight, resumed.weight, atol=1e-6)


def test_state_nothing_is_bound_for_is_reported(tmp_path):
    """A renamed binding would otherwise resume untrained in silence."""
    store = tmp_path / "run"
    program, _ = nested_program(tmp_path)
    Runner(program, store=store, hooks=[Checkpointer(every=1)], device="cpu").run()

    renamed, _ = nested_program(tmp_path)
    renamed.provide = {"elsewhere": renamed.provide["outer"]}
    with pytest.warns(UserWarning, match="Nothing in this run was bound for"):
        Runner(renamed, store=store, resume="last", hooks=[], device="cpu").run()


def test_a_checkpoint_holds_one_file_per_bound_model(tmp_path):
    """A published average is the stage's to record, not a file of its own."""
    from torch.utils.data import TensorDataset
    from chuchichaestli.programs import DataManager, Phase, Train

    store = tmp_path / "run"
    x = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    model = nn.Linear(3, 1, bias=False)
    training = Train(
        "train",
        model="model",
        data=DataManager(TensorDataset(x, torch.zeros(8, 1)), batch_size=4),
        loss=nn.MSELoss(),
        epochs=4,
        lr=0.1,
        ema=0.9,
    )
    program = Program(provide={"model": model}, stages=[Phase.each_pass(4, training)])
    Runner(
        program,
        store=store,
        hooks=[Checkpointer(every=2, unit="epoch")],
        device="cpu",
    ).run()
    written = sorted(p.name for p in sorted(store.glob("ckpt_*"))[-1].iterdir())
    assert written == ["manifest.json", "model.safetensors", "state.safetensors"]


def test_two_artifacts_under_one_name_are_reported(tmp_path):
    """Recording only the later one would lose the other without a word."""
    from chuchichaestli.programs import Phase

    store = tmp_path / "run"
    program = Program(
        provide={"model": nn.Linear(3, 1)},
        stages=[
            Phase(
                "inner",
                [Call("a", fn=lambda ctx: None)],
                provide={"model": nn.Linear(3, 1)},
            )
        ],
    )
    with pytest.warns(UserWarning, match="Two artifacts are bound as 'model'"):
        Runner(
            program,
            store=store,
            hooks=[Checkpointer(every=1, unit="advance")],
            device="cpu",
        ).run()


def carrying_program(epochs: int = 6):
    """Build an each_pass program whose optimizer carries momentum.

    Args:
        epochs: Passes the loop makes in all.
    """
    from torch.utils.data import TensorDataset
    from chuchichaestli.programs import DataManager, Phase, Train
    from chuchichaestli.training import OptimSpec

    x = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    model = nn.Linear(3, 1, bias=False)
    stage = Train(
        "train",
        model="model",
        data=DataManager(TensorDataset(x, torch.zeros(8, 1)), batch_size=4),
        loss=nn.MSELoss(),
        epochs=epochs,
        optim=OptimSpec.sgd(lr=0.01, momentum=0.9),
        ema=0.9,
    )
    program = Program(provide={"model": model}, stages=[Phase.each_pass(epochs, stage)])
    return program, stage, model


def momentum_of(stage):
    """Return the optimizer's momentum buffers.

    Args:
        stage: The training stage to inspect.
    """
    return [
        s["momentum_buffer"]
        for s in stage.update.optimizers[None].state.values()
        if s.get("momentum_buffer") is not None
    ]


def test_what_a_stage_learned_is_recorded_apart_from_its_counters(tmp_path):
    """A phase keeps a child's state only while it is open, which a boundary is not."""
    store = tmp_path / "run"
    program, stage, _ = carrying_program()
    Runner(
        program, store=store, hooks=[Checkpointer(every=1, unit="epoch")], device="cpu"
    ).run()
    components = CheckpointStore(store).load("last").components
    assert sorted(components) == ["program/0:repeat/0:epoch/0:train"]
    assert "optim" in components["program/0:repeat/0:epoch/0:train"]
    assert "progress" not in components["program/0:repeat/0:epoch/0:train"]


def test_a_resumed_pass_at_a_time_run_keeps_its_optimizer(tmp_path):
    """Momentum reset to zero makes a resumed run diverge from an unbroken one."""
    store = tmp_path / "run"
    torch.manual_seed(0)
    whole, trained, weights = carrying_program()
    Runner(
        whole,
        seed=5,
        store=store,
        hooks=[Checkpointer(every=1, unit="epoch")],
        device="cpu",
    ).run()

    torch.manual_seed(0)
    again, resumed, revived = carrying_program()
    Runner(again, seed=5, store=store, resume="last~3", hooks=[], device="cpu").run()

    assert momentum_of(resumed)
    for before, after in zip(momentum_of(trained), momentum_of(resumed)):
        assert torch.allclose(before, after, atol=1e-6)
    assert torch.allclose(weights.weight, revived.weight, atol=1e-6)


def test_a_resumed_pass_at_a_time_run_keeps_its_average(tmp_path):
    """The average is published, so only the stage itself records it."""
    store = tmp_path / "run"
    torch.manual_seed(0)
    whole, trained, _ = carrying_program()
    Runner(
        whole,
        seed=5,
        store=store,
        hooks=[Checkpointer(every=1, unit="epoch")],
        device="cpu",
    ).run()

    torch.manual_seed(0)
    again, resumed, _ = carrying_program()
    Runner(again, seed=5, store=store, resume="last~3", hooks=[], device="cpu").run()
    assert torch.allclose(
        trained._ema["model"].module.weight,
        resumed._ema["model"].module.weight,
        atol=1e-6,
    )


def test_a_repeated_stage_is_recorded_once(tmp_path):
    """Every slot of a repeat holds one object, whose state is not a copy each."""
    from chuchichaestli.programs.serialize import stages_by_path
    from chuchichaestli.programs import Repeat

    inner = Call("a", fn=lambda ctx: None)
    program = Program([Repeat(4, inner)])
    found = stages_by_path(program)
    assert list(found).count("program/0:repeat/0:a") == 1
    assert sum(1 for v in found.values() if v is inner) == 1


def stage_held(name: str):
    """Build a training stage that holds its own model.

    Args:
        name: Identifies the stage.
    """
    from torch.utils.data import TensorDataset
    from chuchichaestli.programs import DataManager, Train

    x = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    return Train(
        name,
        model=nn.Linear(3, 1),
        data=DataManager(TensorDataset(x, torch.zeros(8, 1)), batch_size=4),
        loss=nn.MSELoss(),
        epochs=2,
        lr=0.1,
    )


def weight_files_in(store):
    """Return the weight files of a store's latest checkpoint.

    Args:
        store: Directory the run wrote under.
    """
    latest = sorted(store.glob("ckpt_*"))[-1]
    return sorted(p.name for p in latest.glob("*.safetensors") if p.stem != "state")


def test_a_model_bound_twice_is_written_once(tmp_path):
    """A stage binds the model it holds, which the run also records by path."""
    store = tmp_path / "run"
    Runner(
        Program([stage_held("fit")]),
        store=store,
        hooks=[Checkpointer(every=1)],
        device="cpu",
    ).run()
    assert weight_files_in(store) == ["program-0-fit-model.safetensors"]


def test_two_stages_holding_models_keep_them_apart(tmp_path):
    """Each binds its own as `model`, so the name alone cannot tell them apart."""
    store = tmp_path / "run"
    Runner(
        Program([stage_held("a"), stage_held("b")]),
        store=store,
        hooks=[Checkpointer(every=1)],
        device="cpu",
    ).run()
    assert weight_files_in(store) == [
        "program-0-a-model.safetensors",
        "program-1-b-model.safetensors",
    ]


def run_names(trace) -> list[str]:
    """Return the name each run in a trace opened with.

    Args:
        trace: JSONL file the `Jsonl` hook appended to.
    """
    return [
        e.payload["run_name"] for e in Jsonl.read(trace, only=[EventType.RUN_BEGAN])
    ]


def test_a_checkpoint_records_the_name_of_its_run(tmp_path):
    """The name is how a resumed run finds itself again."""
    store = tmp_path / "run"
    Runner(
        recorder([]),
        store=store,
        run_name="baseline",
        hooks=[Checkpointer(every=1, unit="advance")],
    ).run()
    written = CheckpointStore(store)
    assert {written.load(p).run_name for p in written.checkpoints()} == {"baseline"}
    assert written.read_manifest("last")["run_name"] == "baseline"
    assert written.run_names() == {"baseline"}


def test_a_resumed_run_keeps_its_name(tmp_path):
    """Its points belong in the same tracker run as before the interruption."""
    store = tmp_path / "run"
    Runner(
        recorder([]),
        store=store,
        hooks=[Checkpointer(every=1, unit="advance"), Jsonl(), StopAfter(2)],
    ).run()
    Runner(recorder([]), store=store, resume="last", hooks=[Jsonl()]).run()
    first, second = run_names(store / "trace.jsonl")
    assert first == second


def test_a_given_name_overrides_the_resumed_one(tmp_path):
    """An explicit name is used as given, even on a resume."""
    store = tmp_path / "run"
    Runner(
        recorder([]),
        store=store,
        hooks=[Checkpointer(every=1, unit="advance"), StopAfter(2)],
    ).run()
    Runner(
        recorder([]), store=store, resume="last", run_name="renamed", hooks=[Jsonl()]
    ).run()
    assert run_names(store / "trace.jsonl") == ["renamed"]


def test_an_unreadable_manifest_is_reported(tmp_path):
    """A truncated write must not pass for a checkpoint without a name."""
    store = CheckpointStore(tmp_path)
    written = store.save(
        index=1, program=recorder([], "a"), bindings={}, topology=Local(device="cpu")
    )
    (written.path / store.manifest).write_text("{")
    with pytest.raises(C3liCheckpointError, match="No checkpoint manifest|unreadable"):
        store.read_manifest(written.path)


def test_a_checkpoint_records_the_run_counts(tmp_path):
    """A resume needs them to number its exports and steps where it stopped."""
    store = CheckpointStore(tmp_path)
    written = store.save(
        index=1,
        program=recorder([], "a"),
        bindings={},
        topology=Local(device="cpu"),
        counts={"advance": 4, "epoch": 2, "step": 8},
    )
    assert store.load(written.path).counts == {"advance": 4, "epoch": 2, "step": 8}
