# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the stage vocabulary: blocks, phases and their composition."""

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from chuchichaestli.runtime.runtime import Runtime
from chuchichaestli.runtime.context import Context
from chuchichaestli.runtime.events import Progress, Signal
from chuchichaestli.runtime.stages import (
    Train,
    Barrier,
    Call,
    Every,
    Exporter,
    WeightsExport,
    ImageExport,
    Load,
    Phase,
    Predict,
    Program,
    Repeat,
    StageBlock,
    When,
)
from chuchichaestli.runtime.traits import Stage


class Counting(StageBlock):
    """A block recording the progress it saw on each run."""

    def __init__(self, name: str = "counting"):
        """Constructor.

        Args:
            name: Identifies the stage within its parent.
        """
        super().__init__(name)
        self.seen: list[int] = []

    def core(self, ctx: Context) -> None:
        """Record the global step as it stood on entry.

        Args:
            ctx: Execution context for this entry.
        """
        self.seen.append(self._progress.global_step)


def drive(stage, ctx=None):
    """Run a stage to completion the way the runtime would.

    Args:
        stage: The stage to drive.
        ctx: Context to use; a fresh root one by default.
    """
    ctx = ctx if ctx is not None else Context("program", seed=1)
    signal = stage.enter(ctx)
    while not signal.halts:
        signal = stage.execute(ctx)
    stage.leave(ctx)
    return ctx


def test_blocks_and_phases_satisfy_the_stage_trait():
    """Structural typing, so a custom stage need not inherit anything."""
    assert isinstance(Call("a", fn=lambda c: None), Stage)
    assert isinstance(Program([]), Stage)
    assert isinstance(Repeat(2, Call("a", fn=lambda c: None)), Stage)


def test_children_run_in_order():
    """A phase is a sequence, and nesting does not disturb that."""
    log: list[str] = []
    program = Program(
        [
            Call("a", fn=lambda c: log.append("a")),
            Phase("mid", [Call("b", fn=lambda c: log.append("b"))]),
            Call("c", fn=lambda c: log.append("c")),
        ]
    )
    drive(program)
    assert log == ["a", "b", "c"]


def test_repeat_runs_its_child_n_times():
    """The same object is entered repeatedly."""
    log: list[str] = []
    drive(Program([Repeat(3, Call("b", fn=lambda c: log.append("b")))]))
    assert log == ["b", "b", "b"]


def test_repeat_resets_child_progress_each_entry():
    """`enter` means reset, which is what makes a stage re-entrant."""
    child = Counting()
    drive(Program([Repeat(3, child)]))
    assert child.seen == [0, 0, 0]


def test_repeat_rejects_a_non_positive_count():
    """A count of zero is a plan error, not an empty loop."""
    with pytest.raises(ValueError, match="positive count"):
        Repeat(0, Call("a", fn=lambda c: None))


@pytest.mark.parametrize(("holds", "expected"), [(True, ["x"]), (False, [])])
def test_when_runs_its_child_only_if_the_predicate_holds(holds, expected):
    """The predicate is evaluated once, on entry."""
    log: list[str] = []
    drive(Program([When(lambda c: holds, Call("x", fn=lambda c: log.append("x")))]))
    assert log == expected


def test_when_reads_published_bindings():
    """A later stage can branch on what an earlier one produced."""
    log: list[str] = []
    program = Program(
        [
            Call("probe", fn=lambda c: {"probe/psnr": 31.0}),
            When(
                lambda c: c["probe/psnr"] > 30,
                Call("sample", fn=lambda c: log.append("sample")),
            ),
        ]
    )
    drive(program)
    assert log == ["sample"]


def test_every_runs_on_each_nth_visit():
    """The visit counter survives re-entry, so `Repeat` gives the cadence."""
    log: list[str] = []
    drive(Program([Repeat(6, Every(2, Call("c", fn=lambda c: log.append("c"))))]))
    assert log == ["c", "c", "c"]


def test_every_rejects_a_non_positive_interval():
    """An interval of zero would never fire."""
    with pytest.raises(ValueError, match="positive interval"):
        Every(0, Call("a", fn=lambda c: None))


def test_call_publishes_a_mapping_entry_by_entry():
    """A returned mapping reaches later siblings by name."""
    ctx = drive(Program([Call("m", fn=lambda c: {"one": 1, "two": 2})]))
    assert ctx["one"] == 1 and ctx["two"] == 2


def test_call_publishes_a_bare_value_under_its_single_provides_name():
    """Declaring one name is enough to name the result."""
    ctx = drive(Program([Call("m", fn=lambda c: 7, provides=("answer",))]))
    assert ctx["answer"] == 7


def test_provide_binds_for_the_whole_subtree():
    """Shared objects are built once and named, not passed down by hand."""
    seen: list[object] = []
    model = object()
    drive(
        Program(
            [Phase("mid", [Call("a", fn=lambda c: seen.append(c["model"]))])],
            provide={"model": model},
        )
    )
    assert seen == [model]


def test_walk_reports_the_tree_shape():
    """The manifest records this so a resume can refuse a changed program."""
    program = Program(
        [Call("a", fn=lambda c: None), Repeat(2, Call("b", fn=lambda c: None))]
    )
    assert program.walk() == [
        ("program", "Program"),
        ("program/0:a", "Call"),
        ("program/1:repeat", "Repeat"),
        ("program/1:repeat/0:b", "Call"),
        ("program/1:repeat/1:b", "Call"),
    ]


def test_phase_state_records_only_the_live_child():
    """Finished children are implied by the index, so `Repeat` stays small."""
    phase = Program([Counting("a"), Counting("b")])
    ctx = Context("program", seed=1)
    phase.enter(ctx)
    phase.execute(ctx)
    state = phase.state_dict()
    assert state["index"] == 1
    assert state["entered"] is False
    assert state["child"] is None


def test_phase_state_round_trips():
    """Position and the live child's progress both survive."""
    phase = Program([Counting("a"), Counting("b"), Counting("c")])
    ctx = Context("program", seed=1)
    phase.enter(ctx)
    phase.execute(ctx)
    state = phase.state_dict()

    revived = Program([Counting("a"), Counting("b"), Counting("c")])
    revived.enter(Context("program", seed=1))
    revived.load_state_dict(state)
    assert revived.state_dict()["index"] == state["index"]
    assert revived.progress() == Progress.from_dict(state["progress"])


def test_barrier_synchronises():
    """A no-op locally, but the call site is the same when distributed."""
    drive(Program([Barrier("sync")]))


def test_load_and_export_round_trip(tmp_path):
    """A binding exported from one run loads into another."""
    path = tmp_path / "weights.safetensors"
    source = nn.Linear(3, 2)
    drive(
        Program(
            [WeightsExport("out", path=path, source="model")], provide={"model": source}
        )
    )
    assert path.is_file()

    target = nn.Linear(3, 2)
    with torch.no_grad():
        target.weight.zero_()
    drive(Program([Load("in", path=path, target="model")], provide={"model": target}))
    assert torch.equal(target.weight, source.weight)


def test_load_says_which_file_is_missing(tmp_path):
    """The error has to name the path, not just fail."""
    program = Program(
        [Load("in", path=tmp_path / "absent.safetensors", target="model")],
        provide={"model": nn.Linear(1, 1)},
    )
    with pytest.raises(FileNotFoundError, match="absent.safetensors"):
        drive(program)


def test_stage_block_finishes_after_one_execute():
    """One unit of work, then done."""
    block = Counting()
    ctx = Context("program", seed=1)
    assert block.enter(ctx) is Signal.GO
    assert block.execute(ctx) is Signal.DONE
    assert block.progress().done


@pytest.mark.parametrize("suffix", [".safetensors", ".pt", ".pth"])
def test_load_and_export_round_trip_every_format(tmp_path, suffix):
    """A checkpoint written from elsewhere should load without converting it."""
    path = tmp_path / f"weights{suffix}"
    source = nn.Linear(3, 2)
    drive(
        Program(
            [WeightsExport("out", path=path, source="model")], provide={"model": source}
        )
    )
    target = nn.Linear(3, 2)
    with torch.no_grad():
        target.weight.zero_()
    drive(Program([Load("in", path=path, target="model")], provide={"model": target}))
    assert torch.equal(target.weight, source.weight)


def test_export_writes_the_format_its_suffix_names(tmp_path):
    """A `.pt` path must not be safetensors wearing the wrong extension."""
    path = tmp_path / "weights.pt"
    drive(
        Program(
            [WeightsExport("out", path=path, source="model")],
            provide={"model": nn.Linear(2, 2)},
        )
    )
    assert set(torch.load(path, weights_only=True)) == {"weight", "bias"}


def test_a_torch_file_loads_without_conversion(tmp_path):
    """The whole point: an existing `.pt` checkpoint is usable as-is."""
    path = tmp_path / "foreign.pt"
    source = nn.Linear(3, 2)
    torch.save(source.state_dict(), path)
    target = nn.Linear(3, 2)
    drive(Program([Load("in", path=path, target="model")], provide={"model": target}))
    assert torch.equal(target.weight, source.weight)


def test_an_exporter_numbers_its_files_rather_than_overwriting(tmp_path):
    """A stage entered several times leaves its own files behind."""
    Runtime(
        Program(
            [
                Repeat(
                    3,
                    ImageExport(
                        "preview",
                        path=tmp_path / "samples_{advance:02d}.png",
                        source="images",
                    ),
                )
            ],
            provide={"images": torch.rand(1, 1, 8, 8)},
        ),
        hooks=[],
    ).run()
    files = sorted(p.name for p in tmp_path.glob("samples_*.png"))
    assert files == ["samples_00_0.png", "samples_01_0.png", "samples_02_0.png"]


def test_an_exporter_without_a_template_reuses_its_paths(tmp_path):
    """Numbering is opt-in; a plain path stays the path it was given."""
    path = tmp_path / "samples.png"
    drive(
        Program(
            [Repeat(2, ImageExport("preview", path=path, source="images"))],
            provide={"images": torch.rand(2, 1, 8, 8)},
        )
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "samples_0.png",
        "samples_1.png",
    ]


def test_an_export_is_numbered_by_training_epochs_alone(tmp_path):
    """An inference pass is not an epoch, so it must not shift the numbering."""
    pytest.importorskip("matplotlib")
    val = TensorDataset(torch.rand(2, 1, 4, 4), torch.rand(2, 1, 4, 4))
    fit = Train(
        "fit",
        data=TensorDataset(torch.rand(4, 1, 4, 4), torch.rand(4, 1, 4, 4)),
        batch_size=4,
        loss=nn.MSELoss(),
    )
    drive(
        Program(
            provide={"model": nn.Conv2d(1, 1, 1)},
            stages=[
                Phase.each_pass(
                    2,
                    fit,
                    Predict("sample", data=val, batch_size=2, inputs="x", targets="y"),
                    ImageExport(
                        "preview",
                        path=tmp_path / "s_{epoch}.png",
                        source="sample/pairs",
                    ),
                )
            ],
        )
    )
    assert sorted({p.name.split("_")[1] for p in tmp_path.glob("*.png")}) == ["1", "2"]


def test_image_export_plots_what_predict_published(tmp_path):
    """The pairs a `Predict` publishes are what the exporter is pointed at."""
    pytest.importorskip("matplotlib")
    pairs = TensorDataset(torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8))
    drive(
        Program(
            [
                ImageExport(
                    "preview",
                    path=tmp_path / "pairs.png",
                    source="sample/pairs",
                    labels=("sampled", "truth"),
                )
            ],
            provide={"sample/pairs": pairs},
        )
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "pairs_sampled_0.png",
        "pairs_sampled_1.png",
        "pairs_truth_0.png",
        "pairs_truth_1.png",
    ]


def test_an_exporter_has_to_say_how_it_writes():
    """`Exporter` is the shared machinery, not a stage in its own right."""
    with pytest.raises(TypeError):
        Exporter("out", path="x.png")


@pytest.mark.parametrize(
    ("stage", "match"),
    [(Load, "Cannot read '.ckpt'"), (WeightsExport, "Cannot write '.ckpt'")],
)
def test_an_unknown_suffix_names_what_is_accepted(tmp_path, stage, match):
    """As elsewhere in the package, the error lists the alternatives."""
    path = tmp_path / "weights.ckpt"
    if stage is Load:
        path.write_bytes(b"not a checkpoint")
        block = Load("in", path=path, target="model")
    else:
        block = WeightsExport("out", path=path, source="model")
    with pytest.raises(ValueError, match=match):
        drive(Program([block], provide={"model": nn.Linear(2, 2)}))


def test_image_export_writes_a_file_per_image(tmp_path):
    """A multi-file export cannot go through a single scratch name."""
    drive(
        Program(
            [
                ImageExport(
                    "preview", path=tmp_path / "s_{advance:02d}.png", source="images"
                )
            ],
            provide={"images": torch.rand(3, 1, 8, 8)},
        )
    )
    files = sorted(p.name for p in tmp_path.glob("*.png"))
    assert files == ["s_00_0.png", "s_00_1.png", "s_00_2.png"]


def test_image_export_rejects_a_suffix_it_cannot_write(tmp_path):
    """The image formats are listed, as the weight formats are."""
    program = Program(
        [ImageExport("preview", path=tmp_path / "s.gif", source="images")],
        provide={"images": torch.rand(4, 1, 8, 8)},
    )
    with pytest.raises(ValueError, match="Unsupported image format"):
        drive(program)


def test_each_pass_slots_a_stage_between_the_passes():
    """The pass count is given once, to the phase, not twice."""
    seen: list[str] = []
    training = Train(
        "fit",
        model=nn.Linear(2, 1),
        data=TensorDataset(torch.rand(4, 2), torch.rand(4, 1)),
        batch_size=4,
        epochs=7,
        loss=nn.MSELoss(),
    )
    after = Call("after", fn=lambda ctx: seen.append("after"))
    program = Program(stages=[Phase.each_pass(3, training, after)])
    Runtime(program, hooks=(), device="cpu").run()
    assert training.epochs == 1
    assert seen == ["after"] * 3


def test_each_pass_refuses_a_loop_counting_in_steps():
    """A step budget cannot be divided a pass at a time."""
    counting = Train(
        "fit",
        model=nn.Linear(2, 1),
        data=TensorDataset(torch.rand(4, 2), torch.rand(4, 1)),
        batch_size=4,
        steps=10,
        loss=nn.MSELoss(),
    )
    with pytest.raises(ValueError, match="counts in steps"):
        Phase.each_pass(3, counting)
