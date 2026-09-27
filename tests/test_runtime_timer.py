# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for reporting how long a run's stages and passes took."""

import io
import re

import torch
from torch import nn
from torch.utils.data import TensorDataset

from chuchichaestli.runtime import (
    Call,
    Phase,
    Console,
    Program,
    Repeat,
    Runtime,
    Timer,
    Train,
)
from chuchichaestli.training import OptimSpec


def ramp(n: int = 8) -> TensorDataset:
    """Build a dataset whose inputs count upwards.

    Args:
        n: Number of samples.
    """
    x = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) / 10
    return TensorDataset(x, x[:, :1])


def trainer(name: str = "fit", epochs: int = 3) -> Train:
    """Build a loop that makes a known number of passes.

    Args:
        name: Identifies the stage.
        epochs: Passes over the data to make.
    """
    return Train(
        name,
        model=nn.Linear(3, 1, bias=False),
        data=ramp(),
        batch_size=4,
        epochs=epochs,
        loss=nn.MSELoss(),
        optim=OptimSpec.sgd(lr=0.01),
    )


def report_of(program: Program) -> tuple[str, Timer]:
    """Run a program under a timer and return what it printed.

    Args:
        program: What to run.
    """
    stream = io.StringIO()
    timer = Timer(stream=stream)
    Runtime(program, hooks=[timer], device="cpu").run()
    return stream.getvalue(), timer


def test_a_stage_reports_what_it_took():
    """The summary names every stage the run entered."""
    report, timer = report_of(Program(stages=[trainer()]))
    assert "program/0:fit" in report
    assert timer.elapsed["program/0:fit"] > 0


def test_a_loop_reports_what_its_passes_took():
    """Per-pass timing is what says whether they are steady or drifting."""
    report, timer = report_of(Program(stages=[trainer(epochs=3)]))
    assert len(timer.passes["program/0:fit"]) == 3
    assert re.search(r"3 passes, mean \d+\.\d+s, min \d+\.\d+s, max \d+\.\d+s", report)


def test_a_stage_that_makes_no_passes_reports_none():
    """A block runs once, so there is nothing per-pass to say about it."""
    report, timer = report_of(Program(stages=[Call(fn=lambda ctx: None, name="once")]))
    assert "program/0:once" in report
    assert "passes" not in report
    assert "program/0:once" not in timer.passes


def test_each_repetition_is_timed_on_its_own():
    """`Repeat` indexes every visit, so the two are told apart."""
    program = Program(stages=[Repeat(2, trainer(epochs=3), name="cycle")])
    report, timer = report_of(program)
    visits = ["program/0:cycle/0:fit", "program/0:cycle/1:fit"]
    assert sorted(path for path in timer.passes if ":fit" in path) == visits
    assert all(len(timer.passes[path]) == 3 for path in visits)
    assert report.count("3 passes") == 2


def test_the_console_can_report_the_timing_itself():
    """One hook, one stream: the reporter measures through a timer of its own."""
    stream = io.StringIO()
    Runtime(
        Program(stages=[trainer(epochs=2)]),
        hooks=[Console(every=1, stream=stream, color=False, timing=True)],
        device="cpu",
    ).run()
    report = stream.getvalue()
    assert "pass 0 in" in report
    assert "pass 1 in" in report


def test_the_console_times_the_passes_it_reports():
    """`every` picks which pass is reported, as it does for steps."""
    stream = io.StringIO()
    Runtime(
        Program(stages=[trainer(epochs=4)]),
        hooks=[Console(every=2, stream=stream, color=False, timing=True)],
        device="cpu",
    ).run()
    reported = re.findall(r"pass (\d+) in", stream.getvalue())
    assert reported == ["0", "2"]


def test_the_console_leaves_the_totals_to_a_timer():
    """A summary is a `Timer` hook's to give, so adding one is the way to it."""
    alone = io.StringIO()
    Runtime(
        Program(stages=[trainer(epochs=2)]),
        hooks=[Console(every=1, stream=alone, color=False, timing=True)],
        device="cpu",
    ).run()
    assert "passes, mean" not in alone.getvalue()

    both = io.StringIO()
    Runtime(
        Program(stages=[trainer(epochs=2)]),
        hooks=[
            Console(every=1, stream=both, color=False, timing=True),
            Timer(stream=both),
        ],
        device="cpu",
    ).run()
    assert re.search(r"2 passes, mean \d+\.\d+s", both.getvalue())


def test_the_console_says_nothing_about_timing_unless_asked():
    """The reporter's default output is unchanged by the timer existing."""
    stream = io.StringIO()
    Runtime(
        Program(stages=[trainer(epochs=2)]),
        hooks=[Console(every=1, stream=stream, color=False)],
        device="cpu",
    ).run()
    report = stream.getvalue()
    assert "step" in report
    assert "pass" not in report
    assert "passes" not in report


def test_a_quiet_timer_measures_without_printing():
    """That is how the reporter borrows one without it writing too."""
    stream = io.StringIO()
    timer = Timer(stream=stream, report=False)
    Runtime(Program(stages=[trainer(epochs=2)]), hooks=[timer], device="cpu").run()
    assert stream.getvalue() == ""
    assert len(timer.passes["program/0:fit"]) == 2
    assert timer.summary()


def test_a_timer_reports_only_as_deep_as_it_was_asked():
    """A regimen makes every visit a stage, so a full summary is unreadable."""
    program = Program(stages=[Phase.each_pass(4, trainer(epochs=1))])

    def lines(depth):
        """Return what a timer of a given depth wrote.

        Args:
            depth: How far below the root to report.
        """
        stream = io.StringIO()
        Runtime(program, hooks=[Timer(stream=stream, depth=depth)], device="cpu").run()
        return stream.getvalue().splitlines()

    assert len(lines(0)) == 1
    assert len(lines(1)) == 2
    assert len(lines(None)) > 4
