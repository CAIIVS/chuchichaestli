# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the progress bar held at the foot of the stream."""

import io
import re

import pytest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from chuchichaestli.metrics import MSE
from chuchichaestli.runtime import (
    Eval,
    Phase,
    EventType,
    Console,
    Program,
    Runtime,
    ProgressBar,
    Signal,
    Train,
)
from chuchichaestli.training import OptimSpec
from chuchichaestli.utils.ansi import Pinned


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


def test_a_loop_says_how_many_steps_it_will_take():
    """The bar needs a total, and only the loop knows what it is."""
    totals = []

    class Watching:
        """Records the pass length each step reports."""

        def on(self, event):
            """Note the total a step carried.

            Args:
                event: What the runtime just did.
            """
            if event.type is EventType.STEP_ENDED:
                totals.append(event.payload.get("total"))
            return Signal.GO

    Runtime(Program(stages=[trainer(epochs=2)]), hooks=[Watching()], device="cpu").run()
    assert totals == [4] * 4


def test_the_console_draws_no_bar_unless_asked():
    """It is opt-in, so an ordinary run's lines are unchanged."""
    stream = io.StringIO()
    Runtime(
        Program(stages=[trainer(epochs=2)]),
        hooks=[Console(every=1, stream=stream, color=False)],
        device="cpu",
    ).run()
    assert "[" not in stream.getvalue()
    assert "total=" not in stream.getvalue()


def test_the_bar_fills_across_the_whole_stage():
    """The step number says where in the run it is, so the bar must agree."""
    stream = io.StringIO()
    Runtime(
        Program(stages=[trainer(epochs=2)]),
        hooks=[ProgressBar(stream=stream, live=True)],
        device="cpu",
    ).run()
    bars = re.findall(r"\[([#-]*)\]", stream.getvalue())
    assert {len(bar) for bar in bars} == {24}
    assert [bar.count("#") for bar in bars] == [6, 12, 18, 24]


def test_a_console_line_scrolls_above_the_bar():
    """Both write one cursor, so a line must not be drawn through the bar."""
    stream = io.StringIO()
    console = Console(every=1, stream=stream, color=False)
    Runtime(
        Program(stages=[trainer(epochs=1)]),
        hooks=[console, ProgressBar(stream=stream, live=True, bar_length=8)],
        device="cpu",
    ).run()
    assert console.pinned is not None
    written = stream.getvalue()
    assert f"{Pinned.ERASE}  program/0:fit step 1" in written
    assert f"{Pinned.ERASE}  program/0:fit step 2" in written
    assert "\n  program/0:fit [" in written


def test_the_bar_stays_out_of_a_file():
    """A log keeps every line it is given, so a redrawn one would fill it."""
    stream = io.StringIO()
    Runtime(
        Program(stages=[trainer(epochs=2)]),
        hooks=[ProgressBar(stream=stream, live=False)],
        device="cpu",
    ).run()
    assert stream.getvalue() == ""


def test_the_bar_stays_put_across_stage_boundaries():
    """A regimen re-entering a stage ends stages constantly; the bar stays."""
    stream = io.StringIO()
    Runtime(
        Program(stages=[Phase.each_pass(3, trainer(epochs=1))]),
        hooks=[
            Console(every=1, stream=stream, color=False),
            ProgressBar(stream=stream, live=True, bar_length=8),
        ],
        device="cpu",
    ).run()
    written = stream.getvalue()
    drawn = len(re.findall(r"\[[#-]{8}\]", written))
    assert drawn >= written.count("\n")
    assert written.endswith(Pinned.ERASE)


def test_the_bar_can_span_the_passes_instead_of_the_steps():
    """A visit of two steps can only ever read half, then whole."""
    by_unit = {}
    for unit in ("step", "epoch"):
        stream = io.StringIO()
        Runtime(
            Program(stages=[Phase.each_pass(4, trainer(epochs=1))]),
            hooks=[ProgressBar(unit=unit, stream=stream, live=True, bar_length=8)],
            device="cpu",
        ).run()
        by_unit[unit] = re.findall(r"\] (\d+/\d+)", stream.getvalue())
    assert by_unit["step"] == ["1/2", "2/2"] * 4
    assert by_unit["epoch"] == ["1/4", "2/4", "3/4", "4/4"]


def test_a_bar_unit_names_the_alternatives():
    """The unit picks which event is counted, so a bad one says which do."""
    with pytest.raises(ValueError, match="bar unit"):
        ProgressBar(unit="fortnight")


def test_other_stages_do_not_push_the_count_past_its_total():
    """An eval passes over data too, but it is not what the bar measures."""
    stream = io.StringIO()
    scoring = Eval(
        "score",
        model=nn.Linear(3, 1),
        data=ramp(),
        batch_size=8,
        metrics=[MSE()],
    )
    Runtime(
        Program(stages=[Phase.each_pass(4, trainer(epochs=1), scoring)]),
        hooks=[ProgressBar(unit="epoch", stream=stream, live=True, bar_length=8)],
        device="cpu",
    ).run()
    assert re.findall(r"\] (\d+/\d+)", stream.getvalue()) == [
        "1/4",
        "2/4",
        "3/4",
        "4/4",
    ]
