# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for cancelling a run once a budget is spent."""

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from chuchichaestli.programs import (
    CancelAfter,
    Checkpointer,
    EventType,
    Program,
    Runner,
    Signal,
    Train,
)
from chuchichaestli.programs import hooks
from chuchichaestli.programs.events import C3liRunCancelled
from chuchichaestli.training import OptimSpec


class Ended:
    """Keeps the closing event and counts training steps."""

    def __init__(self):
        """Constructor."""
        self.steps = 0
        self.payload: dict = {}

    def on(self, event):
        """Count steps and keep how the run ended.

        Args:
            event: What the runner just did.
        """
        if event.type is EventType.STEP_ENDED:
            self.steps += 1
        if event.type is EventType.RUN_ENDED:
            self.payload = dict(event.payload)
        return Signal.GO


def program(epochs: int = 3) -> Program:
    """Build a small training program, the same every time.

    Args:
        epochs: Passes over eight samples, four steps each.
    """
    torch.manual_seed(0)
    data = TensorDataset(torch.rand(8, 3), torch.rand(8, 1))
    fit = Train(
        "fit",
        data=data,
        batch_size=2,
        epochs=epochs,
        loss=nn.MSELoss(),
        optim=OptimSpec.sgd(lr=0.1),
    )
    return Program([fit], provide={"model": nn.Linear(3, 1)})


def run(prog: Program, *hooks_, **kwargs) -> None:
    """Run a program on the CPU.

    Args:
        prog: What to run.
        hooks_: The run's hooks.
        kwargs: Passed on to `Runner`.
    """
    Runner(prog, hooks=list(hooks_), device="cpu", seed=3, **kwargs).run()


def weights(prog: Program) -> torch.Tensor:
    """Return the trained weights of a program's model.

    Args:
        prog: The program run.
    """
    return prog.provide["model"].weight.detach().clone()


def test_a_cancelled_run_resumes_to_the_uninterrupted_result(tmp_path):
    """The step in progress finishes, so the final checkpoint holds a whole state."""
    whole = program()
    run(whole)
    store = tmp_path / "run"
    ended = Ended()
    with pytest.raises(C3liRunCancelled, match="cancelled after 5 steps"):
        run(
            program(),
            Checkpointer(every=1, unit="epoch"),
            CancelAfter(steps=5),
            ended,
            store=store,
        )
    assert ended.steps == 5
    assert ended.payload["cancelled"] and ended.payload["at_boundary"]
    carried = program()
    run(carried, Checkpointer(every=1, unit="epoch"), store=store, resume="last")
    assert torch.equal(weights(carried), weights(whole))


def test_a_pass_budget_stops_after_that_many_passes():
    """Four steps make a pass, so one pass is four steps."""
    ended = Ended()
    with pytest.raises(C3liRunCancelled, match="cancelled after 1 passes"):
        run(program(), CancelAfter(epochs=1), ended)
    assert ended.steps == 4


def test_a_time_budget_stops_once_the_clock_runs_past_it(monkeypatch):
    """Whichever budget runs out first wins; here the clock does."""
    clock = iter(range(0, 1000, 10))
    monkeypatch.setattr(hooks.time, "monotonic", lambda: next(clock))
    ended = Ended()
    with pytest.raises(C3liRunCancelled, match=r"cancelled after \d+ s"):
        run(program(), CancelAfter(seconds=25, steps=100), ended)
    assert 0 < ended.steps < 12


def test_each_resumed_run_gets_a_budget_of_its_own(tmp_path):
    """So a long run can be split across jobs that each stay within a limit."""
    store = tmp_path / "run"
    first, second = Ended(), Ended()
    with pytest.raises(C3liRunCancelled):
        run(
            program(),
            Checkpointer(every=1, unit="step"),
            CancelAfter(steps=3),
            first,
            store=store,
        )
    with pytest.raises(C3liRunCancelled):
        run(
            program(),
            Checkpointer(every=1, unit="step"),
            CancelAfter(steps=3),
            second,
            store=store,
            resume="last",
        )
    assert (first.steps, second.steps) == (3, 3)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({}, "needs a budget"), ({"steps": 0}, r"\['steps'\] must be positive")],
)
def test_a_budget_is_required_and_positive(kwargs, match):
    """A canceller that never cancels, or always does, is a mistake."""
    with pytest.raises(ValueError, match=match):
        CancelAfter(**kwargs)


def test_the_canceller_is_never_dropped():
    """A muted canceller would silently stop responding, as for GracefulStop."""
    assert CancelAfter(steps=1).critical
