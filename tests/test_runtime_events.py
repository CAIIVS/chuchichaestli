# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the runtime's event and signal vocabulary."""

import pytest

from chuchichaestli.runtime.events import (
    C3liRuntimeError,
    Event,
    EventType,
    Progress,
    Signal,
    filter_priority,
)


def test_signal_halts():
    """DONE and BREAK end a stage; GO and SKIP do not."""
    assert Signal.DONE.halts and Signal.BREAK.halts
    assert not Signal.GO.halts and not Signal.SKIP.halts


@pytest.mark.parametrize(
    ("signals", "expected"),
    [
        ((), Signal.GO),
        ((Signal.GO, Signal.GO), Signal.GO),
        ((Signal.GO, Signal.SKIP), Signal.SKIP),
        ((Signal.GO, Signal.BREAK, Signal.SKIP), Signal.BREAK),
        ((Signal.DONE, Signal.SKIP), Signal.DONE),
    ],
)
def test_filter_priority(signals, expected):
    """One hook asking to stop outweighs any number asking to continue."""
    assert filter_priority(signals) is expected


def test_abort_is_an_exception():
    """Stopping the whole program is raised, not returned."""
    assert issubclass(C3liRuntimeError, RuntimeError)
    with pytest.raises(C3liRuntimeError, match="because"):
        raise C3liRuntimeError("because")


def test_progress_transitions():
    """Counters advance without mutating the original."""
    start = Progress()
    stepped = start.next_step(samples=4).next_step(samples=4)
    assert (stepped.step, stepped.global_step, stepped.samples) == (2, 2, 8)
    assert start.step == 0
    rolled = stepped.next_epoch()
    assert (rolled.epoch, rolled.step, rolled.global_step) == (1, 0, 2)
    assert rolled.finish().done and not rolled.done


def test_progress_round_trip():
    """Progress survives its mapping form unchanged."""
    progress = Progress(epoch=2, step=3, global_step=11, samples=99)
    assert Progress.from_dict(progress.to_dict()) == progress


def test_progress_from_dict_ignores_unknown_keys():
    """A checkpoint written by a later version still loads."""
    assert Progress.from_dict({"epoch": 1, "invented": 5}) == Progress(epoch=1)


def test_event_is_json_serializable():
    """The trace needs no custom encoder, which is what makes it comparable."""
    import json

    event = Event(
        EventType.STEP_ENDED, "program/0:train", Progress(step=3), {"loss": 0.5}
    )
    restored = Event.from_dict(json.loads(json.dumps(event.to_dict())))
    assert restored == event


def test_event_types_are_distinct_strings():
    """Dotted values keep the enum readable in a trace file."""
    values = [member.value for member in EventType]
    assert len(values) == len(set(values))
    assert EventType.STEP_ENDED.value == "step.ended"
