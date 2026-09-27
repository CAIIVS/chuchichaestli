# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the JSON Lines trace and how often it reaches the filesystem."""

import json

import pytest

from chuchichaestli.runtime.events import Event, EventType, Progress
from chuchichaestli.runtime.hooks import Jsonl
from chuchichaestli.runtime.runtime import Runtime
from chuchichaestli.runtime.stages import Call, Program


def step(n: int) -> Event:
    """Build a step event.

    Args:
        n: Step number to record.
    """
    return Event(EventType.STEP_ENDED, "program/0:a", Progress(global_step=n), {"i": n})


def test_the_default_reaches_the_file_immediately(tmp_path):
    """A killed run must leave the records it already emitted."""
    path = tmp_path / "trace.jsonl"
    hook = Jsonl("trace.jsonl", store=tmp_path)
    for n in range(3):
        hook.on(step(n))
    assert len(path.read_text().splitlines()) == 3


def test_a_larger_interval_buffers(tmp_path):
    """Raising the interval is what trades durability for fewer syscalls."""
    path = tmp_path / "trace.jsonl"
    hook = Jsonl("trace.jsonl", store=tmp_path, flush_every=1000)
    for n in range(3):
        hook.on(step(n))
    assert path.read_text() == ""


def test_closing_flushes_what_was_buffered(tmp_path):
    """Nothing is lost when the run ends of its own accord."""
    path = tmp_path / "trace.jsonl"
    hook = Jsonl("trace.jsonl", store=tmp_path, flush_every=1000)
    for n in range(3):
        hook.on(step(n))
    hook.close()
    assert len(path.read_text().splitlines()) == 3


def test_a_run_ending_flushes_even_when_buffered(tmp_path):
    """`RUN_ENDED` closes the file, so a graceful stop keeps the whole trace."""
    path = tmp_path / "trace.jsonl"
    Runtime(
        Program([Call("a", fn=lambda c: None)]),
        hooks=[Jsonl("trace.jsonl", flush_every=1000)],
        store=tmp_path,
    ).run()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["type"] == "run.began"
    assert lines[-1]["type"] == "run.ended"


def test_the_interval_counts_records_not_events(tmp_path):
    """Filtered-out events must not advance the counter towards a flush."""
    path = tmp_path / "trace.jsonl"
    hook = Jsonl(
        "trace.jsonl", store=tmp_path, only=(EventType.STEP_ENDED,), flush_every=2
    )
    hook.on(Event(EventType.STAGE_BEGAN, "program"))
    hook.on(step(0))
    assert path.read_text() == ""
    hook.on(step(1))
    assert len(path.read_text().splitlines()) == 2


def test_a_non_positive_interval_is_rejected(tmp_path):
    """Zero would never flush, which is a silent trap rather than a setting."""
    with pytest.raises(ValueError, match="positive flush interval"):
        Jsonl("t.jsonl", flush_every=0)


def test_a_store_of_its_own_overrides_the_run(tmp_path):
    """A trace can be put somewhere else without moving the checkpoints."""
    elsewhere = tmp_path / "elsewhere"
    Runtime(
        Program([Call("a", fn=lambda c: None)]),
        hooks=[Jsonl("trace.jsonl", store=elsewhere)],
        store=tmp_path / "store",
    ).run()
    assert (elsewhere / "trace.jsonl").exists()
    assert not (tmp_path / "store" / "trace.jsonl").exists()


def test_a_trace_told_where_to_go_needs_no_run_store(tmp_path):
    """Only a hook relying on the run's store forces one to be given."""
    Runtime(
        Program([Call("a", fn=lambda c: None)]),
        hooks=[Jsonl("trace.jsonl", store=tmp_path)],
    ).run()
    assert (tmp_path / "trace.jsonl").exists()


def test_a_trace_reads_back_as_the_events_it_recorded(tmp_path):
    """A trace is only useful as an oracle if it round-trips."""
    hook = Jsonl("trace.jsonl", store=tmp_path)
    written = [step(1), step(2)]
    for event in written:
        hook.on(event)
    hook.close()
    assert Jsonl.read(tmp_path / "trace.jsonl") == written


def test_reading_a_trace_can_pick_one_kind_of_event(tmp_path):
    """The whole point at a call site: the steps, not the stage chatter."""
    hook = Jsonl("trace.jsonl", store=tmp_path)
    for event in (Event(EventType.STAGE_BEGAN, "program"), step(1)):
        hook.on(event)
    hook.close()
    read = Jsonl.read(tmp_path / "trace.jsonl", only=[EventType.STEP_ENDED])
    assert read == [step(1)]


def test_reading_a_trace_that_is_not_there_says_so(tmp_path):
    """The error has to name the file, as `Load` does."""
    with pytest.raises(FileNotFoundError, match="absent.jsonl"):
        Jsonl.read(tmp_path / "absent.jsonl")


def test_a_trace_written_by_a_run_reads_back(tmp_path):
    """The round trip that matters: what a real run left behind."""
    Runtime(
        Program([Call("a", fn=lambda c: None)]),
        hooks=[Jsonl("trace.jsonl", store=tmp_path)],
    ).run()
    events = Jsonl.read(tmp_path / "trace.jsonl")
    assert events[0].type is EventType.RUN_BEGAN
    assert events[-1].type is EventType.RUN_ENDED


def test_a_filtered_trace_still_closes_when_the_run_ends(tmp_path):
    """A buffered record is lost if the filter skips the close path."""
    hook = Jsonl(
        "trace.jsonl", store=tmp_path, only=[EventType.STEP_ENDED], flush_every=100
    )
    hook.on(step(1))
    hook.on(Event(EventType.RUN_ENDED, "program"))
    assert hook._handle is None
    assert Jsonl.read(tmp_path / "trace.jsonl") == [step(1)]
