# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for turning cancellation signals into graceful stops."""

import os
import signal

import pytest

from chuchichaestli.runtime.events import C3liRuntimeError, Event, EventType, Signal
from chuchichaestli.runtime.hooks import GracefulStop
from chuchichaestli.runtime.runtime import Runtime
from chuchichaestli.runtime.stages import Call, Program


class Watch:
    """A hook keeping every event it saw."""

    def __init__(self):
        """Constructor."""
        self.events: list[Event] = []

    def on(self, event: Event) -> Signal:
        """Record the event.

        Args:
            event: What the runtime just did.
        """
        self.events.append(event)
        return Signal.GO


def raise_signal(name: str):
    """Build a stage body that sends a signal to this process.

    Args:
        name: Name of the signal to send.
    """

    def send(ctx):
        """Send the signal.

        Args:
            ctx: Execution context for this entry.
        """
        os.kill(os.getpid(), getattr(signal, name))

    return send


@pytest.mark.parametrize("name", ["SIGINT", "SIGTERM"])
def test_a_signal_stops_the_run_gracefully(name):
    """SIGTERM otherwise kills the process without unwinding at all."""
    watch = Watch()
    program = Program([Call("a", fn=raise_signal(name)), Call("b", fn=lambda c: None)])
    with pytest.raises(C3liRuntimeError, match=f"cancelled by {name}"):
        Runtime(program, hooks=[GracefulStop(), watch]).run()

    ended = [e for e in watch.events if e.type is EventType.RUN_ENDED]
    assert ended and ended[0].payload["aborted"] == f"cancelled by {name}"
    began = [e.path for e in watch.events if e.type is EventType.STAGE_BEGAN]
    assert "program/1:b" not in began


def test_handlers_are_restored_after_the_run():
    """A library taking over SIGTERM must give it back."""
    before = signal.getsignal(signal.SIGTERM)
    Runtime(Program([Call("a", fn=lambda c: None)]), hooks=[GracefulStop()]).run()
    assert signal.getsignal(signal.SIGTERM) is before


def test_handlers_are_restored_even_when_the_run_fails():
    """The restore rides on RUN_ENDED, which is emitted from a finally."""
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(C3liRuntimeError):
        Runtime(
            Program([Call("a", fn=raise_signal("SIGTERM"))]), hooks=[GracefulStop()]
        ).run()
    assert signal.getsignal(signal.SIGTERM) is before


def test_an_unknown_signal_name_warns_rather_than_raises():
    """Signal names are not portable, and a missing one must not end the run."""
    with pytest.warns(UserWarning, match="No signal named"):
        Runtime(
            Program([Call("a", fn=lambda c: None)]),
            hooks=[GracefulStop(signals=("SIGNOTREAL",))],
        ).run()


def test_cancel_is_critical():
    """A muted canceller would silently stop responding to scancel."""
    assert GracefulStop.critical is True
