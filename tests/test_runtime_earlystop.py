# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests that early stopping agrees with torch about what a plateau is."""

import pytest
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from chuchichaestli.runtime.events import Event, EventType, Signal
from chuchichaestli.runtime.hooks import EarlyStop


SERIES = {
    "slow decay": [1.0, 0.9, 0.8999, 0.8998] + [0.7] * 14,
    "flat": [0.5] * 14,
    "improving": [1.0 - 0.05 * i for i in range(14)],
    "noisy": [1.0, 0.8, 0.81, 0.79, 0.795, 0.6] + [0.61] * 12,
}


def observe(hook: EarlyStop, value: float) -> Signal:
    """Show one monitored value to the hook.

    Args:
        hook: The hook under test.
        value: The value to report.
    """
    return hook.on(Event(EventType.STEP_ENDED, "p", payload={"m": value}))


@pytest.mark.parametrize("name", sorted(SERIES))
@pytest.mark.parametrize("mode", ["min", "max"])
@pytest.mark.parametrize("threshold_mode", ["rel", "abs"])
def test_it_fires_when_the_torch_scheduler_would(name, mode, threshold_mode):
    """The plateau test is `ReduceLROnPlateau`'s, so both must agree exactly."""
    series = SERIES[name]
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = ReduceLROnPlateau(
        opt,
        mode=mode,
        patience=3,
        threshold=1e-4,
        threshold_mode=threshold_mode,
        factor=0.5,
    )
    hook = EarlyStop(
        "m", mode=mode, patience=3, threshold=1e-4, threshold_mode=threshold_mode
    )
    torch_fired = ours_fired = None
    for i, value in enumerate(series):
        before = opt.param_groups[0]["lr"]
        scheduler.step(value)
        if torch_fired is None and opt.param_groups[0]["lr"] != before:
            torch_fired = i
        if ours_fired is None and observe(hook, value) is Signal.BREAK:
            ours_fired = i
    assert ours_fired == torch_fired


def test_the_defaults_match_the_scheduler():
    """Pairing the two must not need the arguments repeating."""
    hook = EarlyStop("m")
    assert (hook.mode, hook.patience, hook.threshold, hook.threshold_mode) == (
        "min",
        10,
        1e-4,
        "rel",
    )


def test_patience_tolerates_that_many_then_stops():
    """`patience` counts the bad observations tolerated, as in torch."""
    hook = EarlyStop("m", patience=2)
    observe(hook, 1.0)
    assert observe(hook, 1.0) is Signal.GO
    assert observe(hook, 1.0) is Signal.GO
    assert observe(hook, 1.0) is Signal.BREAK


def test_a_non_numeric_value_is_ignored():
    """A payload key that is not a number must not count as a bad observation."""
    hook = EarlyStop("m", patience=0)
    assert hook.on(Event(EventType.STEP_ENDED, "p", payload={"m": "n/a"})) is Signal.GO
    assert hook.on(Event(EventType.STEP_ENDED, "p", payload={})) is Signal.GO
    assert hook.waited == 0


def test_a_bool_is_not_a_measurement():
    """`bool` is an `int`, so it would otherwise slip through as a value."""
    hook = EarlyStop("m", patience=0)
    assert hook.on(Event(EventType.STEP_ENDED, "p", payload={"m": True})) is Signal.GO
    assert hook.waited == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mode": "lower"}, "Unsupported mode"),
        ({"threshold_mode": "percent"}, "Unsupported threshold mode"),
    ],
)
def test_unknown_settings_are_rejected(kwargs, match):
    """The error lists what is accepted, as elsewhere in the package."""
    with pytest.raises(ValueError, match=match):
        EarlyStop("m", **kwargs)


def test_state_round_trips():
    """A resumed run must not forget how long it has been waiting."""
    hook = EarlyStop("m", patience=5)
    for value in (1.0, 1.0, 1.0):
        observe(hook, value)
    revived = EarlyStop("m", patience=5)
    revived.load_state_dict(hook.state_dict())
    assert (revived.best, revived.waited) == (hook.best, hook.waited)
