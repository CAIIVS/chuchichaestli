# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests that a hook's unit names come from the type that accepts them."""

from typing import get_args


def test_the_names_a_type_accepts_are_the_names_the_runtime_accepts():
    """Written twice they drift; a hook would take a unit it cannot act on."""
    from chuchichaestli.runtime.hooks import (
        CHECKPOINT_UNIT_MAP,
        UNIT_MAP,
        CheckpointUnitTypes,
        ModeTypes,
        MODES,
        ThresholdModeTypes,
        THRESHOLD_MODES,
    )

    assert set(CHECKPOINT_UNIT_MAP) == set(get_args(CheckpointUnitTypes))
    assert set(MODES) == set(get_args(ModeTypes))
    assert set(THRESHOLD_MODES) == set(get_args(ThresholdModeTypes))
    assert set(CHECKPOINT_UNIT_MAP) <= set(UNIT_MAP)
    assert all(UNIT_MAP[name] is event for name, event in CHECKPOINT_UNIT_MAP.items())


def test_a_bar_takes_only_the_units_it_can_draw():
    """Its units are a subset, so a Literal of its own is what it derives from."""
    from chuchichaestli.runtime.hooks import (
        BAR_UNIT_MAP,
        UNIT_MAP,
        BarUnitTypes,
    )

    assert set(BAR_UNIT_MAP) == set(get_args(BarUnitTypes))
    assert set(BAR_UNIT_MAP) < set(UNIT_MAP)
    assert all(UNIT_MAP[name] is event for name, event in BAR_UNIT_MAP.items())
