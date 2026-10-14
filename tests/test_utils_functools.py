# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for picking the plain numbers out of a mapping."""

import pytest

from chuchichaestli.utils.functools import as_number, numbers


@pytest.mark.parametrize(
    ("value", "expected"),
    [(3, 3), (0.5, 0.5), (True, None), ("1", None), (None, None)],
    ids=["int", "float", "bool", "str", "none"],
)
def test_only_a_plain_number_counts_as_one(value, expected):
    """A bool is an int to Python, but not a value worth plotting."""
    assert as_number(value) == expected


def test_numbers_keep_their_type():
    """An epoch stays an int, so it prints as one."""
    picked = numbers({"loss": 0.25, "epoch": 3, "trains": True, "stage": "Train"})
    assert picked == {"loss": 0.25, "epoch": 3}
    assert isinstance(picked["epoch"], int)


def test_skipped_keys_are_left_out():
    """A planned total is a number, but not a measurement."""
    assert numbers({"loss": 0.25, "total": 8}, skip=("total",)) == {"loss": 0.25}
