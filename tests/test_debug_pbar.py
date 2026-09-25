# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the console progress bar."""

import pytest

from chuchichaestli.debug import cli_pbar


def drawn(line: str) -> str:
    """Return just the bar from a rendered line.

    Args:
        line: What `cli_pbar` returned.
    """
    return line.split("[", 1)[1].split("]", 1)[0]


@pytest.mark.parametrize("filled", range(17))
def test_the_bar_is_the_width_it_was_asked_for(filled):
    """A bar that shrinks a character on most fractions has a ragged edge.

    Args:
        filled: Sixteenth of the bar to fill.
    """
    assert len(drawn(cli_pbar(filled / 16, bar_length=24))) == 24


def test_the_bar_fills_in_proportion():
    """The ends are exact, and the middle is the fraction rounded down."""
    assert drawn(cli_pbar(0.0, bar_length=8)) == "-" * 8
    assert drawn(cli_pbar(0.5, bar_length=8)) == "####----"
    assert drawn(cli_pbar(1.0, bar_length=8)) == "#" * 8
    assert drawn(cli_pbar(0.3, bar_length=8)) == "##------"
