# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for terminal colouring."""

import io

import pytest

from chuchichaestli.utils.ansi import cli_pbar, ANSIShade, paint, ansi_supported


class Tty(io.StringIO):
    """A stream that claims to be a terminal."""

    def isatty(self) -> bool:
        """Report as a terminal."""
        return True


def test_paint_wraps_and_resets():
    """Every sequence has to be closed, or the colour bleeds into later output."""
    painted = paint("hi", ANSIShade.CYAN)
    assert painted.startswith(ANSIShade.CYAN.value)
    assert painted.endswith(ANSIShade.RESET.value)
    assert "hi" in painted


def test_paint_applies_shades_in_order():
    """Combining bold with a colour has to emit both."""
    assert paint("x", ANSIShade.BOLD, ANSIShade.RED) == "\033[1m\033[31mx\033[0m"


def test_paint_off_returns_the_text_untouched():
    """The plain path must be byte-identical, not merely visually similar."""
    assert paint("hi", ANSIShade.CYAN, on=False) == "hi"
    assert paint("hi") == "hi"
    assert paint("", ANSIShade.CYAN) == ""


def test_a_non_terminal_is_never_coloured():
    """A redirected log must not receive escape codes."""
    assert ansi_supported(io.StringIO()) is False


def test_a_terminal_is_coloured(monkeypatch):
    """A real terminal should get colour by default."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert ansi_supported(Tty()) is True


def test_no_color_wins_over_a_terminal(monkeypatch):
    """The NO_COLOR convention must override detection."""
    monkeypatch.setenv("NO_COLOR", "1")
    assert ansi_supported(Tty()) is False


def test_a_stream_without_isatty_is_not_coloured():
    """Anything file-like should be safe to pass, not just real streams."""

    class Bare:
        """A writable object with no `isatty`."""

        def write(self, text: str) -> None:
            """Discard the text.

            Args:
                text: What was written.
            """

    assert ansi_supported(Bare()) is False


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
