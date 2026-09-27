# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Terminal colouring, for output a person reads rather than a file keeps."""

from __future__ import annotations
import os
from enum import Enum
from typing import IO, TextIO


__all__ = ["ANSIShade", "Pinned", "ansi_supported", "cli_pbar", "paint"]


class ANSIShade(str, Enum):
    """Select-graphic-rendition codes, named for how they are meant to read."""

    DIM = "\033[2m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    RESET = "\033[0m"


def ansi_supported(stream: IO[str]) -> bool:
    """Whether a stream should be written to in colour.

    Honours the `NO_COLOR` convention, and answers `False` for anything that is
    not a terminal, so escape codes never reach a redirected log.

    Args:
        stream: Destination the output is headed for.
    """
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", bool)())


def paint(text: str, *shades: ANSIShade, on: bool = True) -> str:
    """Wrap text in graphic-rendition codes, or return it unchanged.

    Args:
        text: What to colour.
        shades: Codes to apply, in order.
        on: Whether to colour at all; `False` returns `text` untouched.
    """
    if not on or not shades or not text:
        return text
    return "".join(s.value for s in shades) + text + ANSIShade.RESET.value


class Pinned:
    """Keeps one line at the foot of a stream while the rest scroll above it.

    Attributes:
        line: What is currently pinned, or `""` when nothing is.
    """

    ERASE = "\r\x1b[2K"

    def __init__(self, stream: TextIO, live: bool = True):
        """Constructor.

        Args:
            stream: Where both the pinned line and the rest are written.
            live: Whether the stream redraws. A file keeps every line it is
                given, so nothing is pinned to it.
        """
        self.stream = stream
        self.live = live
        self.line = ""

    def scroll(self, text: str) -> None:
        """Write a line above whatever is pinned.

        Args:
            text: The line to write.
        """
        if not self.live:
            print(text, file=self.stream, flush=True)
            return
        self.stream.write(f"{self.ERASE}{text}\n{self.line}")
        self.stream.flush()

    def pin(self, text: str) -> None:
        """Hold a line at the foot of the stream, replacing any before it.

        Args:
            text: The line to hold.
        """
        if not self.live:
            return
        self.line = text
        self.stream.write(f"{self.ERASE}{text}")
        self.stream.flush()

    def drop(self) -> None:
        """Let go of the pinned line, leaving the cursor on a fresh one."""
        if not self.live or not self.line:
            return
        self.stream.write(f"{self.ERASE}")
        self.stream.flush()
        self.line = ""


def cli_pbar(
    r_fill: float,
    prefix: str | list = "",
    postfix: str | list = "",
    bar_length: int = 60,
    fill_symbol: str = "#",
    empty_symbol: str = "-",
    float_fmt: str = "{:.2f}",
    int_fmt: str = "{:4d}",
) -> str:
    """Return a progress bar of a given relative length, with its labels.

    Args:
        r_fill: How far along the bar is, between 0 and 1.
        prefix: Label written before the bar.
        postfix: Label written after the bar.
        bar_length: Width of the bar itself; the labels alone when it is not
            positive.
        fill_symbol: What the filled part is drawn with.
        empty_symbol: What the rest is drawn with.
        float_fmt: How floats in the labels are formatted.
        int_fmt: How ints in the labels are formatted.
    """
    if isinstance(prefix, list | tuple):
        for i, p in enumerate(prefix):
            if isinstance(p, float):
                prefix[i] = float_fmt.format(p)
            elif isinstance(p, int):
                prefix[i] = int_fmt.format(p)
            if not isinstance(prefix[i], str):
                prefix[i] = f"{prefix[i]}"
        prefix = " ".join(prefix)
    if isinstance(postfix, list | tuple):
        for i, p in enumerate(postfix):
            if isinstance(p, float):
                postfix[i] = float_fmt.format(p)
            elif isinstance(p, int):
                postfix[i] = int_fmt.format(p)
            if not isinstance(postfix[i], str):
                postfix[i] = f"{postfix[i]}"
        postfix = " ".join(postfix)
    filled = int(r_fill * bar_length)
    bar = fill_symbol * filled + empty_symbol * (bar_length - filled)
    if bar_length > 0:
        line = f"{prefix} [{bar}] {postfix}"
    else:
        line = f"{prefix}\t{postfix}"
    return line
