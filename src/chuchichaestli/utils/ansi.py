# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Terminal colouring, for output a person reads rather than a file keeps."""

from __future__ import annotations
import os
from enum import Enum
from typing import IO


__all__ = ["ANSIShade", "ansi_supported", "paint"]


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
