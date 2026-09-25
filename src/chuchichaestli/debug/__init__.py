# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Debugging utilities module of chuchichaestli."""


def as_bytes(byte_str: str = "MB") -> int:
    """Convert bytes strings to integers."""
    match byte_str.lower():
        case "kib":
            return 1 << 10
        case "mib":
            return 1 << 20
        case "gib":
            return 1 << 30
        case "tib":
            return 1 << 40
        case "kb":
            return 1_000
        case "mb":
            return 1_000_000
        case "gb":
            return 1_000_000_000
        case "tb":
            return 1_000_000_000_000
        case _:
            return 1


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
