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
