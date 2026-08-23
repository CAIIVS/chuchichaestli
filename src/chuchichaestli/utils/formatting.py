# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""String formatting and manipulation functions for chuchichaestli."""

__all__ = ["metric_suffix"]

METRIC_UNITS = {
    "K": 10**3,
    "M": 10**6,
    "G": 10**9,
    "T": 10**12,
    "P": 10**15,
}


def metric_suffix(num: int | float, precision: int = 1) -> str:
    """Format large numbers with metric unit suffixes."""
    for k in ["P", "T", "G", "M", "K"]:
        if num >= METRIC_UNITS[k]:
            return f"{num / METRIC_UNITS[k]:.{precision}f}{k}"
    return str(num)
