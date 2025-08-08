# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Various utility functions for chuchichaestli."""

import sys
from functools import partialmethod


def partialclass(name: str, cls: type[object], *args, **kwargs):
    """Partial for __init__ class constructors."""
    part_cls = type(
        name, (cls,), {"__init__": partialmethod(cls.__init__, *args, **kwargs)}
    )
    try:
        part_cls.__module__ = sys._getframe(1).f_globals.get("__name__", "__main__")
    except (AttributeError, ValueError):
        pass
    return part_cls
