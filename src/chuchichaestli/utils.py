"""Various utility functions for chuchichaestli.

This file is part of Chuchichaestli.

Chuchichaestli is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Chuchichaestli is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Chuchichaestli.  If not, see <http://www.gnu.org/licenses/>.

Developed by the Intelligent Vision Systems Group at ZHAW.
"""

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
