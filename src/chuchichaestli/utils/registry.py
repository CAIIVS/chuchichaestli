# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Registry lookups that name the alternatives when they fail."""

from __future__ import annotations
from collections.abc import Callable, Collection, Mapping
from typing import Any


__all__ = ["require"]


def require(
    name: Any,
    registry: Collection[Any],
    context: str = "",
    message: Callable[[list[Any]], str] | None = None,
    fallback: Callable[[Any], Any] | None = None,
) -> Any:
    """Return what a registry holds under a name, or the name itself.

    Args:
        name: Key to look up.
        registry: Entries accepted at this position. A mapping's value is
            returned; any other collection returns the name, so it validates.
        context: What the position is, used by the default message.
        message: Builds the error from the sorted alternatives, replacing the
            default. Called only on failure.
        fallback: Resolves a name the registry lacks, e.g. by importing it.

    Raises:
        ValueError: If the registry holds no such name.
    """
    if name in registry:
        return registry[name] if isinstance(registry, Mapping) else name
    if fallback is not None:
        return fallback(name)
    options = sorted(registry)
    if message is not None:
        raise ValueError(message(options))
    qualifier = f" {context}" if context else ""
    raise ValueError(f"Unsupported{qualifier}: {name!r}. Use one of {options}.")
