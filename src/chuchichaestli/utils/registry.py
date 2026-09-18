# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Registry lookups that name the alternatives when they fail."""

from __future__ import annotations
from collections.abc import Callable, Collection, Mapping
from typing import Any


__all__ = ["require", "UNSUPPORTED"]


UNSUPPORTED = "Unsupported{context}: {name!r}. Use one of {options}."


def require(
    name: Any,
    registry: Collection[Any],
    context: str = "",
    message: str = UNSUPPORTED,
    fallback: Callable[[Any], Any] | None = None,
) -> Any:
    """Return what a registry holds under a name, or the name itself.

    Args:
        name: Key to look up.
        registry: Entries accepted at this position. A mapping's value is
            returned; any other collection returns the name, so it validates.
        context: What the position is, used by the default message.
        message: Template for the error, taking `{name}`, `{context}` and
            `{options}`. `{context}` already carries its leading space.
        fallback: Resolves a name the registry lacks, e.g. by importing it.

    Raises:
        ValueError: If the registry holds no such name.
    """
    if name in registry:
        return registry[name] if isinstance(registry, Mapping) else name
    if fallback is not None:
        return fallback(name)
    raise ValueError(
        message.format(
            name=name,
            context=f" {context}" if context else "",
            options=sorted(registry),
        )
    )
