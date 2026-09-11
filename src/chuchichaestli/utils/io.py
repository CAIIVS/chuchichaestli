# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Filesystem helpers for reading and writing states."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


__all__ = ["staged"]


@contextmanager
def staged(*targets: Path) -> Iterator[list[Path]]:
    """Yield a scratch path per target; all are moved into place, or none are.

    Args:
        targets: Files the write is to land on.
    """
    scratches = [t.with_stem(f"{t.stem}.part") for t in targets]
    try:
        yield scratches
    except BaseException:
        for scratch in scratches:
            scratch.unlink(missing_ok=True)
        raise
    for scratch, target in zip(scratches, targets):
        scratch.replace(target)
