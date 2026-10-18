# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measuring the memory a device has allocated."""

from __future__ import annotations
from typing import Any
import torch

__all__ = ["reset_peak_memory", "peak_memory_allocated"]


def _counters(device: torch.device) -> Any:
    """Return what keeps a device's peak-memory counter, or `None` if nothing does.

    Args:
        device: The device to measure.
    """
    if device.type == "cpu":
        return None
    accelerator = getattr(torch, "accelerator", None)
    if hasattr(accelerator, "max_memory_allocated"):
        return accelerator
    module = getattr(torch, device.type, None)
    has_peak = hasattr(module, "max_memory_allocated")
    return module if has_peak and hasattr(module, "reset_peak_memory_stats") else None


def reset_peak_memory(device: torch.device | str) -> bool:
    """Start a device's peak-memory counter afresh, if it keeps one.

    Args:
        device: The device to measure.
    """
    device = torch.device(device)
    counters = _counters(device)
    if counters is None:
        return False
    try:
        counters.reset_peak_memory_stats(device)
    except Exception:  # noqa: BLE001 - a device refusing has no counter to reset
        return False
    return True


def peak_memory_allocated(device: torch.device | str) -> int | None:
    """Return a device's peak allocated bytes since its reset, or `None` if unmeasured.

    Args:
        device: The device to measure.
    """
    device = torch.device(device)
    counters = _counters(device)
    if counters is None:
        return None
    try:
        return int(counters.max_memory_allocated(device))
    except Exception:  # noqa: BLE001 - a device refusing has no peak to report
        return None
