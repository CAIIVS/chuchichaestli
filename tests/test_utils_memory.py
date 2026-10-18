# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for measuring a device's peak memory."""

import torch

from chuchichaestli.utils import memory
from chuchichaestli.utils.memory import peak_memory_allocated, reset_peak_memory


class Counters:
    """A device memory counter peaking at 3 MiB."""

    def __init__(self, refuses: bool = False):
        """Constructor.

        Args:
            refuses: Whether the device raises when asked.
        """
        self.refuses = refuses
        self.resets: list[torch.device] = []

    def reset_peak_memory_stats(self, device):
        """Start a new peak."""
        if self.refuses:
            raise RuntimeError("no counter here")
        self.resets.append(device)

    def max_memory_allocated(self, device):
        """Report the peak."""
        if self.refuses:
            raise RuntimeError("no counter here")
        return 3 * 2**20


def test_the_cpu_has_no_peak_to_measure():
    """Nothing is reported rather than a misleading zero."""
    assert reset_peak_memory("cpu") is False
    assert peak_memory_allocated("cpu") is None


def test_a_device_s_counter_is_reset_and_read(monkeypatch):
    """Through whatever keeps the device's counter."""
    counters = Counters()
    monkeypatch.setattr(memory, "_counters", lambda device: counters)
    assert reset_peak_memory("cuda")
    assert counters.resets == [torch.device("cuda")]
    assert peak_memory_allocated("cuda") == 3 * 2**20


def test_a_device_refusing_is_unmeasured_not_an_error(monkeypatch):
    """A backend without the counter costs the number, not the run."""
    monkeypatch.setattr(memory, "_counters", lambda device: Counters(refuses=True))
    assert reset_peak_memory("mps") is False
    assert peak_memory_allocated("mps") is None


def test_older_torch_falls_back_to_the_device_s_own_module(monkeypatch):
    """Before `torch.accelerator` had counters, `torch.cuda` kept them."""
    monkeypatch.delattr(torch.accelerator, "max_memory_allocated", raising=False)
    assert memory._counters(torch.device("cuda")) is torch.cuda
    assert memory._counters(torch.device("meta")) is None
