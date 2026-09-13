# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Process layouts a run can execute across."""

from __future__ import annotations
from typing import Any
import os
import torch
from torch import nn
from chuchichaestli.runtime.traits import Topology


__all__ = ["Local", "default_device", "auto_topology"]


def default_device() -> torch.device:
    """Return the device a run uses when none is given.

    Defers to torch's own accelerator lookup, which covers CUDA, ROCm, Intel
    XPU, MPS and MTIA, plus out-of-tree backends such as Gaudi that register
    themselves once their plugin is imported.
    """
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None and accelerator.is_available():
        return accelerator.current_accelerator()
    for backend in ("cuda", "xpu", "mps"):
        probe = getattr(torch, backend, None) or getattr(torch.backends, backend, None)
        if probe is not None and probe.is_available():
            return torch.device(backend)
    return torch.device("cpu")


def auto_topology(device: torch.device | str | None = None) -> Topology:
    """Pick a topology from the environment.

    A run launched under `torchrun` has `RANK` and `WORLD_SIZE` set, so going
    distributed needs no code change.

    Args:
        device: Device to pin, or `None` to detect one.

    Raises:
        NotImplementedError: Under `torchrun`, until `Ddp` lands.
    """
    if os.environ.get("RANK") and os.environ.get("WORLD_SIZE"):
        raise NotImplementedError(
            "Distributed runs are not supported yet; pass topology= explicitly."
        )
    return Local(device=device)


class Local:
    """A run confined to one process.

    Every collective is a no-op, which is precisely why stages can call them
    unconditionally instead of branching on whether the run is distributed.

    Attributes:
        device: Device this process computes on.
    """

    def __init__(self, device: torch.device | str | None = None):
        """Constructor.

        Args:
            device: Device this process computes on; detected when `None`.
        """
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.device = torch.device(device) if device is not None else default_device()

    @property
    def is_main(self) -> bool:
        """Whether this process is the one that writes; always true."""
        return True

    def barrier(self) -> None:
        """Block until every process has arrived; nothing to wait for."""

    def reduce(self, value: torch.Tensor, op: str = "mean") -> torch.Tensor:
        """Combine a tensor across processes.

        Args:
            value: Tensor held by this process.
            op: Reduction to apply; ignored with a single process.
        """
        return value

    def wrap(self, module: nn.Module) -> nn.Module:
        """Prepare a module for distributed training.

        Args:
            module: Module to wrap; returned unchanged.
        """
        return module

    def broadcast(self, value: Any) -> Any:
        """Return rank 0's value on every process.

        Args:
            value: This process's candidate value, which is already rank 0's.
        """
        return value

    def state_of(self, stateful: Any) -> dict[str, Any]:
        """Capture a component's state.

        Args:
            stateful: Component whose state is wanted.
        """
        return stateful.state_dict()

    def load_state(self, stateful: Any, state: dict[str, Any]) -> None:
        """Restore state captured by `state_of`.

        Args:
            stateful: Component to restore.
            state: Mapping as returned by `state_of`.
        """
        stateful.load_state_dict(state)

    def __repr__(self) -> str:
        """Return a short description of the topology."""
        return f"Local(device={self.device})"
