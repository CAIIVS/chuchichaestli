# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Process layouts a run can execute across."""

from __future__ import annotations
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from datetime import timedelta
from typing import Any, Literal
import os
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
)
from torch.nn.parallel import DistributedDataParallel
from chuchichaestli.programs.events import C3liRunCancelled, C3liRunError, Signal
from chuchichaestli.programs.traits import Stateful, Topology
from chuchichaestli.utils.registry import require


__all__ = [
    "ReduceTypes",
    "REDUCE_OPS",
    "reduce_metrics",
    "lockstep",
    "Local",
    "Ddp",
    "default_device",
    "auto_topology",
]


ReduceTypes = Literal["mean", "sum", "min", "max"]

REDUCE_OPS: dict[str, Any] = {}
if dist.is_available():
    REDUCE_OPS = {
        "mean": dist.ReduceOp.SUM,
        "sum": dist.ReduceOp.SUM,
        "min": dist.ReduceOp.MIN,
        "max": dist.ReduceOp.MAX,
    }


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
    """
    if dist.is_available() and (
        dist.is_initialized()
        or (os.environ.get("RANK") and os.environ.get("WORLD_SIZE"))
    ):
        return Ddp(device=device)
    return Local(device=device)


def lockstep(topology: Topology, call: Any) -> Any:
    """Run a call on every process, agreeing on both outcome and failure.

    Ranks must issue the same collectives in the same order or the run
    deadlocks, and an exception unwinds only the process that raised it.

    Args:
        topology: Process layout the run executes across.
        call: Performs the work, returning a signal.

    Raises:
        C3liRunCancelled: On every rank that did not raise, if one was
            cancelled.
        C3liRunError: On every rank that did not raise, if any of them aborted.
        Exception: On the rank that raised one, after every rank has agreed
            that the run is over, so none is left waiting in a collective.
    """
    raised: Exception | None = None
    try:
        signal = call()
    except Exception as error:
        signal, raised = Signal.BREAK, error
    # 0: carried on, 1: aborted, 2: cancelled; the worst outcome wins
    outcome = 0 if raised is None else 2 if isinstance(raised, C3liRunCancelled) else 1
    if topology.world_size > 1:
        outcome = int(topology.reduce(torch.tensor([float(outcome)]), op="max"))
    if outcome:
        if raised is not None:
            raise raised
        if outcome == 2:
            raise C3liRunCancelled("another process was cancelled")
        raise C3liRunError("another process aborted the run")
    return topology.broadcast(signal)


def reduce_metrics(metrics: Any, topology: Topology) -> None:
    """Combine what every process accumulated, in place.

    Each metric says which of its state sums, which keeps the lowest or
    highest seen, and which is an OR flag (true if true anywhere).

    Args:
        metrics: The metrics to combine, each declaring its own state.
        topology: Process layout the run executes across.
    """
    if topology.world_size == 1:
        return
    for metric in metrics:
        for op, names in (
            ("sum", getattr(metric, "ADDITIVE", ())),
            ("min", getattr(metric, "SMALLEST", ())),
            ("max", getattr(metric, "LARGEST", ())),
        ):
            for name in names:
                state = getattr(metric, name, None)
                if isinstance(state, torch.Tensor):
                    setattr(metric, name, topology.reduce(state, op=op))
        for name in getattr(metric, "FLAGS", ()):
            state = getattr(metric, name, None)
            if isinstance(state, torch.Tensor):
                setattr(metric, name, topology.reduce(state.any(), op="max"))


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

    def unwrap(self, module: nn.Module) -> nn.Module:
        """Return the module `torch.compile` wrapped, or the module itself.

        Args:
            module: Module to unwrap.
        """
        while isinstance(getattr(module, "_orig_mod", None), nn.Module):
            module = module._orig_mod
        return module

    def no_sync(self, modules: Iterable[nn.Module]) -> nullcontext[None]:
        """Return a context holding nothing back, as one process averages nothing.

        Args:
            modules: Modules whose gradients are not averaged yet.
        """
        return nullcontext()

    def broadcast(self, value: Any) -> Any:
        """Return rank 0's value on every process.

        Args:
            value: This process's candidate value, which is already rank 0's.
        """
        return value

    def gather(self, value: Any) -> list[Any]:
        """Return this process's value as the only one there is.

        Args:
            value: This process's value.
        """
        return [value]

    def state_of(self, stateful: Stateful) -> dict[str, Any]:
        """Capture a component's state, a compiled module's under its own names.

        Args:
            stateful: Component whose state is wanted.
        """
        if isinstance(stateful, nn.Module):
            return self.unwrap(stateful).state_dict()
        return stateful.state_dict()

    def load_state(self, stateful: Stateful, state: dict[str, Any]) -> None:
        """Restore state captured by `state_of`.

        Args:
            stateful: Component to restore.
            state: Mapping as returned by `state_of`.
        """
        if isinstance(stateful, nn.Module):
            self.unwrap(stateful).load_state_dict(state)
            return
        stateful.load_state_dict(state)

    def __repr__(self) -> str:
        """Return a short description of the topology."""
        return f"Local(device={self.device})"


class Ddp:
    """A run spread over one process per device, replicating the model.

    Attributes:
        device: Device this process computes on, carrying its own index.
        find_unused_parameters: Whether to let a backward leave some
            parameters without a gradient, at a cost per step.
    """

    def __init__(
        self,
        backend: str | None = None,
        device: torch.device | str | None = None,
        find_unused_parameters: bool = False,
        timeout: float | None = None,
    ):
        """Constructor.

        Joins the process group the launcher set up, starting one from the
        environment if the caller has not already.

        Args:
            backend: Collective backend; `"nccl"` on an accelerator and
                `"gloo"` otherwise when not given.
            device: Device this process computes on; derived from the local
                rank when `None`.
            find_unused_parameters: Whether a backward may leave parameters
                without a gradient. The escape hatch for objectives that
                cannot detach what they do not train.
            timeout: Seconds a collective may block before failing, or
                `None` for torch's own default.

        Raises:
            RuntimeError: If torch was built without distributed support.
        """
        if not dist.is_available():
            raise RuntimeError(
                "Ddp needs torch built with distributed support. "
                "Use topology=Local() instead."
            )
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.find_unused_parameters = find_unused_parameters
        self.device = self._pin_to_rank(device)
        self._pin_process()
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend or self._backend(),
                rank=self.rank,
                world_size=self.world_size,
                timeout=None if timeout is None else timedelta(seconds=timeout),
            )
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def _pin_process(self) -> None:
        """Make this rank's device the current one for the whole process.

        A collective that is handed no device of its own uses the process's
        current one.
        """
        if self.device.type == "cpu" or self.device.index is None:
            return
        accelerator = getattr(torch, "accelerator", None)
        if accelerator is not None and hasattr(accelerator, "set_device_index"):
            accelerator.set_device_index(self.device.index)
            return
        backend = getattr(torch, self.device.type, None)
        if backend is not None and hasattr(backend, "set_device"):
            backend.set_device(self.device.index)

    def _pin_to_rank(self, device: torch.device | str | None) -> torch.device:
        """Return the device this rank computes on.

        Args:
            device: Device asked for, or `None` to derive one.
        """
        device = torch.device(device) if device is not None else default_device()
        if device.type == "cpu" or device.index is not None:
            return device
        return torch.device(device.type, self.local_rank)

    def _backend(self) -> str:
        """Return the collective backend suiting this rank's device."""
        return "gloo" if self.device.type == "cpu" else "nccl"

    @property
    def is_main(self) -> bool:
        """Whether this process is the one that writes."""
        return self.rank == 0

    def barrier(self) -> None:
        """Block until every process has arrived."""
        dist.barrier()

    def reduce(self, value: torch.Tensor, op: ReduceTypes = "mean") -> torch.Tensor:
        """Combine a tensor across processes.

        Args:
            value: Tensor held by this process.
            op: Reduction to apply.

        Raises:
            ValueError: If the reduction is not one this can apply.
        """
        require(op, REDUCE_OPS, "reduction")
        combined = value.detach().to(self.device).clone()
        if combined.dtype == torch.bool:
            combined = combined.to(torch.float32)
        dist.all_reduce(combined, op=REDUCE_OPS[op])
        if op == "mean":
            combined = combined / self.world_size
        return combined.to(dtype=value.dtype, device=value.device)

    def wrap(self, module: nn.Module) -> nn.Module:
        """Replicate a module so its gradients are averaged across ranks.

        Args:
            module: Module to wrap, already placed on this rank's device.
        """
        if not any(True for _ in module.parameters()):
            return module
        return DistributedDataParallel(
            module,
            device_ids=None if self.device.type == "cpu" else [self.device.index],
            find_unused_parameters=self.find_unused_parameters,
        )

    def unwrap(self, module: nn.Module) -> nn.Module:
        """Return the module a replica holds, for its parameters and spec.

        Args:
            module: Module to unwrap; returned unchanged when it carries no
                parameters, which `wrap` leaves alone.
        """
        while True:
            if isinstance(module, DistributedDataParallel):
                module = module.module
            elif isinstance(getattr(module, "_orig_mod", None), nn.Module):
                module = module._orig_mod
            else:
                return module

    @contextmanager
    def no_sync(self, modules: Iterable[nn.Module]) -> Iterator[None]:
        """Hold back averaging the replicas' gradients while the context lasts.

        Args:
            modules: Modules whose gradients are not averaged yet; those not
                replicated are left alone.
        """
        with ExitStack() as stack:
            for module in modules:
                if isinstance(module, DistributedDataParallel):
                    stack.enter_context(module.no_sync())
            yield

    def broadcast(self, value: Any) -> Any:
        """Return rank 0's value on every process.

        Args:
            value: This process's candidate value.
        """
        carrier = [value]
        dist.broadcast_object_list(carrier, src=0)
        return carrier[0]

    def gather(self, value: Any) -> list[Any]:
        """Return every process's value on every process, in rank order.

        Args:
            value: This process's value; tensors in it should be on the host.
        """
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return gathered

    def state_of(self, stateful: Stateful) -> dict[str, Any]:
        """Capture a component's state, gathered across processes.

        A replicated module reports its weights under a `module.` prefix and
        a sharded one reports pieces, so neither is asked directly; what
        comes back loads into a single-process run unchanged.

        Args:
            stateful: Component whose state is wanted.
        """
        if isinstance(stateful, nn.Module):
            return dict(get_model_state_dict(stateful))
        return stateful.state_dict()

    def load_state(self, stateful: Stateful, state: dict[str, Any]) -> None:
        """Restore state captured by `state_of`, scattered across processes.

        Args:
            stateful: Component to restore.
            state: Mapping as returned by `state_of`.
        """
        if isinstance(stateful, nn.Module):
            set_model_state_dict(stateful, dict(state))
            return
        stateful.load_state_dict(state)

    def close(self) -> None:
        """Leave the process group, if this object started one."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def __repr__(self) -> str:
        """Return a short description of the topology."""
        return f"Ddp(rank={self.rank}/{self.world_size}, device={self.device})"
