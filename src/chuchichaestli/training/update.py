# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gradient clipping, optimizer stepping and weight averaging."""

from __future__ import annotations
import functools
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, get_args
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.swa_utils import (
    AveragedModel,
    get_ema_multi_avg_fn,
    get_swa_multi_avg_fn,
    update_bn,
)
from chuchichaestli.training.optim import OptimSpec
from chuchichaestli.utils.registry import require


__all__ = [
    "ClipTypes",
    "ReductionTypes",
    "NonFiniteTypes",
    "CLIP_MODES",
    "NONFINITE_MODES",
    "grad_norm",
    "UpdatePolicy",
    "Ema",
    "Swa",
    "average_targets",
]


ClipTypes = Literal["norm", "value"]
ReductionTypes = Literal["mean", "sum"]
NonFiniteTypes = Literal["raise", "skip", "ignore"]

CLIP_MODES: frozenset[str] = frozenset(get_args(ClipTypes))
NONFINITE_MODES: frozenset[str] = frozenset(get_args(NonFiniteTypes))


def grad_norm(params: Iterable[nn.Parameter]) -> torch.Tensor:
    """Return the total 2-norm of the gradients, as norm clipping measures it.

    Args:
        params: Parameters holding the gradients; those without one are skipped.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.zeros(())
    total = getattr(nn.utils, "get_total_norm", None)
    if total is not None:
        return total(grads)
    device = grads[0].device
    norms = [torch.linalg.vector_norm(g).to(device) for g in grads]
    return torch.linalg.vector_norm(torch.stack(norms))


def _clip_to_norm(
    params: list[nn.Parameter], max_norm: float, norm: torch.Tensor
) -> None:
    """Scale gradients of a known total norm down to a maximum, in place.

    Args:
        params: Parameters whose gradients are scaled.
        max_norm: Norm the gradients may have at most.
        norm: Their total norm.
    """
    clip = getattr(nn.utils, "clip_grads_with_norm_", None)
    if clip is not None:
        clip(params, max_norm, norm)
        return
    coefficient = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
    for grad in (p.grad for p in params if p.grad is not None):
        grad.mul_(coefficient.to(grad.device))


@dataclass(frozen=True, slots=True)
class UpdatePolicy:
    """Clipping, stepping and loss weighting for one optimizer.

    Attributes:
        clip: Threshold gradients are clipped to, or `None`.
        clip_mode: Whether `clip` bounds the gradient norm or each value.
        reduction: How the objective reduced over its batch.
        nonfinite: Action for non-finite gradients: `"raise"` stops the run,
            `"skip"` warns and leaves weights as is, `"ignore"` steps anyway.
    """

    clip: float | None = None
    clip_mode: ClipTypes = "norm"
    reduction: ReductionTypes = "mean"
    nonfinite: NonFiniteTypes = "raise"

    def __post_init__(self) -> None:
        """Validate the clipping, reduction and non-finite settings.

        Raises:
            ValueError: If `clip_mode`, `reduction` or `nonfinite` is unknown,
                or `clip` is not positive.
        """
        require(self.clip_mode, CLIP_MODES, "clip mode")
        require(self.reduction, ("mean", "sum"), "reduction")
        require(self.nonfinite, NONFINITE_MODES, "non-finite mode")
        if self.clip is not None and self.clip <= 0:
            raise ValueError(f"A clip threshold must be positive, got {self.clip!r}.")

    def share(self, loss: torch.Tensor, fraction: float) -> torch.Tensor:
        """Return the loss scaled by a micro-batch's share of the step.

        Args:
            loss: What the objective returned for this micro-batch.
            fraction: This micro-batch's samples over the step's samples.
        """
        return loss * fraction if self.reduction == "mean" else loss

    def clip_grads(
        self,
        params: Iterable[nn.Parameter],
        norm: Callable[[], torch.Tensor] | None = None,
    ) -> None:
        """Clip the gradients in place.

        Args:
            params: Parameters whose gradients are clipped.
            norm: Returns their total norm, computed once for whoever asks.
        """
        if self.clip is None:
            return
        params = list(params)
        if self.clip_mode == "value":
            nn.utils.clip_grad_value_(params, self.clip)
        else:
            _clip_to_norm(params, self.clip, norm() if norm else grad_norm(params))

    @staticmethod
    def finite_grads(
        params: Iterable[nn.Parameter],
        norm: Callable[[], torch.Tensor] | None = None,
    ) -> bool:
        """Whether every gradient is finite; a finite total norm proves it.

        Args:
            params: Parameters whose gradients are checked.
            norm: Returns their total norm, computed once for whoever asks.
        """
        grads = [p.grad for p in params if p.grad is not None]
        if not grads or (norm is not None and bool(torch.isfinite(norm()))):
            return True
        return bool(torch.stack([torch.isfinite(g).all() for g in grads]).all())

    def zero_grad(self, optimizer: Optimizer) -> None:
        """Zero the optimizer's gradients.

        Args:
            optimizer: The optimizer to clear.
        """
        optimizer.zero_grad(set_to_none=True)

    def step(
        self,
        optimizer: Optimizer,
        params: Iterable[nn.Parameter],
        closure: Callable[[], torch.Tensor] | None = None,
        norm: Callable[[], torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """Clip the gradients and step the optimizer.

        Args:
            optimizer: The optimizer to step.
            params: Parameters whose gradients are clipped.
            closure: Recomputes the loss, for optimizers that need it.
            norm: Returns the gradients' total norm, computed once.

        Raises:
            ValueError: If the optimizer needs a closure and none was given.
        """
        if closure is None:
            if OptimSpec.needs_closure(optimizer):
                raise ValueError(
                    f"{type(optimizer).__name__} re-evaluates the loss, so it "
                    "needs a closure to step through."
                )
            self.clip_grads(params, norm)
            return optimizer.step()
        return optimizer.step(closure)

    def closure_for(
        self,
        optimizer: Optimizer,
        params: Iterable[nn.Parameter],
        compute: Callable[[], torch.Tensor],
        check: Callable[[Callable[[], torch.Tensor]], None] | None = None,
    ) -> Callable[[], torch.Tensor]:
        """Return a closure that zeroes, computes, backpropagates and clips.

        Args:
            optimizer: The optimizer to clear each pass.
            params: Parameters whose gradients are clipped.
            compute: Returns the loss for the current batch.
            check: Called with the gradients' norm before they are clipped;
                raising abandons the step.
        """
        kept = list(params)

        def closure() -> torch.Tensor:
            self.zero_grad(optimizer)
            loss = compute()
            loss.backward()
            norm = functools.cache(lambda: grad_norm(kept))
            if check is not None:
                check(norm)
            self.clip_grads(kept, norm)
            return loss

        return closure


class Ema(AveragedModel):
    """Exponential moving average of a model's weights.

    Attributes:
        decay: Weight the average keeps of itself at each update.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: torch.device | str | None = None,
        use_buffers: bool = True,
    ):
        """Constructor.

        Args:
            model: Model whose weights are averaged.
            decay: Weight the average keeps of itself at each update.
            device: Device the average is kept on; the model's when absent.
            use_buffers: Whether buffers are averaged too. `False` leaves
                them at their first value until `update_bn` is called.

        Raises:
            ValueError: If `decay` is outside `[0, 1]`.
        """
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"A decay lies in [0, 1], got {decay!r}.")
        super().__init__(
            model,
            device=device,
            multi_avg_fn=get_ema_multi_avg_fn(decay),
            use_buffers=use_buffers,
        )
        self.decay = decay


class Swa(AveragedModel):
    """Equally weighted average of the models seen so far.

    Buffers are left alone by default: an equally weighted running average
    divides, which integer buffers such as a batch norm's `num_batches_tracked`
    cannot survive. Call `refresh_batch_stats` afterwards to restore them.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str | None = None,
        use_buffers: bool = False,
    ):
        """Constructor.

        Args:
            model: Model whose weights are averaged.
            device: Device the average is kept on; the model's when absent.
            use_buffers: Whether buffers are averaged too. `True` fails on
                integer buffers once more than one model has been averaged.
        """
        super().__init__(
            model,
            device=device,
            multi_avg_fn=get_swa_multi_avg_fn(),
            use_buffers=use_buffers,
        )

    @property
    def has_batch_norm(self) -> bool:
        """Whether the averaged weights carry batch statistics."""
        return any(
            isinstance(module, nn.modules.batchnorm._BatchNorm)
            for module in self.modules()
        )

    def refresh_batch_stats(
        self,
        batches: Iterable[Any],
        device: torch.device | str | None = None,
    ) -> None:
        """Recompute the batch statistics from a pass over the data.

        The average carries whichever running statistics the last iterate
        left behind, so a model with batch normalisation needs this pass
        before the averaged weights can be used.

        Args:
            batches: Model inputs to run through the averaged weights.
            device: Device to run the pass on.
        """
        update_bn(batches, self, device=device)


def average_targets(request: Any, default: str) -> dict[str, Any]:
    """Normalize an averaging request into a setting per binding name.

    Args:
        request: What to average: nothing, a flag, the binding to average,
            several of them, or a mapping carrying a setting per binding.
        default: Binding to average when the request names none.
    """
    if request is None or request is False:
        return {}
    if request is True:
        return {default: None}
    if isinstance(request, Mapping):
        return dict(request)
    if isinstance(request, str):
        return {request: None}
    if isinstance(request, (float, int, Ema)):
        return {default: request}
    return dict.fromkeys(request)
