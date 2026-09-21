# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gradient clipping, optimizer stepping and weight averaging."""

from __future__ import annotations
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.swa_utils import (
    AveragedModel,
    get_ema_multi_avg_fn,
    get_swa_multi_avg_fn,
)
from chuchichaestli.training.optim import OptimSpec
from chuchichaestli.utils.registry import require


__all__ = [
    "ClipTypes",
    "ReductionTypes",
    "CLIP_FUNCTIONS",
    "UpdatePolicy",
    "Ema",
    "Swa",
]


ClipTypes = Literal["norm", "value"]
ReductionTypes = Literal["mean", "sum"]

CLIP_FUNCTIONS: dict[str, Callable[..., torch.Tensor | None]] = {
    "norm": nn.utils.clip_grad_norm_,
    "value": nn.utils.clip_grad_value_,
}


@dataclass(frozen=True, slots=True)
class UpdatePolicy:
    """Clipping, stepping and loss weighting for one optimizer.

    Attributes:
        clip: Threshold gradients are clipped to, or `None`.
        clip_mode: Whether `clip` bounds the gradient norm or each value.
        reduction: How the objective reduced over its batch.
    """

    clip: float | None = None
    clip_mode: ClipTypes = "norm"
    reduction: ReductionTypes = "mean"

    def __post_init__(self) -> None:
        """Validate the clipping and reduction settings.

        Raises:
            ValueError: If `clip_mode` or `reduction` is unknown, or `clip`
                is not positive.
        """
        require(self.clip_mode, CLIP_FUNCTIONS, "clip mode")
        require(self.reduction, ("mean", "sum"), "reduction")
        if self.clip is not None and self.clip <= 0:
            raise ValueError(f"A clip threshold must be positive, got {self.clip!r}.")

    def share(self, loss: torch.Tensor, fraction: float) -> torch.Tensor:
        """Return the loss scaled by a micro-batch's share of the step.

        Args:
            loss: What the objective returned for this micro-batch.
            fraction: This micro-batch's samples over the step's samples.
        """
        return loss * fraction if self.reduction == "mean" else loss

    def clip_grads(self, params: Iterable[nn.Parameter]) -> None:
        """Clip the gradients in place.

        Args:
            params: Parameters whose gradients are clipped.
        """
        if self.clip is None:
            return
        CLIP_FUNCTIONS[self.clip_mode](list(params), self.clip)

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
    ) -> torch.Tensor | None:
        """Clip the gradients and step the optimizer.

        Args:
            optimizer: The optimizer to step.
            params: Parameters whose gradients are clipped.
            closure: Recomputes the loss, for optimizers that need it.

        Raises:
            ValueError: If the optimizer needs a closure and none was given.
        """
        if closure is None:
            if OptimSpec.needs_closure(optimizer):
                raise ValueError(
                    f"{type(optimizer).__name__} re-evaluates the loss, so it "
                    "needs a closure to step through."
                )
            self.clip_grads(params)
            return optimizer.step()
        return optimizer.step(closure)

    def closure_for(
        self,
        optimizer: Optimizer,
        params: Iterable[nn.Parameter],
        compute: Callable[[], torch.Tensor],
    ) -> Callable[[], torch.Tensor]:
        """Return a closure that zeroes, computes, backpropagates and clips.

        Args:
            optimizer: The optimizer to clear each pass.
            params: Parameters whose gradients are clipped.
            compute: Returns the loss for the current batch.
        """
        kept = list(params)

        def closure() -> torch.Tensor:
            self.zero_grad(optimizer)
            loss = compute()
            loss.backward()
            self.clip_grads(kept)
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
    cannot survive. Call `update_bn` afterwards to restore them.
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
