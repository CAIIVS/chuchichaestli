# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""The loss pairs a generator and its discriminator are trained with.

A variant is two formulas, not one: what the generator minimises and what the
discriminator minimises are not the same function.
"""

from __future__ import annotations
from collections.abc import Callable
from typing import Literal
import torch
from torch import nn


__all__ = [
    "AdversarialTypes",
    "ADV_GEN_LOSSES",
    "ADV_DISC_LOSSES",
]


AdversarialTypes = Literal["bce", "hinge", "least_squares", "wasserstein"]


def _bce_generator(fake: torch.Tensor) -> torch.Tensor:
    """Return the non-saturating loss pushing scores towards real.

    Args:
        fake: Scores the discriminator gave the model's output.
    """
    return nn.functional.binary_cross_entropy_with_logits(fake, torch.ones_like(fake))


def _bce_discriminator(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Return the loss separating real scores from produced ones.

    Args:
        real: Scores the discriminator gave the batch.
        fake: Scores it gave the model's output.
    """
    return nn.functional.binary_cross_entropy_with_logits(
        real, torch.ones_like(real)
    ) + nn.functional.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))


def _hinge_generator(fake: torch.Tensor) -> torch.Tensor:
    """Return the loss raising the scores of the model's output.

    Args:
        fake: Scores the discriminator gave the model's output.
    """
    return -fake.mean()


def _hinge_discriminator(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Return the margin loss separating real scores from produced ones.

    Args:
        real: Scores the discriminator gave the batch.
        fake: Scores it gave the model's output.
    """
    return nn.functional.relu(1 - real).mean() + nn.functional.relu(1 + fake).mean()


def _least_squares_generator(fake: torch.Tensor) -> torch.Tensor:
    """Return the squared distance from the model's scores to real.

    Args:
        fake: Scores the discriminator gave the model's output.
    """
    return ((fake - 1) ** 2).mean()


def _least_squares_discriminator(
    real: torch.Tensor, fake: torch.Tensor
) -> torch.Tensor:
    """Return the squared distances of both sides from their targets.

    Args:
        real: Scores the discriminator gave the batch.
        fake: Scores it gave the model's output.
    """
    return ((real - 1) ** 2).mean() + (fake**2).mean()


def _wasserstein_discriminator(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Return the estimated distance between the two distributions.

    The critic is only a valid estimate while it stays Lipschitz, which this
    does not enforce; pair it with weight clipping or a gradient penalty.

    Args:
        real: Scores the critic gave the batch.
        fake: Scores it gave the model's output.
    """
    return fake.mean() - real.mean()


ADV_GEN_LOSSES: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "bce": _bce_generator,
    "hinge": _hinge_generator,
    "least_squares": _least_squares_generator,
    "wasserstein": _hinge_generator,
}

ADV_DISC_LOSSES: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "bce": _bce_discriminator,
    "hinge": _hinge_discriminator,
    "least_squares": _least_squares_discriminator,
    "wasserstein": _wasserstein_discriminator,
}
