# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Losses that weigh the high-frequency content of a reconstruction."""

import torch
from torch import nn

from chuchichaestli.dwt.modes import ExtensionModeTypes
from chuchichaestli.dwt.wavelet import Wavelet
from chuchichaestli.metrics.functional import charbonnier
from chuchichaestli.models.blur import GaussianBlurND
from chuchichaestli.models.dwt import MultilevelWaveletTransformND
from collections.abc import Callable


__all__ = ["GaussianLoss", "WaveletLoss"]


class WaveletLoss(nn.Module):
    """Charbonnier penalty on the detail subbands of a wavelet decomposition.

    An autoencoder trained on a pixel-wise loss alone tends to blur, because the
    detail bands carry little of the total energy. Decomposing both sides and
    penalizing only the detail bands weighs them independently of that energy.
    """

    def __init__(
        self,
        dimensions: int = 2,
        wavelet: str | Wavelet = "haar",
        mode: ExtensionModeTypes = "periodization",
        levels: int = 1,
        eps: float = 1e-3,
        reduction: Callable | None = torch.mean,
    ):
        """Constructor.

        Args:
            dimensions: Number of spatial dimensions.
            wavelet: Wavelet to decompose with, by name or as a `Wavelet`.
            mode: Signal extension mode.
            levels: Number of wavelet levels to penalize.
            eps: Constant rounding off the penalty at zero.
            reduction: Reduction function, e.g. `torch.mean` or `torch.sum`;
                `None` leaves the penalty unreduced.
        """
        super().__init__()
        self.dimensions = dimensions
        self.eps = eps
        self.reduction = reduction
        self.dwt = MultilevelWaveletTransformND(
            dimensions, wavelet, mode, "subband", levels=levels
        )

    @property
    def levels(self) -> int:
        """Number of wavelet levels the loss penalizes."""
        return self.dwt.levels

    def forward(
        self, data: torch.Tensor, prediction: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute the wavelet loss.

        Args:
            data: Observed data.
            prediction: Predicted data.
            kwargs: Additional keyword arguments, e.g. `reduction`.

        Returns:
            Loss value.

        Raises:
            ValueError: If the two inputs do not have the same shape.
        """
        if data.shape != prediction.shape:
            raise ValueError(
                f"Observation and prediction must have the same shape;"
                f" got {tuple(data.shape)} and {tuple(prediction.shape)}."
            )
        reduction = kwargs.get("reduction", self.reduction)
        channels = data.shape[1]
        # the approximation band leads every level, so the details follow it
        penalties = [
            charbonnier(
                observed[:, channels:], predicted[:, channels:], self.eps, None
            ).flatten(1)
            for observed, predicted in zip(
                self.dwt(data), self.dwt(prediction), strict=True
            )
        ]
        penalty = torch.cat(penalties, dim=1)
        return penalty if reduction is None else reduction(penalty)


class GaussianLoss(nn.Module):
    """Absolute error between the high-pass residuals of two images.

    Subtracting a Gaussian blur leaves what the blur removed, so comparing the
    two residuals weighs the fine detail on its own.
    """

    def __init__(
        self,
        dimensions: int = 2,
        kernel_size: int = 5,
        sigma: float = 1.0,
        reduction: Callable | None = torch.mean,
    ):
        """Constructor.

        Args:
            dimensions: Number of spatial dimensions.
            kernel_size: Number of taps of the Gaussian along each axis.
            sigma: Standard deviation of the Gaussian, in samples.
            reduction: Reduction function, e.g. `torch.mean` or `torch.sum`;
                `None` leaves the penalty unreduced.
        """
        super().__init__()
        self.dimensions = dimensions
        self.reduction = reduction
        self.blur = GaussianBlurND(dimensions, kernel_size, sigma)

    def forward(
        self, data: torch.Tensor, prediction: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute the high-frequency loss.

        Args:
            data: Observed data.
            prediction: Predicted data.
            kwargs: Additional keyword arguments, e.g. `reduction`.

        Returns:
            Loss value.

        Raises:
            ValueError: If the two inputs do not have the same shape.
        """
        if data.shape != prediction.shape:
            raise ValueError(
                f"Observation and prediction must have the same shape;"
                f" got {tuple(data.shape)} and {tuple(prediction.shape)}."
            )
        reduction = kwargs.get("reduction", self.reduction)
        residual = (data - self.blur(data)) - (prediction - self.blur(prediction))
        penalty = residual.abs()
        return penalty if reduction is None else reduction(penalty)
