# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Auxiliary classes for sampling noise from different distributions."""

from abc import ABC, abstractmethod
import torch


class DistributionAdapter(ABC):
    """Base class for distribution adapters."""

    def __init__(self, device: str = "cpu") -> None:
        """Initialize the distribution adapter."""
        self.device = device

    @abstractmethod
    def __call__(
        self,
        shape: torch.Size,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample noise from the distribution.

        Args:
            shape: Shape of the noise tensor.
            generator: Draws from this rather than the global RNG, so a
                process owning one stays reproducible through its noise.
        """
        pass


class HalfNormalDistribution(DistributionAdapter):
    """Half normal distribution adapter."""

    def __init__(
        self,
        mean: float | torch.Tensor,
        scale: float | torch.Tensor = 1.0,
        device: str = "cpu",
    ) -> None:
        """Initialize the half normal distribution adapter."""
        super().__init__(device)
        self.mean = torch.tensor(mean, device=device)
        self.scale = torch.tensor(scale, device=device)

    def __call__(
        self,
        shape: torch.Size,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample noise from the distribution.

        Args:
            shape: Shape of the noise tensor.
            generator: Draws from this rather than the global RNG.
        """
        noise = torch.randn(shape, generator=generator, device=self.device)
        return noise.abs() * self.scale + self.mean


class NormalDistribution(DistributionAdapter):
    """Normal distribution adapter."""

    def __init__(
        self,
        mean: float | torch.Tensor,
        scale: float | torch.Tensor = 1.0,
        device: str = "cpu",
    ) -> None:
        """Initialize the normal distribution adapter."""
        super().__init__(device)
        self.mean = mean
        self.scale = scale

    def __call__(
        self,
        shape: torch.Size,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample noise from the distribution.

        Args:
            shape: Shape of the noise tensor.
            generator: Draws from this rather than the global RNG.
        """
        noise = torch.randn(shape, generator=generator, device=self.device)
        return noise * self.scale + self.mean
