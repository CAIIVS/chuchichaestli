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
    ) -> torch.Tensor:
        """Sample noise from the distribution."""
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
    ) -> torch.Tensor:
        """Sample noise from the distribution."""
        return torch.randn(shape, device=self.device).abs() * self.scale + self.mean


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
    ) -> torch.Tensor:
        """Sample noise from the distribution."""
        noise = torch.randn(shape, device=self.device)
        return noise * self.scale + self.mean
