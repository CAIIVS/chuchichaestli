"""Auxiliary classes for sampling noise from different distributions.

This file is part of Chuchichaestli.

Chuchichaestli is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Chuchichaestli is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Chuchichaestli.  If not, see <http://www.gnu.org/licenses/>.

Developed by the Intelligent Vision Systems Group at ZHAW.
"""

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
