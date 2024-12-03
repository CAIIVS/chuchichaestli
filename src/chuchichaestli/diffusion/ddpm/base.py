"""Base class for diffusion processes.

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
from typing import Any
from collections.abc import Generator

import torch

from chuchichaestli.diffusion.distributions import DistributionAdapter

SCHEDULES = {
    # A simple linear schedule from `beta_start` to `beta_end`
    "linear": lambda beta_start, beta_end, num_timesteps, device: torch.linspace(
        beta_start, beta_end, num_timesteps, device=device
    ),
    # Scaled linear schedule to account for different step sizes
    "linear_scaled": lambda beta_start, beta_end, num_timesteps, device: torch.linspace(
        (1000 / num_timesteps) * beta_start,
        (1000 / num_timesteps) * beta_end,
        num_timesteps,
        device=device,
    ),
    # A quadratic growth schedule from `beta_min` to `beta_max`
    "squared": lambda beta_min, beta_max, num_timesteps, device: beta_min
    + (
        (beta_max - beta_min)
        * (
            torch.arange(num_timesteps, dtype=torch.float32, device=device)
            / num_timesteps
        )
        ** 2
    ),
    # A sigmoid-shaped schedule to achieve a smoother transition from `beta_start` to `beta_end`
    "sigmoid": lambda beta_start, beta_end, num_timesteps, device: (
        torch.sigmoid(torch.linspace(-6, 6, num_timesteps, device=device))
        * (beta_end - beta_start)
        + beta_start
    ),
    # A cosine-shaped schedule for smoother transitions, emphasizing the start and the end
    "cosine": lambda beta_min, beta_max, num_timesteps, device: beta_min
    + (
        (beta_max - beta_min)
        * (
            1
            - torch.cos(
                (
                    torch.arange(num_timesteps, dtype=torch.float32, device=device)
                    / num_timesteps
                )
                * torch.pi
                / 2
            )
        )
    ),
    # An exponential schedule to smoothly increase the beta value over time
    "exponential": lambda beta_min, beta_max, num_timesteps, device: beta_min
    * (
        (beta_max / beta_min)
        ** (
            torch.arange(num_timesteps, dtype=torch.float32, device=device)
            / num_timesteps
        )
    ),
}


SAMPLERS = {
    "DDIM": "ddim",
    "default": "default",
}


class DiffusionProcess(ABC):
    """Base class for diffusion processes."""

    def __init__(
        self,
        timesteps: int,
        device: str = "cpu",
        noise_distribution: DistributionAdapter | None = None,
        generator: torch.Generator | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the diffusion process."""
        self.num_time_steps = timesteps
        self.device = device
        self.noise_distribution = noise_distribution
        self.generator = generator

    @abstractmethod
    def noise_step(
        self, x_t: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Noise step for the diffusion process.

        Args:
            x_t: Tensor of shape (batch_size, *).
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of the sampled tensor, noise tensor and timesteps.
        """
        pass

    @abstractmethod
    def generate(
        self,
        model: Any,
        condition: torch.Tensor | None = None,
        n: int = 1,
        yield_intermediate: bool = False,
        *args,
        **kwargs,
    ) -> Generator[torch.Tensor, None, torch.Tensor]:
        """Sample from the diffusion process.

        Samples n times the number of conditions from the diffusion process.

        Args:
            model: Model to use for sampling.
            condition: Tensor to condition generation on. For unconditional generation, set to None and use the shape parameter instead.
            n: Number of samples to generate (batch size).
            yield_intermediate: Yield intermediate results. This turns the function into a generator.
            *args: Additional arguments.
            **kwargs: Additional keyword
        """
        pass

    def sample_timesteps(self, n: int) -> torch.Tensor:
        """Sample timesteps for the diffusion process."""
        return torch.randint(0, self.num_time_steps, (n,), device=self.device)

    def sample_noise(self, shape: torch.Size) -> torch.Tensor:
        """Sample noise for the diffusion process.

        Args:
            shape: Shape of the noise tensor.
            mean: Mean of the noise tensor.
            scale: Scale of the noise tensor.
        """
        if self.noise_distribution is not None:
            return self.noise_distribution(shape)
        return torch.randn(shape, generator=self.generator, device=self.device)
