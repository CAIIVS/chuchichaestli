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

import torch

SCHEDULES = {
    "linear": lambda beta_start, beta_end, num_timesteps, device: torch.linspace(
        beta_start, beta_end, num_timesteps, device=device
    ),
    "linear_scaled": lambda beta_start, beta_end, num_timesteps, device: torch.linspace(
        (1000 / num_timesteps) * beta_start,
        (1000 / num_timesteps) * beta_end,
        num_timesteps,
        device=device,
    ),
    "squared": lambda beta_min, beta_max, num_timesteps, device: beta_min
    + (beta_max - beta_min)
    * (torch.arange(num_timesteps, dtype=torch.float32, device=device) / num_timesteps)
    ** 2,
    "sigmoid": lambda beta_start, beta_end, num_timesteps, device: torch.sigmoid(
        torch.linspace(-6, 6, num_timesteps, device=device)
    )
    * (beta_end - beta_start)
    + beta_start,
    "cosine": lambda beta_min, beta_max, num_timesteps, device: beta_min
    + (beta_max - beta_min)
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
    ),
    "exponential": lambda beta_min, beta_max, num_timesteps, device: beta_min
    * (beta_max / beta_min)
    ** (
        torch.arange(num_timesteps, dtype=torch.float32, device=device) / num_timesteps
    ),
}


class DiffusionProcess(ABC):
    """Base class for diffusion processes."""

    def __init__(self, timesteps: int, device: str = "cpu") -> None:
        """Initialize the diffusion process."""
        self.num_time_steps = timesteps
        self.device = device

    @abstractmethod
    def noise_step(
        self, x_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Noise step for the diffusion process.

        Args:
            x_t: Tensor of shape (batch_size, *).

        Returns:
            Tuple of the sampled tensor, noise tensor and timesteps.
        """
        pass

    @abstractmethod
    def denoise_step(
        self, x_t: torch.Tensor, t: int, model_output: torch.Tensor
    ) -> torch.Tensor:
        """Sample from the diffusion process.

        Args:
            x_t: Tensor of shape (batch_size, *).
            t: Current timestep.
            model_output: Output of the model at the current timestep.
        """
        pass

    def sample_timesteps(self, n: int) -> torch.Tensor:
        """Sample timesteps for the diffusion process."""
        return torch.randint(0, self.num_time_steps, (n,), device=self.device)

    def sample_noise(self, shape: torch.Size) -> torch.Tensor:
        """Sample noise for the diffusion process.

        Args:
            shape: Shape of the noise tensor.
        """
        return torch.empty(shape, device=self.device).normal_()
