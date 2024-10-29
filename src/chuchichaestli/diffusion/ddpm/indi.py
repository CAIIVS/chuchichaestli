"""Implementation of Inversion by Direct Iteration (InDI).

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

from collections.abc import Generator
from typing import Any

import torch

from chuchichaestli.diffusion.ddpm.base import DiffusionProcess


class InDI(DiffusionProcess):
    """Degradation process for inversion by direct iteration (InDI).

    The InDI process is described in the paper "Inversion by Direct Iteration: An Alternative to
    Denoising Diffusion for Image Restoration" by Delbracio and Milanfar.
    See https://arxiv.org/abs/2303.11435.
    """

    def __init__(
        self,
        num_timesteps: int,
        epsilon: float | torch.Tensor = 0.01,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        """Initialize the InDI algorithm.

        Args:
            num_timesteps: Number of time steps in the diffusion process.
            epsilon: Noise level. Can be a scalar or a tensor of shape (num_timesteps,).
                For eps_t = eps_0 / sqrt(t), the noise perturbation is a pure Brownian motion.
                For eps = eps_0 = cst, the InDI scheme is recovered.
            device: Device to use for the computation.
            kwargs: Additional keyword arguments.

        """
        super().__init__(timesteps=num_timesteps, device=device, **kwargs)
        self.num_time_steps = num_timesteps
        self.delta = 1.0 / num_timesteps
        if isinstance(epsilon, float):
            self.epsilon = torch.full((num_timesteps,), epsilon, device=device)
        else:
            self.epsilon = epsilon
            self.epsilon.to(device)

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample timesteps for the InDI diffusion process.

        Args:
            batch_size: Number of samples to generate.

        Returns:
            Tensor of shape (batch_size,) with the sampled timesteps.
        """
        return (
            torch.randint(0, self.num_time_steps, (batch_size,), device=self.device)
            * self.delta
        )

    def noise_step(
        self, x: torch.Tensor, y: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Noise step for the diffusion process.

        Args:
            x: High quality sample, tensor of shape (batch_size, *).
            y: Corresponding low quality sample, tensor of shape (batch_size, *).
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of the sampled tensor, noise tensor and timesteps.
        """
        timesteps = self.sample_timesteps(x.shape[0])
        timesteps = timesteps.view(-1, *([1] * (x.dim() - 1)))

        x_t = (1 - timesteps) * x + timesteps * y
        # TODO: Verify that this is correct.
        noise = x_t - x

        return x_t, noise, timesteps.view(-1)

    def generate(
        self,
        model: Any,
        condition: torch.Tensor,
        n: int = 1,
        yield_intermediate: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor | Generator[torch.Tensor, None, torch.Tensor]:
        """Sample from the diffusion process.

        Samples n times the number of conditions from the diffusion process.

        Args:
            model: Model to use for sampling.
            condition: Tensor to condition generation on. For unconditional generation, supply a zero-tensor of the sample shape.
            n: Number of samples to generate (batch size).
            yield_intermediate: Yield intermediate results. This turns the function into a generator.
            *args: Additional arguments.
            **kwargs: Additional keyword
        """
        c = (
            condition.unsqueeze(0)
            .expand(n, *condition.shape)
            .reshape(-1, *condition.shape[1:])
        )

        x_t = c + self.epsilon[-1] * self.sample_noise(c.shape)
        for i in reversed(range(1, self.num_time_steps)):
            eps_t = self.epsilon[i]
            eps_tmdelta = self.epsilon[i - 1]

            t = i / self.num_time_steps
            noise = self.sample_noise(c.shape)

            x_tmdelta = (
                (self.delta / t) * model(x_t, i)
                + (1 - self.delta / t) * x_t
                + (t - self.delta) * torch.sqrt(eps_tmdelta**2 - eps_t**2) * noise
            )
            x_t = x_tmdelta
            if yield_intermediate:
                yield x_t

        yield x_tmdelta
