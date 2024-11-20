"""Implementation of the Diffusion Probabilistic Model (DDPM) noise process as described in the paper.

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

from typing import Any
from collections.abc import Generator

import torch
import torch.types

from chuchichaestli.diffusion.ddpm.base import DiffusionProcess, SCHEDULES


class DDPM(DiffusionProcess):
    """Diffusion Probabilistic Model (DDPM) noise process.

    The DDPM noise process is described in the paper "Denoising Diffusion Probabilistic Models" by Ho et al.
    See https://arxiv.org/abs/2006.11239.
    """

    def __init__(
        self,
        num_timesteps: int,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu",
        schedule: str = "linear",
        **kwargs,
    ) -> None:
        """Initialize the DDPM algorithm.

        Args:
            num_timesteps: Number of time steps in the diffusion process.
            beta_start: Start value for beta.
            beta_end: End value for beta.
            device: Device to use for the computation.
            schedule: Schedule for beta.
            kwargs: Additional keyword arguments.
        """
        super().__init__(timesteps=num_timesteps, device=device, **kwargs)
        self.num_time_steps = num_timesteps
        self.beta = SCHEDULES[schedule](beta_start, beta_end, num_timesteps, device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_1m_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.coef_inner = (1 - self.alpha) / self.sqrt_1m_alpha_cumprod
        self.coef_outer = 1.0 / torch.sqrt(self.alpha)

    def noise_step(
        self, x_t: torch.Tensor, condition: torch.Tensor | None = None, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Noise step for the diffusion process.

        Args:
            x_t: Tensor of shape (batch_size, *).
            condition: Tensor of shape (batch_size, *).
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of the sampled tensor, noise tensor and timesteps.
        """
        timesteps = self.sample_timesteps(x_t.shape[0])
        noise = self.sample_noise(x_t.shape)

        s_shape = [-1] + [1] * (x_t.dim() - 1)

        s1 = self.sqrt_alpha_cumprod[timesteps].reshape(s_shape)
        s2 = self.sqrt_1m_alpha_cumprod[timesteps].reshape(s_shape)

        x_t = s1 * x_t + s2 * noise
        if condition is not None:
            x_t = torch.cat([condition, x_t], dim=1)

        return x_t, noise, timesteps

    def generate(
        self,
        model: Any,
        condition: torch.Tensor | None = None,
        shape: tuple[int, ...] | None = None,
        n: int = 1,
        yield_intermediate: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor | Generator[torch.Tensor, None, torch.Tensor]:
        """Sample from the diffusion process.

        Samples n times the number of conditions from the diffusion process.

        Args:
            model: Model to use for sampling.
            condition: Tensor to condition generation on. For unconditional generation, set to None and use the shape parameter instead.
            shape: Shape of the generated tensor. Only used if condition is None.
            n: Number of samples to generate (batch size).
            yield_intermediate: Yield intermediate results. This turns the function into a generator.
            *args: Additional arguments.
            **kwargs: Additional keyword
        """
        if condition is not None:
            c = (
                condition.unsqueeze(0)
                .expand(n, *condition.shape)
                .reshape(-1, *condition.shape[1:])
            )
            x_t = self.sample_noise(c.shape)
        elif shape is not None:
            x_t = self.sample_noise((n,) + shape)
        else:
            raise ValueError("Either condition or shape must be provided.")
        coef_shape = [-1] + [1] * (x_t.dim() - 1)

        for t in reversed(range(0, self.num_time_steps)):
            if condition is not None:
                eps_t = model(torch.cat([c, x_t], dim=1), t)
            else:
                eps_t = model(x_t, t)
            coef_inner_t = self.coef_inner[t].reshape(coef_shape)
            coef_outer_t = self.coef_outer[t].reshape(coef_shape)
            x_tm1 = coef_outer_t * (x_t - coef_inner_t * eps_t)

            if t > 0:
                noise = self.sample_noise(x_t.shape)
                sigma_t = self.beta[t] ** 0.5
                x_t = x_tm1 + sigma_t * noise
                if yield_intermediate:
                    yield x_t

        yield x_tm1
