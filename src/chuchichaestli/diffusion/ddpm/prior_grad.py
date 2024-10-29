"""Implementation of PriorGrad.

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

import torch
from typing import Any
from collections.abc import Generator

from chuchichaestli.diffusion.distributions import NormalDistribution
from chuchichaestli.diffusion.ddpm.ddpm import DDPM


class PriorGrad(DDPM):
    """PriorGrad noise process.

    Implementation of "PriorGrad: Improving conditional denoising diffusion models with data-dependent adataptive prior"
    by Lee et al. (see https://arxiv.org/abs/2106.06406)
    """

    def __init__(
        self,
        mean: float | torch.Tensor,
        scale: float | torch.Tensor,
        num_timesteps: int,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu",
        schedule: str = "linear",
        **kwargs,
    ):
        """Initialize the PriorGrad algorithm.

        Args:
            mean: Mean value of the noise process.
            scale: Scale value of the noise process.
            num_timesteps: Number of time steps in the diffusion process.
            beta_start: Start value for beta.
            beta_end: End value for beta.
            device: Device to use for the computation.
            schedule: Schedule for beta.
            kwargs: Additional keyword arguments.
        """
        scale = scale.to(device)
        mean = mean.to(device)
        distr = NormalDistribution(0, scale, device=device)
        super().__init__(
            noise_distribution=distr,
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device,
            schedule=schedule,
            **kwargs,
        )
        self.mean = mean

    def noise_step(
        self, x_t: torch.Tensor, condition: torch.Tensor, *args, **kwargs
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

        x_t = s1 * (x_t - self.mean) + s2 * noise

        return torch.cat([condition, x_t], dim=1), noise, timesteps

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
        x_t = self.sample_noise(c.shape)
        coef_shape = [-1] + [1] * (x_t.dim() - 1)

        for t in reversed(range(0, self.num_time_steps)):
            eps_t = model(torch.cat([c, x_t], dim=1), t)
            coef_inner_t = self.coef_inner[t].reshape(coef_shape)
            coef_outer_t = self.coef_outer[t].reshape(coef_shape)
            x_tm1 = coef_outer_t * (x_t - coef_inner_t * eps_t)

            if t > 0:
                noise = self.sample_noise(x_t.shape)
                sigma_t = self.beta[t] ** 0.5
                x_t = x_tm1 + sigma_t * noise
                if yield_intermediate:
                    yield x_t

        yield x_tm1 + self.mean
