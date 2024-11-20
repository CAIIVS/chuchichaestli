"""Implementation of the Denoising Diffusion Implicit Model.

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
from chuchichaestli.diffusion.ddpm import DDPM

import torch
from torch import Tensor


class DDIM(DDPM):
    """Denoising Diffusion Implicit Model.

    See https://arxiv.org/abs/2010.02502.
    """

    def __init__(
        self,
        num_timesteps: int,
        num_sample_steps: int,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cpu",
        schedule: str = "linear",
        **kwargs,
    ) -> None:
        """Initialize the DDIM scheme.

        Args:
            num_timesteps: Number of timesteps.
            num_sample_steps: Number of timesteps to sample.
            beta_start: Start value for beta.
            beta_end: End value for beta.
            device: Device to use.
            schedule: Annealing schedule
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            num_timesteps, beta_start, beta_end, device, schedule, **kwargs
        )
        # Reverse the timesteps for sampling
        self.tau = torch.linspace(
            self.num_time_steps - 1, 0, num_sample_steps, dtype=int
        ).to(device)  # Select timesteps in reverse order

        # Precompute alpha_cumprod_prev
        self.alpha_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alpha_cumprod[:-1]]
        )
        self.sqrt_alpha_cumprod_prev = torch.sqrt(self.alpha_cumprod_prev)
        self.sqrt_1m_alpha_cumprod_prev = torch.sqrt(1.0 - self.alpha_cumprod_prev)

    def generate(
        self,
        model: Any,
        condition: torch.Tensor | None = None,
        shape: tuple[int, ...] | None = None,
        n: int = 1,
        yield_intermediate: bool = False,
        *args,
        **kwargs,
    ) -> Tensor | Generator[Tensor, None, torch.Tensor]:
        """Sample from the DDPM model using DDIM."""
        if condition is not None:
            c = (
                condition.unsqueeze(0)
                .expand(n, *condition.shape)
                .reshape(-1, *condition.shape[1:])
            )
            x_tau = self.sample_noise(c.shape).to(self.device)
        elif shape is not None:
            x_tau = self.sample_noise((n, *shape)).to(self.device)
        else:
            raise ValueError("Either condition or shape must be provided.")
        coef_shape = [-1] + [1] * (x_tau.dim() - 1)

        for idx, t in enumerate(self.tau):
            t = int(t)
            if condition is not None:
                eps_tau = model(torch.cat([c, x_tau], dim=1), t)
            else:
                eps_tau = model(x_tau, t)
            x_0_hat = (
                x_tau - self.sqrt_1m_alpha_cumprod[t] * eps_tau
            ) / self.sqrt_alpha_cumprod[t].reshape(coef_shape)

            x_tau = (
                self.sqrt_alpha_cumprod_prev[t] * x_0_hat
                + self.sqrt_1m_alpha_cumprod_prev[t] * eps_tau
            )
            if yield_intermediate:
                yield x_tau

        yield x_tau
