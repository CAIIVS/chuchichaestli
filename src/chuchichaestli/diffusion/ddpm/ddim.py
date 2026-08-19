# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Implementation of the Denoising Diffusion Implicit Model."""

from collections.abc import Generator
from typing import Any
from chuchichaestli.diffusion.ddpm import DDPM

import torch
from torch import Tensor


class DDIM(DDPM):
    """Denoising Diffusion Implicit Model (DDIM).

    As described in the paper:
    "Denoising Diffusion Implicit Models" by Song et al. (2020);
    see https://arxiv.org/abs/2010.02502.
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
