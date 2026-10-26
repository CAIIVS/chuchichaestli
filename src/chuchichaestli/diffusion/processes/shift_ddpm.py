# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Implementation of the Shifted Diffusion Probabilistic Model (ShiftDDPM) noise process."""

import torch
import torch.types
from torch import nn

from collections.abc import Callable, Generator

from chuchichaestli.diffusion.processes import DDPM


class ShiftDDPM(DDPM):
    """Shift DDPM noise process that improves conditioned DDPMs.

    Intended for the use if latent diffusion, but could probably be used for other tasks as well.

    As described in the paper:
    "ShiftDDPMs: Exploring Conditional Diffusion Models by Shifting Diffusion Trajectories"
    by Zhang et al. (2023);
    see https://arxiv.org/abs/2302.02373.
    """

    def __init__(
        self,
        num_timesteps: int,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cpu",
        schedule: str = "linear",
        E: nn.Module | None = None,
        **kwargs,
    ) -> None:
        """Initialize the ShiftDDPM algorithm.

        Args:
            num_timesteps: Number of time steps in the diffusion process.
            beta_start: Start value for beta.
            beta_end: End value for beta.
            device: Device to use for the computation.
            schedule: Schedule for beta.
            E: Prior shift predictor.
            kwargs: Passed on to `DDPM`, e.g. a `generator` to draw from.
        """
        super().__init__(
            num_timesteps, beta_start, beta_end, device, schedule, **kwargs
        )
        self.k = self.sqrt_alpha_cumprod * (1 - self.sqrt_alpha_cumprod)
        self.E = E

    def noise_step(
        self, x_0: torch.Tensor, condition: torch.Tensor | None = None, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A single step of the noise process.

        Args:
            x_0: Ground truth input tensor
            condition: Tensor of shape (batch_size, *).
            args: Additional arguments
            kwargs: Additional keyword

        Returns:
            Tuple of the noised tensor, noise, and the timestep.
        """
        E = self.E(condition)
        timesteps = self.sample_timesteps(x_0.shape[0])
        noise = self.sample_noise(x_0.shape)
        s_shape = [-1] + [1] * (x_0.dim() - 1)
        k = self.k[timesteps].reshape(s_shape)
        sqrt_alpha = self.sqrt_alpha_cumprod[timesteps].reshape(s_shape)
        sqrt_1m_alpha = self.sqrt_1m_alpha_cumprod[timesteps].reshape(s_shape)
        coef_inner = self.coef_inner[timesteps].reshape(s_shape)
        x_t = sqrt_alpha * x_0 + k * E + coef_inner * noise
        noise = (x_t - sqrt_alpha * x_0) / sqrt_1m_alpha
        return x_t, noise, timesteps

    def generate(
        self,
        model: Callable[..., torch.Tensor],
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
            E: Prior shift predictor.
            condition: Tensor to condition generation on. For unconditional generation, supply a zero-tensor of the sample shape.
            n: Number of samples to generate (batch size).
            yield_intermediate: Yield intermediate results. This turns the function into a generator.
            *args: Additional arguments.
            **kwargs: Additional keyword
        """
        if n > 1:
            condition = condition.repeat(n, 1)
        E = self.E(condition)
        x_t = self.sample_noise(condition.shape)

        for t in reversed(range(0, self.num_time_steps)):
            eps_t = model(torch.cat([condition, x_t], dim=1), t)
            z = self.sample_noise(x_t.shape) if t > 1 else 0.0
            s_t = self.k[t] * E
            s_tm1 = self.k[t - 1] * E
            x_tm1 = self.coef_outer[t] * (
                x_t - (self.beta[t] / self.sqrt_1m_alpha_cumprod[t]) * eps_t
            )
            x_tm1 -= (
                torch.sqrt(self.alpha[t])(1 - self.alpha_cumprod[t - 1])
                / (1 - self.alpha_cumprod[t])
            ) * s_t + s_tm1
            x_tm1 += (
                torch.sqrt(
                    ((1 - self.alpha_cumprod[t - 1]) / (1 - self.alpha_cumprod[t]))
                    * self.beta[t]
                )
                * z
            )
            x_t = x_tm1
            if yield_intermediate:
                yield x_t
        return x_t
