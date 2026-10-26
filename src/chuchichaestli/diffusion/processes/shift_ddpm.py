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

    Intended for use in latent diffusion, but could probably be used for other tasks as well.

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
            E: Prior shift predictor; the condition itself when absent.
            kwargs: Passed on to `DDPM`, e.g. a `generator` to draw from.
        """
        super().__init__(
            num_timesteps, beta_start, beta_end, device, schedule, **kwargs
        )
        self.k = self.sqrt_alpha_cumprod * (1 - self.sqrt_alpha_cumprod)
        # the step before the first has no noise and no shift
        self.alpha_cumprod_prev = torch.cat(
            [self.alpha_cumprod.new_ones(1), self.alpha_cumprod[:-1]]
        )
        self.k_prev = torch.cat([self.k.new_zeros(1), self.k[:-1]])
        self.E = E if E is not None else nn.Identity()

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
            Tuple of the noised tensor, the noise including the shift (what the model
            learns to predict), and the timestep.

        Raises:
            ValueError: If no condition is given.
        """
        if condition is None:
            raise ValueError("A ShiftDDPM shifts by a condition, so it needs one.")
        E = self.E(condition)
        timesteps = self.sample_timesteps(x_0.shape[0])
        noise = self.sample_noise(x_0.shape)
        s_shape = [-1] + [1] * (x_0.dim() - 1)
        k = self.k[timesteps].reshape(s_shape)
        sqrt_alpha = self.sqrt_alpha_cumprod[timesteps].reshape(s_shape)
        sqrt_1m_alpha = self.sqrt_1m_alpha_cumprod[timesteps].reshape(s_shape)
        x_t = sqrt_alpha * x_0 + k * E + sqrt_1m_alpha * noise
        noise = (x_t - sqrt_alpha * x_0) / sqrt_1m_alpha
        return x_t, noise, timesteps

    def generate(
        self,
        model: Callable[..., torch.Tensor],
        condition: torch.Tensor | None = None,
        shape: tuple[int, ...] | None = None,
        n: int = 1,
        yield_intermediate: bool = False,
        *args,
        **kwargs,
    ) -> Generator[torch.Tensor, None, None]:
        """Sample from the diffusion process, shifted by the condition.

        Samples n times the number of conditions from the diffusion process.

        Args:
            model: Model to use for sampling; it sees the noised sample only.
            condition: Tensor the trajectory is shifted by, through `E`.
            shape: Unused; a shifted process takes its shape from the condition.
            n: Number of samples to generate per condition.
            yield_intermediate: Yield each intermediate sample before the final one.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If no condition is given.
        """
        if condition is None:
            raise ValueError("A ShiftDDPM shifts by a condition, so it needs one.")
        c = condition.unsqueeze(0).expand(n, *condition.shape)
        c = c.reshape(-1, *condition.shape[1:])
        E = self.E(c)
        # the reverse process starts from N(s_T, I), as the forward one ends there
        x_t = self.k[-1] * E + self.sample_noise(c.shape)
        coef_shape = [-1] + [1] * (x_t.dim() - 1)

        for t in reversed(range(0, self.num_time_steps)):
            eps_t = model(x_t, t)
            alpha_prev = self.alpha_cumprod_prev[t]
            coef_shift = (
                self.alpha[t].sqrt() * (1 - alpha_prev) / (1 - self.alpha_cumprod[t])
            )
            coef_inner_t = self.coef_inner[t].reshape(coef_shape)
            coef_outer_t = self.coef_outer[t].reshape(coef_shape)
            x_tm1 = (
                coef_outer_t * (x_t - coef_inner_t * eps_t)
                - coef_shift * self.k[t] * E
                + self.k_prev[t] * E
            )

            if t > 0:
                noise = self.sample_noise(x_t.shape)
                variance = (1 - alpha_prev) / (1 - self.alpha_cumprod[t]) * self.beta[t]
                x_t = x_tm1 + variance.sqrt() * noise
                if yield_intermediate:
                    yield x_t

        yield x_tm1
