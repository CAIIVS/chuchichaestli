"""Implementation of the classifier-free conditional diffusion probabilistic model (CF-DDPM) noise process.

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
import torch.types

from chuchichaestli.diffusion.ddpm.base import DiffusionProcess

from typing import Any
from collections.abc import Generator


class CFGDDPM(DiffusionProcess):
    """Classifier-free conditional diffusion probabilistic model (CF-DDPM).

    See https://arxiv.org/abs/2207.12598.
    """

    def __init__(
        self,
        num_timesteps: int,
        beta_start: float = -20,
        beta_end: float = 20,
        uncond_prob: float = 0.1,
        guidance_strength: float = 0.0,
        noise_interpolation_coeff: float = 0.3,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        """Initialize the DDPM algorithm.

        A note on parameter selection:
        - num_timesteps = 256 and w = 0.3 performs best in terms of FID
        - num_timesteps = 128 and w = 4.0 performs best in terms of IS.
        - uncond_prob in [0.1, 0.2] is always better than 0.5.

        Args:
            num_timesteps: Number of time steps in the diffusion process.
            beta_start: Start value for beta. This corresponds to lambda_min in the paper. Naming retained for consistency with other schedules.
            beta_end: End value for beta. This corresponds to lambda_max in the paper. Naming retained for consistency with other schedules.
            uncond_prob: Probability of dropping the condition (p_uncond in paper).
            guidance_strength: Strength of the guidance signal (w in paper). 0 means no guidance in generation.
            noise_interpolation_coeff: Coefficient for noise interpolation (v in paper).
            device: Device to use for the computation.
            kwargs: Additional keyword arguments.
        """
        super().__init__(timesteps=num_timesteps, device=device, **kwargs)
        beta_start = torch.tensor(beta_start, device=device)
        beta_end = torch.tensor(beta_end, device=device)
        self.b = torch.arctan(torch.exp(-beta_end / 2))
        self.a = torch.arctan(torch.exp(-beta_start / 2)) - self.b
        self.p = torch.tensor(uncond_prob, device=device)
        self.w = torch.tensor(guidance_strength, device=device)
        self.v = torch.tensor(noise_interpolation_coeff, device=device)
        self.u_generation = torch.linspace(0, 1, num_timesteps, device=device)

    def noise_step(
        self,
        x_t: torch.Tensor,
        condition: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Noise step for the diffusion process.

        Args:
            x_t: Tensor of shape (batch_size, *).
            condition: Tensor of shape (batch_size, *).
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple of the sampled tensor, noise tensor and timesteps.
        """
        u = torch.rand(x_t.shape, device=self.device)
        lam = -2 * torch.log(torch.tan(self.a * u + self.b))
        alpha_square = torch.sigmoid(lam)
        alpha = torch.sqrt(alpha_square)
        sigma = torch.sqrt(1 - alpha_square)
        noise = self.sample_noise(x_t.shape)

        zeros = torch.zeros_like(condition)
        condition = torch.where(
            torch.bernoulli(self.p * torch.ones_like(x_t)).type(torch.bool),
            zeros,
            condition,
        )
        x_t = alpha * x_t + sigma * noise
        x_t = torch.cat([condition, x_t], dim=1)

        return x_t, noise, lam

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
        zeros = torch.zeros_like(c)
        z_t = self.sample_noise(c.shape)

        for t in reversed(range(1, self.num_time_steps)):
            lambda_t = -2 * torch.log(torch.tan(self.a * self.u_generation[t] + self.b))
            lambda_tp1 = -2 * torch.log(
                torch.tan(self.a * self.u_generation[t - 1] + self.b)
            )
            eps_t = model(torch.cat([zeros, z_t], dim=1), t)
            ceps_t = model(torch.cat([c, z_t], dim=1), t)
            eps_tilde_t = (1 + self.w) * ceps_t - self.w * eps_t
            alpha_lambda_t = self.alpha(lambda_t)
            sigma_lambda_t = self.sigma(lambda_t)
            sigma_lambda_tp1 = self.sigma(lambda_tp1)
            x_tilde_t = (z_t - sigma_lambda_t * eps_tilde_t) / alpha_lambda_t
            if yield_intermediate:
                yield x_tilde_t
            if t < self.num_time_steps:
                mu_tilde = (
                    torch.exp(lambda_t - lambda_tp1)
                    * (self.alpha(lambda_tp1) / self.alpha(lambda_t))
                    * z_t
                ) + (1 - torch.exp(lambda_t - lambda_tp1)) * self.alpha(
                    lambda_tp1
                ) * x_tilde_t

                sigma_tilde_squared = (
                    1 - torch.exp(lambda_t - lambda_tp1)
                ) * sigma_lambda_tp1**2

                sigma_lambda_t_reverse = (
                    1 - torch.exp(lambda_t - lambda_tp1)
                ) * sigma_lambda_t**2

                print(
                    lambda_t,
                    lambda_tp1,
                    lambda_t - lambda_tp1,
                    torch.exp(lambda_t - lambda_tp1),
                    1 - torch.exp(lambda_t - lambda_tp1),
                )

                std = (
                    sigma_tilde_squared ** (1 - self.v) * sigma_lambda_t_reverse**self.v
                )
                z_t = torch.normal(mu_tilde, std)

        yield x_tilde_t

    @staticmethod
    def alpha(lambda_t: torch.Tensor) -> torch.Tensor:
        """Compute alpha from lambda."""
        alpha_square = torch.sigmoid(lambda_t)
        return torch.sqrt(alpha_square)

    @staticmethod
    def sigma(lambda_t: torch.Tensor) -> torch.Tensor:
        """Compute sigma from lambda."""
        alpha_square = torch.sigmoid(lambda_t)
        return torch.sqrt(1 - alpha_square)
