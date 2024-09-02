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

from chuchichaestli.diffusion.base import NormalDistribution
from chuchichaestli.diffusion.ddpm import DDPM


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
        distr = NormalDistribution(0, scale)
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
        self, x_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Noise step for the diffusion process.

        Args:
            x_t: Tensor of shape (batch_size, *).

        Returns:
            Tuple of the sampled tensor, noise tensor and timesteps.
        """
        timesteps = self.sample_timesteps(x_t.shape[0])
        noise = self.sample_noise(x_t.shape)

        s_shape = [-1] + [1] * (x_t.dim() - 1)

        s1 = self.sqrt_alpha_cumprod[timesteps].reshape(s_shape)
        s2 = self.sqrt_1m_alpha_cumprod[timesteps].reshape(s_shape)
        return s1 * (x_t - self.mean) + s2 * noise, noise, timesteps

    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: int,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from the diffusion process.

        Args:
            x_t: Tensor of shape (batch_size, *).
            t: Current timestep.
            model_output: Output of the model at the current timestep.
            mean: Mean of the noise.
            scale: Scale of the noise.
        """
        coef_shape = [-1] + [1] * (x_t.dim() - 1)
        coef_inner_t = self.coef_inner[t].reshape(coef_shape)
        coef_outer_t = self.coef_outer[t].reshape(coef_shape)

        x_tm1 = coef_outer_t * (x_t - coef_inner_t * model_output)

        if t > 0:
            noise = self.sample_noise(x_t.shape)
            sigma_t = self.beta[t] ** 0.5
            return x_tm1 + sigma_t * noise

        return x_tm1 + self.mean
