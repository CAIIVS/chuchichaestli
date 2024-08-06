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

import torch
import torch.types

from chuchichaestli.diffusion.base import DiffusionProcess, SCHEDULES


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
    ) -> None:
        """Initialize the DDPM algorithm.

        Args:
            num_timesteps: Number of time steps in the diffusion process.
            beta_start: Start value for beta.
            beta_end: End value for beta.
            device: Device to use for the computation.
            schedule: Schedule for beta.
        """
        super().__init__(timesteps=num_timesteps, device=device)
        self.num_time_steps = num_timesteps
        self.beta = SCHEDULES[schedule](beta_start, beta_end, num_timesteps, device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_1m_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.coef_inner = (1 - self.alpha) / self.sqrt_1m_alpha_cumprod
        self.coef_outer = 1.0 / torch.sqrt(self.alpha)

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
        return s1 * x_t + s2 * noise, noise, timesteps

    def denoise_step(
        self, x_t: torch.Tensor, t: int, model_output: torch.Tensor
    ) -> torch.Tensor:
        """Sample from the diffusion process.

        Args:
            x_t: Tensor of shape (batch_size, *).
            t: Current timestep.
            model_output: Output of the model at the current timestep.
        """
        coef_shape = [-1] + [1] * (x_t.dim() - 1)
        coef_inner_t = self.coef_inner[t].reshape(coef_shape)
        coef_outer_t = self.coef_outer[t].reshape(coef_shape)

        x_tm1 = coef_outer_t * (x_t - coef_inner_t * model_output)

        if t > 0:
            noise = torch.empty_like(x_t).normal_()
            sigma_t = self.beta[t] ** 0.5
            return x_tm1 + sigma_t * noise

        return x_tm1
