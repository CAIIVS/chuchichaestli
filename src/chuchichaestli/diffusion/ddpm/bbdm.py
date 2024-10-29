"""Implementation of the Brownian Bridge Diffusion Model.

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
from torch import Tensor

from chuchichaestli.diffusion.ddpm.base import DiffusionProcess
from chuchichaestli.diffusion.distributions import DistributionAdapter


class BBDM(DiffusionProcess):
    """Brownian Bridge Diffusion Model.

    C.f. https://openaccess.thecvf.com/content/CVPR2023/papers/Li_BBDM_Image-to-Image_Translation_With_Brownian_Bridge_Diffusion_Models_CVPR_2023_paper.pdf
    """

    def __init__(
        self,
        num_timesteps: int,
        s: float = 1.0,
        device: str = "cpu",
        noise_distribution: DistributionAdapter | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the Brownian Bridge Diffusion Model.

        Args:
            num_timesteps: Number of time steps in the diffusion process.
            s: Scaling factor for the Brownian Bridge.
            device: Device to use for the computation.
            noise_distribution: Noise distribution to use for the diffusion process.
            args: Additional positional arguments.
            kwargs: Additional keyword arguments.
        """
        super().__init__(num_timesteps, device, noise_distribution, *args, **kwargs)
        self.s = s

    def noise_step(
        self, x_0: Tensor, condition: Tensor, *args, **kwargs
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Noise step for the Brownian Bridge Diffusion Model.

        Args:
            x_0: Clean input tensor.
            condition: Condition tensor.
            args: Additional positional arguments.
            kwargs: Additional keyword arguments.
        """
        timesteps = self.sample_timesteps(x_0.shape[0])
        noise = self.sample_noise(x_0.shape)
        s_shape = [-1] + [1] * (x_0.dim() - 1)
        m_t = (timesteps / self.num_time_steps).reshape(s_shape)
        delta_t = 2 * self.s * (m_t - m_t**2).reshape(s_shape)

        print(m_t.shape, x_0.shape, condition.shape, delta_t.shape, noise.shape)
        return (
            (1 - m_t) * x_0 + m_t * condition + torch.sqrt(delta_t) * noise,
            noise,
            timesteps,
        )

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
        x_t = c
        coef_shape = [x_t.shape[0]] + [1] * (x_t.dim() - 1)
        for t in reversed(range(1, self.num_time_steps + 1)):
            z = self.sample_noise(x_t.shape) if t > 1 else torch.zeros_like(x_t)
            tt = torch.full(coef_shape, t, device=self.device)
            m_t = tt / self.num_time_steps
            m_tm1 = (tt - 1) / self.num_time_steps
            delta_t = 2 * self.s * (m_t - m_t**2)
            delta_tm1 = 2 * self.s * (m_tm1 - m_tm1**2)
            delta_t_tm1 = (
                delta_t - delta_tm1 * ((1 - m_t) ** 2 / (1 - m_tm1) ** 2)
                if t < self.num_time_steps
                else torch.zeros_like(tt)
            )
            delta_tilde_t = (
                (delta_t_tm1 - delta_tm1) / delta_t
                if t < self.num_time_steps
                else torch.zeros_like(tt)
            )
            c_xt = (
                (delta_tm1 / delta_t) * ((1 - m_t) / (1 - m_tm1))
                + (delta_t_tm1 / delta_t) * (1 - m_tm1)
                if t < self.num_time_steps
                else torch.zeros_like(tt)
            )
            c_yt = (
                m_tm1 - m_t * ((1 - m_t) / (1 - m_tm1)) * (delta_tm1 / delta_t)
                if t < self.num_time_steps
                else m_tm1
            )
            c_et = (
                (1 - m_t) * (delta_t_tm1 / delta_t)
                if t < self.num_time_steps
                else torch.zeros_like(tt)
            )
            x_t = (
                c_xt * x_t
                + c_yt * c
                + c_et * model(x_t, t)
                + torch.sqrt(delta_tilde_t) * z
            )
            if yield_intermediate:
                yield x_t
        yield x_t
