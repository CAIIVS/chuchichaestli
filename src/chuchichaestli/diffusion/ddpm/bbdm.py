# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Implementation of the Brownian Bridge Diffusion Model."""

from collections.abc import Generator
from typing import Any

import torch
from torch import Tensor

from chuchichaestli.diffusion.ddpm.base import DiffusionProcess
from chuchichaestli.diffusion.distributions import DistributionAdapter


class BBDM(DiffusionProcess):
    """Brownian Bridge Diffusion Model (BBDM).

    As described in the paper:
    "BBDM: Image-to-image Translation with Brownian Bridge Diffusion Models" by Li et al. (2022);
     see https://arxiv.org/abs/2205.07680.
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
        self,
        x_0: Tensor,
        condition: Tensor,
        timesteps: Tensor | None = None,
        *args,
        **kwargs,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Noise step for the Brownian Bridge Diffusion Model.

        Args:
            x_0: Clean input tensor.
            condition: Condition tensor.
            timesteps: Timesteps to use for the diffusion process.
            args: Additional positional arguments.
            kwargs: Additional keyword arguments.
        """
        if timesteps is not None:
            x_0 = (
                x_0.unsqueeze(0)
                .expand(len(timesteps), *x_0.shape)
                .reshape(-1, *x_0.shape[1:])
            )
            timesteps = timesteps.repeat_interleave(x_0.shape[0] // len(timesteps))
        else:
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
