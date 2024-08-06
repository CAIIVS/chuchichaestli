"""Implementation of the ILVR algorithm for conditioned sampling from unconditioned DDPMs.

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
import torch.nn.functional as F

from chuchichaestli.diffusion.ddpm import DDPM

from functools import partial


def low_pass_filter(
    x: torch.Tensor,
    N: float = 4,
    mode: str = "bicubic",
    align_corners: bool = False,
    antialias: bool = False,
):
    """Low-pass filter for the ILVR algorithm."""
    x_down = F.interpolate(
        x,
        scale_factor=1 / N,
        mode=mode,
        align_corners=align_corners,
        antialias=antialias,
    )
    x_up = F.interpolate(
        x_down,
        scale_factor=N,
        mode=mode,
        align_corners=align_corners,
        antialias=antialias,
    )
    return x_up


LinearLowPassFilter4 = partial(low_pass_filter, N=4, mode="linear")
BicubicLowPassFilter4 = partial(low_pass_filter, N=4, mode="bicubic")
LanczosLowPassFilter4 = partial(low_pass_filter, N=4, mode="lanczos", antialias=True)


class ILVR(DDPM):
    """Implementation of the ILVR algorithm for conditioned sampling from unconditioned DDPMs.

    The ILVR algorithm is described in the paper "ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models"
    by Choi et al. See https://arxiv.org/abs/2108.02938.
    """

    def __init__(
        self,
        num_timesteps: int,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        low_pass_filter=LinearLowPassFilter4,
        device: str = "cpu",
        schedule: str = "linear",
    ) -> None:
        """Initialize the ILVR algorithm.

        Args:
            num_timesteps: Number of time steps in the diffusion process.
            beta_start: Start value for beta.
            beta_end: End value for beta.
            low_pass_filter: Low-pass filter function.
            device: Device to use for the computation.
            schedule: Schedule for beta.
        """
        super().__init__(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device,
            schedule=schedule,
        )
        self.low_pass_filter = low_pass_filter

    def denoise_step(
        self,
        x_t: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from the diffusion process.

        Args:
            x_t: Tensor of shape (batch_size, *).
            y: Reference tensor (batch_size, *).
            t: Current timestep.
            model_output: Output of the model at the current timestep.
        """
        x_tm1_uncond = super().denoise_step(x_t, t, model_output)
        y_tm1, _, _ = self.noise_step(y)
        x_tm1 = (
            self.low_pass_filter(y_tm1)
            + x_tm1_uncond
            - self.low_pass_filter(x_tm1_uncond)
        )
        return x_tm1
