"""Implementation of the UNpaired Image Translation with Denoising DIffusion Proabilistic Models (UNITDDPM) noise process.

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

from torch import Tensor
from torch import nn
from chuchichaestli.diffusion.ddpm.samplers import DDPM


class UNITDDPM(DDPM):
    """UNITDDPM noise process for unpaired image translation with DDPMs.

    UNITDDPM is described in https://arxiv.org/abs/2104.05358.
    """

    def __init__(
        self,
        num_timesteps: int,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cpu",
        schedule: str = "linear",
        g_a: nn.Module = nn.Identity(),
        g_b: nn.Module = nn.Identity(),
    ) -> None:
        """Initialize the UNITDDPM scheme."""
        super().__init__(num_timesteps, beta_start, beta_end, device, schedule)
        self.g_a = g_a
        self.g_b = g_b

    def noise_step(self, x_a0: Tensor, x_b0: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """A single step of the noise process.

        Args:
            x_a0: Ground truth input tensor A.
            x_b0: Ground truth input tensor B.
        """
        t_a = self.sample_timesteps(x_a0.shape[0])
        t_b = self.sample_timesteps(x_b0.shape[0])
        noise_a = self.sample_noise(x_a0.shape)
        noise_b = self.sample_noise(x_b0.shape)

        x_a_tilde = self.g_b(x_b0)
        x_b_tilde = self.g_a(x_a0)

        x_ata = (
            self.sqrt_alpha_cumprod[t_a] * x_a0
            + self.sqrt_1m_alpha_cumprod[t_a] * noise_a
        )
        x_btb = (
            self.sqrt_alpha_cumprod[t_b] * x_b0
            + self.sqrt_1m_alpha_cumprod[t_b] * noise_b
        )

        x_atb = (
            self.sqrt_alpha_cumprod[t_b] * x_a_tilde
            + self.sqrt_1m_alpha_cumprod[t_b] * noise_a
        )
        x_bta = (
            self.sqrt_alpha_cumprod[t_a] * x_b_tilde
            + self.sqrt_1m_alpha_cumprod[t_a] * noise_b
        )

        return (x_ata, x_btb, x_atb, x_bta), (noise_a, noise_b), (t_a, t_b)

    def denoise_step(self, x_t: Tensor, t: int, model_output: Tensor) -> Tensor:
        """A single step of the denoising process."""
        return super().denoise_step(x_t, t, model_output)
