"""Implementation of the Attention Gate mechanism from "Attention U-Net: Learning Where to Look for the Pancreas".

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
from torch import nn
from torch.nn import functional as F
from chuchichaestli.models.maps import DIM_TO_CONV_MAP, UPSAMPLE_MODE


class AttentionGate(nn.Module):
    """Attention Gate mechanism from "Attention U-Net: Learning Where to Look for the Pancreas".

    C.f. https://arxiv.org/abs/1804.03999
    """

    def __init__(
        self,
        dimension: int = 2,
        num_channels_x: int = 1,
        num_channels_g: int = 1,
        num_channels_inter: int = 1,
        subsample_factor: int | tuple[int, ...] = 2,
        **kwargs,
    ):
        """Initialize the AttentionGate."""
        super().__init__()

        if dimension not in DIM_TO_CONV_MAP:
            raise ValueError(f"Invalid dimension: {dimension}")
        conv_cls = DIM_TO_CONV_MAP[dimension]
        self.upsample_mode = UPSAMPLE_MODE[dimension]

        # TODO: What about a "multi-scale attention gate"?

        self.W_g = conv_cls(
            num_channels_g,
            num_channels_inter,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.W_x = conv_cls(
            num_channels_x,
            num_channels_inter,
            kernel_size=subsample_factor,
            stride=subsample_factor,
            padding=0,
            bias=False,
        )  # Eq. 1 doesn't have bias for W_x
        self.psi = conv_cls(
            num_channels_inter, 1, kernel_size=1, stride=1, padding=0, bias=True
        )

        self.W_out = conv_cls(
            num_channels_x,
            num_channels_x,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.sigma1 = nn.ReLU()
        self.sigma2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor, g: torch.TensorType):
        """Forward pass of the AttentionGate.

        Args:
            x: The input features.
            g: The gating signal.

        Returns:
            The attended features.
        """
        input_size = x.size()

        theta_x = self.W_x(x)
        phi_g = self.W_g(g)

        phi_g = F.interpolate(phi_g, size=theta_x.size()[2:], mode=self.upsample_mode)
        q = self.psi(self.sigma1(theta_x + phi_g))
        alpha = F.interpolate(
            self.sigma2(q), size=input_size[2:], mode=self.upsample_mode
        )
        x_hat = alpha.expand_as(x) * x
        x_hat = self.W_out(x_hat)
        return x_hat
