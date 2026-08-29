# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Implementation of the Attention Gate mechanism."""

import torch
from torch import nn
from torch.nn import functional as F
from chuchichaestli.models.maps import DIM_TO_CONV_MAP, UPSAMPLE_MODE


class AttentionGate(nn.Module):
    """Attention Gate module.

    As described in the paper:
    "Attention U-Net: Learning Where to Look for the Pancreas" by Oktay et al. (2018);
    see https://arxiv.org/abs/1804.03999.
    """

    def __init__(
        self,
        dimension: int = 2,
        num_channels_x: int = 1,
        num_channels_g: int = 1,
        num_channels_inter: int | None = None,
        subsample_factor: int | tuple[int, ...] = 2,
        **kwargs,
    ):
        """Initialize the AttentionGate.

        Args:
            dimension: Number of spatial dimensions.
            num_channels_x: Number of channels of the input features.
            num_channels_g: Number of channels of the gating signal.
            num_channels_inter: Number of intermediate channels; halves the input
                features by default, as in the reference implementations.
            subsample_factor: Stride at which the input features are sampled.
            kwargs: Ignored, for compatibility with the other attention modules.
        """
        super().__init__()

        if dimension not in DIM_TO_CONV_MAP:
            raise ValueError(f"Invalid dimension: {dimension}")
        if num_channels_inter is None:
            num_channels_inter = max(num_channels_x // 2, 1)
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

    def forward(self, x: torch.Tensor, g: torch.Tensor):
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
