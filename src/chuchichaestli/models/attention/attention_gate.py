# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Implementation of the Attention Gate mechanism."""

import torch
from torch import nn
from torch.nn import functional as F
from chuchichaestli.models.maps import DIM_TO_CONV_MAP, UPSAMPLE_MODE
from chuchichaestli.models.norm import Norm, NormTypes


class AttentionGate(nn.Module):
    """Attention Gate module.

    As described in the paper:
    "Attention U-Net: Learning Where to Look for the Pancreas" by Oktay et al. (2018);
    see https://arxiv.org/abs/1804.03999.

    The gate prunes the input features `x` (the encoder skip connection) using a
    gating signal `g` carrying the coarser, more contextual decoder activations.
    `subsample_factor` strides `W_x`, which is what puts `x` on the grid of `g` in
    the paper, where `g` comes from one level below `x`. Attention coefficients are
    then resampled back onto the grid of `x`. Both resampling steps are skipped when
    the grids already agree, as they do when `x` and `g` enter the gate at the same
    resolution and `subsample_factor` is 1.
    """

    def __init__(
        self,
        dimension: int = 2,
        num_channels_x: int = 1,
        num_channels_g: int = 1,
        num_channels_inter: int | None = None,
        subsample_factor: int | tuple[int, ...] = 1,
        out_norm_type: NormTypes | None = "batch",
        out_norm_groups: int = 1,
        **kwargs,
    ):
        """Initialize the AttentionGate.

        Args:
            dimension: Number of spatial dimensions.
            num_channels_x: Number of channels of the input features.
            num_channels_g: Number of channels of the gating signal.
            num_channels_inter: Number of intermediate channels; halves the input
                features by default, as in the reference implementations.
            subsample_factor: Stride at which the input features are sampled; the
                grid the attention coefficients are computed on is coarser than the
                one of `x` by this factor.
            out_norm_type: Normalization applied after the output transform.
                Defaults to `"batch"`, as in the reference implementation.
            out_norm_groups: Number of groups, if `out_norm_type` is `"group"`.
            kwargs: Ignored, for compatibility with the other attention modules.
        """
        super().__init__()

        if dimension not in DIM_TO_CONV_MAP:
            raise ValueError(f"Invalid dimension: {dimension}")
        if num_channels_inter is None:
            num_channels_inter = max(num_channels_x // 2, 1)
        conv_cls = DIM_TO_CONV_MAP[dimension]
        self.upsample_mode = UPSAMPLE_MODE[dimension]

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
        self.norm_out = (
            Norm(dimension, out_norm_type, num_channels_x, out_norm_groups)
            if out_norm_type is not None
            else None
        )

        self.sigma1 = nn.ReLU()
        self.sigma2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor, g: torch.Tensor):
        """Forward pass of the AttentionGate.

        Args:
            x: The input features; the encoder skip connection.
            g: The gating signal; the coarser decoder activations.

        Returns:
            The attended features, on the grid of `x`.
        """
        input_size = x.size()

        theta_x = self.W_x(x)
        phi_g = self.W_g(g)

        if phi_g.shape[2:] != theta_x.shape[2:]:
            phi_g = F.interpolate(
                phi_g, size=theta_x.shape[2:], mode=self.upsample_mode
            )
        alpha = self.sigma2(self.psi(self.sigma1(theta_x + phi_g)))
        if alpha.shape[2:] != input_size[2:]:
            alpha = F.interpolate(alpha, size=input_size[2:], mode=self.upsample_mode)
        x_hat = alpha.expand_as(x) * x
        x_hat = self.W_out(x_hat)
        x_hat = self.norm_out(x_hat) if self.norm_out is not None else x_hat
        return x_hat
