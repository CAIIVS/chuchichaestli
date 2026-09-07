# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Self-modulated convolutions for 1, 2, and 3D inputs."""

import torch
from torch import nn

from chuchichaestli.models.maps import DIM_TO_CONV_FN_MAP, DIM_TO_CONV_MAP


__all__ = ["SMConvND"]


class SMConvND(nn.Module):
    """Convolution that normalizes and rescales its own weight.

    A learnable scale per input channel modulates the weight, which is then
    demodulated back to unit norm per output channel:

        w'[o, i, k] = s[i] w[o, i, k] / sqrt(sum_{i, k} (s[i] w[o, i, k])**2 + eps)

    Rescaling the weight rather than the activations removes the need for a
    normalization layer in front of the convolution, which is what makes it a
    drop-in replacement for a normalization and convolution pair.

    Attributes:
        weight: Convolution weight, shaped `(out_channels, in_channels, *kernel)`.
        bias: Convolution bias, or `None`.
        scales: Learnable modulation, one scale per input channel.
        gain: Learnable scalar applied to the demodulated weight.
    """

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int | None = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | str = "same",
        bias: bool = True,
        eps: float = 1e-8,
        **kwargs,
    ):
        """Constructor.

        Args:
            dimensions: Number of spatial dimensions.
            in_channels: Number of input channels.
            out_channels: Number of output channels; as many as the input if omitted.
            kernel_size: Kernel size of the convolution.
            stride: Stride of the convolution.
            padding: Padding of the convolution.
            bias: Whether the convolution learns a bias.
            eps: Small constant keeping the demodulation finite.
            kwargs: Additional keyword arguments for the convolution.

        Raises:
            ValueError: If a keyword argument the forward pass cannot honour is
                given.
        """
        super().__init__()
        unsupported = {"groups", "padding_mode"} & kwargs.keys()
        if unsupported:
            raise ValueError(
                f"A self-modulated convolution cannot honour {sorted(unsupported)};"
                f" its weight is modulated per input channel."
            )
        out_channels = in_channels if out_channels is None else out_channels
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = kwargs.get("dilation", 1)
        self.eps = eps
        conv = DIM_TO_CONV_MAP[dimensions](
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            **kwargs,
        )
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.bias = nn.Parameter(conv.bias.detach().clone()) if bias else None
        self.scales = nn.Parameter(conv.weight.new_ones(in_channels))
        self.gain = nn.Parameter(conv.weight.new_ones(1))

    def modulated_weight(self) -> torch.Tensor:
        """Return the weight after modulation, demodulation and the gain."""
        axes = tuple(range(1, self.dimensions + 2))
        broadcast = (1, -1) + (1,) * self.dimensions
        weight = self.weight * (
            self.weight.pow(2).mean(dim=axes, keepdim=True).add(self.eps).rsqrt()
        )
        scales = self.scales * self.scales.pow(2).mean().add(self.eps).rsqrt()
        weight = weight * scales.view(broadcast)
        weight = weight * weight.pow(2).sum(dim=axes, keepdim=True).add(self.eps).rsqrt()
        return weight * self.gain.expand(self.in_channels).view(broadcast)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the self-modulated convolution."""
        out = DIM_TO_CONV_FN_MAP[self.dimensions](
            x,
            self.modulated_weight(),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        if self.bias is None:
            return out
        return out + self.bias.view((1, -1) + (1,) * self.dimensions)

    def extra_repr(self) -> str:
        """Report the channel counts and the kernel size."""
        return (
            f"{self.in_channels}, {self.out_channels},"
            f" kernel_size={tuple(self.weight.shape[2:])}, stride={self.stride}"
        )
