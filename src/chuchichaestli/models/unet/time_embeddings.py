# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later AND Apache-2.0
"""Time embeddings for the U-Net model.

Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py.
Original code is licensed under the Apache License, Version 2.0.
Modifications made by CAIIVS are licensed under the GNU General Public License v3.0.
"""

import torch
from torch import nn
from chuchichaestli.models.activations import ACTIVATION_FUNCTIONS
from typing import Literal

__all__ = [
    "GaussianFourierProjection",
    "SinusoidalTimeEmbedding",
    "DeepSinusoidalTimeEmbedding",
]


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(
        self,
        embedding_size: int = 256,
        scale: float = 1.0,
        log: bool = True,
        flip_sin_to_cos: bool = False,
    ):
        """Gaussian Fourier embeddings for noise levels.

        Args:
            embedding_size: The size of the embedding. Defaults to 256.
            scale: The scale of the embedding. Defaults to 1.0.
            log: Whether to take the log of the input. Defaults to True.
            flip_sin_to_cos: Whether to flip the sin to cos. Defaults to False.
        """
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(embedding_size) * scale, requires_grad=False
        )
        self.log = log
        self.flip_sin_to_cos = flip_sin_to_cos

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        if self.log:
            x = torch.log(x)

        x_proj = x[:, None] * self.weight[None, :] * 2 * torch.pi

        if self.flip_sin_to_cos:
            out = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        else:
            out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return out


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embeddings as described in Denoising Diffusion Probabilistic Models."""

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = False,
        downscale_freq_shift: float = 1.0,
        **kwargs,
    ):
        """Sinusoidal time embeddings.

        Args:
            num_channels: The number of channels.
            flip_sin_to_cos: Whether to flip the sin to cos.
            downscale_freq_shift: The downscale frequency shift.
            kwargs: Additional keyword arguments for compatibility (have no effect).
        """
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(
        self, timesteps: torch.Tensor, scale: float = 1.0, max_period: float = 1e4
    ):
        """Forward step.

        Args:
            timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
            scale: The scale of the embeddings. Defaults to 1.0.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            [N x dim] Tensor of positional embeddings.
        """
        half_dim = self.num_channels // 2
        exponent = -torch.log(torch.tensor(max_period)) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - self.downscale_freq_shift)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        # scale embeddings
        emb = scale * emb

        # concat sine and cosine embeddings
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # flip sine and cosine embeddings
        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

        # zero pad
        if self.num_channels % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


class TimestepEmbedding(nn.Module):
    """Linear timestep embedding with optional conditioning.

    Typicall used to further process sinusoidal embedding representations.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        out_dim: int = None,
        activation: Literal[
            "silu",
            "swish",
            "mish",
            "gelu",
            "relu",
            "prelu",
            "leakyrelu",
            "leakyrelu,0.1",
            "leakyrelu,0.2",
            "softplus",
        ] = "silu",
        post_activation: bool = False,
        condition_dim: int | None = None,
    ):
        """Linear timestep embedding.

        Args:
            input_dim: The input dimension.
            embedding_dim: The embedding dimension.
            out_dim: The output dimension. If not set, will use embedding_dim.
            activation: The activation function. Defaults to "silu".
            post_activation: Whether to use an activation function at the end.
            condition_dim: The condition dimension. Defaults to None.
                If set, will condition the input on the condition tensor.
        """
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, embedding_dim, bias=True)
        self.activation = ACTIVATION_FUNCTIONS.get(activation, "silu")()
        self.linear_2 = nn.Linear(
            embedding_dim, out_dim if out_dim is not None else embedding_dim, bias=True
        )
        self.post_activation = (
            ACTIVATION_FUNCTIONS.get(activation, "silu")()
            if post_activation
            else nn.Identity()
        )
        self.proj_cond = (
            nn.Linear(condition_dim, input_dim, bias=True)
            if condition_dim is not None
            else None
        )

    def forward(self, sample: torch.Tensor, condition: torch.Tensor | None = None):
        """Forward pass.

        Args:
            sample: The input tensor.
            condition: An optional tensor to condition the input.
        """
        if condition is not None and self.proj_cond is not None:
            sample += self.proj_cond(condition)
        sample = self.linear_1(sample)
        sample = self.activation(sample)
        sample = self.linear_2(sample)
        sample = self.post_activation(sample)
        return sample


class DeepSinusoidalTimeEmbedding(nn.Module):
    """Deep sinusoidal time embeddings, i.e. sinusoidal embedding and MLP)."""

    def __init__(
        self,
        num_channels: int,
        embedding_dim: int | None = None,
        flip_sin_to_cos: bool = False,
        downscale_freq_shift: float = 1.0,
        activation: Literal[
            "silu",
            "swish",
            "mish",
            "gelu",
            "relu",
            "prelu",
            "leakyrelu",
            "leakyrelu,0.1",
            "leakyrelu,0.2",
            "softplus",
        ] = "silu",
        post_activation: bool = False,
        condition_dim: int = None,
        **kwargs,
    ):
        """Deep sinusoidal time embeddings.

        Args:
            num_channels: The number of channels.
            embedding_dim: The dimension for the deep embedding.
            flip_sin_to_cos: Whether to flip the sin to cos.
            downscale_freq_shift: The downscale frequency shift.
            activation: The activation function. Defaults to "silu".
            post_activation: Whether to use an activation function at the end.
            condition_dim: The condition dimension. Defaults to None.
                If set, will condition the input on the condition tensor.
            kwargs: Additional keyword arguments for compatibility (have no effect).
        """
        super().__init__()
        self.sinusoidal_embedding = SinusoidalTimeEmbedding(
            num_channels=num_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            downscale_freq_shift=downscale_freq_shift,
        )
        embedding_dim = embedding_dim if embedding_dim is not None else num_channels
        self.mlp = TimestepEmbedding(
            num_channels,
            embedding_dim,
            out_dim=num_channels,
            activation=activation,
            post_activation=post_activation,
        )

    def forward(
        self,
        timesteps: torch.Tensor,
        scale: float = 1.0,
        max_period: float = 1e4,
        condition: torch.Tensor | None = None,
    ):
        """Forward step.

        Args:
            timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
            scale: The scale of the embeddings. Defaults to 1.0.
            max_period: controls the minimum frequency of the embeddings.
            condition: An optional tensor to condition the input.

        Returns:
            [N x dim] Tensor of positional embeddings.
        """
        emb = self.sinusoidal_embedding(timesteps, scale=scale, max_period=max_period)
        emb = self.mlp(emb, condition=condition)
        return emb
