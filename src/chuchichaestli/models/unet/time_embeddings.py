"""Time embeddings for the U-Net model.

Copyright 2024 The HuggingFace Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Additional modifications made by the Intelligent Vision Systems Group at ZHAW under the
GNU General Public License v3.0 which extends the conditions of the License for further
redistribution and use. See the GPLv3 license at

    http://www.gnu.org/licenses/gpl-3.0.html

This file is part of Chuchichaestli and has been modified for use in this project.
"""

import math

import numpy as np
import torch
from torch import nn
from chuchichaestli.models.activations import ACTIVATION_FUNCTIONS
from typing import Literal


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

        x_proj = x[:, None] * self.weight[None, :] * 2 * np.pi

        if self.flip_sin_to_cos:
            out = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        else:
            out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return out


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embeddings as described in Denoising Diffusion Probabilistic Models."""

    def __init__(
        self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float
    ):
        """Sinusoidal timestep embeddings as described in Denoising Diffusion Probabilistic Models.

        Args:
            num_channels: The number of channels.
            flip_sin_to_cos: Whether to flip the sin to cos.
            downscale_freq_shift: The downscale frequency shift.
        """
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor, scale: float = 1.0, max_period: float = 1e4):
        """Forward step.

        Args:
            timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
            scale: The scale of the embeddings. Defaults to 1.0.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            [N x dim] Tensor of positional embeddings.
        """
        half_dim = self.num_channels // 2
        exponent = -math.log(max_period) * torch.arange(
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
    """Timestep embedding with optional conditioning."""

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
        condition_dim: int = None,
    ):
        """Timestep embedding.

        Args:
            input_dim: The input dimension.
            embedding_dim: The embedding dimension.
            out_dim: The output dimension. If not set, will use embedding_dim.
            activation: The activation function. Defaults to "silu".
            condition_dim (int, optional): The condition dimension. Defaults to None.
                If set, will condition the input on the condition tensor.
        """
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, embedding_dim, bias=True)
        self.activation = ACTIVATION_FUNCTIONS.get(activation)

        if self.activation is None:
            raise ValueError(f"Activation function {activation} not found.")

        if out_dim is not None:
            self.linear_2 = nn.Linear(embedding_dim, out_dim, bias=True)
        else:
            self.linear_2 = nn.Linear(embedding_dim, embedding_dim, bias=True)

        self.post_activation = ACTIVATION_FUNCTIONS.get(activation)

        if condition_dim is not None:
            self.proj_cond = nn.Linear(condition_dim, embedding_dim, bias=True)
        else:
            self.proj_cond = None

    def forward(self, sample: torch.Tensor, condition: torch.Tensor | None = None):
        """Forward pass.

        Args:
            sample: The input tensor.
            condition: An optional tensor to condition the input.
        """
        if condition is not None:
            sample += self.proj_cond(condition)

        sample = self.linear_1(sample)
        sample = self.activation(sample)
        sample = self.linear_2(sample)
        sample = self.post_activation(sample)

        return sample
