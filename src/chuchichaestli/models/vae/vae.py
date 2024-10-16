"""Variational autoencoder implementation.

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
import math

from chuchichaestli.models.activations import ACTIVATION_FUNCTIONS
from chuchichaestli.models.maps import DIM_TO_CONV_MAP, DIM_TO_CONVT_MAP


class VAE(nn.Module):
    """Variational autoencoder implementation."""

    def __init__(
        self,
        in_shape: tuple[int, ...] = (3, 64, 64),
        n_channels: int = 32,
        latent_dim: int = 256,
        block_out_channel_mults: tuple[int, ...] = (2, 2, 2, 2),
        act: str = "silu",
        output_activation: str = "sigmoid",
        kernel_size: int | tuple[int, ...] = 4,
        stride: int | tuple[int, ...] = 2,
        padding: int | tuple[int, ...] = 1,
        dilation: int | tuple[int, ...] = 1,
        out_kernel_size: int | tuple[int, ...] = 3,
        out_stride: int | tuple[int, ...] = 1,
        out_padding: int | tuple[int, ...] | str = "same",
        out_dilation: int | tuple[int, ...] = 1,
    ):
        """Initialize the VAE model.

        Args:
            in_shape: Shape of the input excluding the batch dimension (i.e. only channels and spatial dimensions).
            n_channels: Number of channels in the first layer of the encoder
            latent_dim: Dimensionality of the latent space
            block_out_channel_mults: Multipliers for the number of channels in each block
            act: Activation function to use
            output_activation: Activation function to use for the output
            kernel_size: Kernel size of the convolutional layers.
            stride: Stride of the convolutional layers.
            padding: Padding of the convolutional layers.
            dilation: Dilation of the convolutional layers.
            out_kernel_size: Kernel size of the output convolutional layer.
            out_stride: Stride of the output convolutional layer.
            out_padding: Padding of the output convolutional layer.
            out_dilation: Dilation of the output convolutional layer.
        """
        super().__init__()

        in_channels = in_shape[0]
        dimensions = len(in_shape) - 1

        Conv = DIM_TO_CONV_MAP[dimensions]
        ConvT = DIM_TO_CONVT_MAP[dimensions]

        kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size,) * dimensions
        )
        stride = stride if isinstance(stride, tuple) else (stride,) * dimensions
        padding = padding if isinstance(padding, tuple) else (padding,) * dimensions
        dilation = dilation if isinstance(dilation, tuple) else (dilation,) * dimensions

        def DownBlock(in_channels, out_channels):
            return nn.Sequential(
                Conv(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                ),
                ACTIVATION_FUNCTIONS[act](),
            )

        def UpBlock(in_channels, out_channels):
            return nn.Sequential(
                ConvT(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                ),
                ACTIVATION_FUNCTIONS[act](),
            )

        image_channels = in_channels

        # Encoder
        encoder_layers = []
        out_channels = n_channels
        for mult in block_out_channel_mults:
            encoder_layers.append(DownBlock(in_channels, out_channels))
            in_channels = out_channels
            out_channels = out_channels * mult
        self.encoder = nn.Sequential(*encoder_layers)

        self.bottleneck_channels = in_channels

        self.bottleneck_dims, self.encoded_size = self._compute_encoded_size(
            block_out_channel_mults,
            in_shape,
            padding,
            dilation,
            kernel_size,
            stride,
        )
        self.encoded_size *= self.bottleneck_channels
        self.fc_mu = nn.Linear(self.encoded_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoded_size, latent_dim)
        self.softplus = nn.Softplus()

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.encoded_size)

        decoder_layers = []
        out_channels = in_channels // block_out_channel_mults[-1]
        for mult in reversed(block_out_channel_mults):
            decoder_layers.append(UpBlock(in_channels, out_channels))
            in_channels = out_channels
            out_channels = max(out_channels // mult, image_channels)

        decoder_layers.append(
            Conv(
                in_channels,
                image_channels,
                kernel_size=out_kernel_size,
                stride=out_stride,
                padding=out_padding,
                dilation=out_dilation,
            )
        )
        decoder_layers.append(ACTIVATION_FUNCTIONS[output_activation]())

        self.decoder = nn.Sequential(*decoder_layers)
        print(self.decoder)

    @staticmethod
    def _compute_encoded_size(
        block_out_channel_mults,
        in_shape,
        padding,
        dilation,
        kernel_size,
        stride,
    ):
        dims = in_shape[1:]
        for _ in range(len(block_out_channel_mults)):
            dims = tuple(
                (s + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1)
                // stride[i]
                + 1
                for i, s in enumerate(dims)
            )
        return dims, math.prod(dims)

    def encode(self, x, eps: float = 1e-8):
        """Encodes the input data into the latent space.

        Args:
            x: Input data
            eps: Small value to avoid numerical instability

        Returns:
            torch.distributions.MultivariateNormal: Normal distribution of the encoded data.
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)

        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)

    def reparameterize(self, dist):
        """Reparameterizes the encoded data to sample from the latent space.

        Args:
            dist: Normal distribution of the encoded data.

        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        return dist.rsample()

    def decode(self, z):
        """Decodes the data from the latent space to the original input space.

        Args:
            z (torch.Tensor): Data in the latent space.

        Returns:
            torch.Tensor: Reconstructed data in the original input space.
        """
        z = self.fc_decode(z)
        z = z.view(z.size(0), self.bottleneck_channels, *self.bottleneck_dims)
        return self.decoder(z)

    def forward(self, x, eps: float = 1e-8):
        """Forward pass through the model.

        This function is primarily used for training the model.
        For encoding and decoding, use the `encode` and `decode` functions.

        Args:
            x: Input data
            eps: Small value to avoid numerical instability

        Returns:
            torch.Tensor: Reconstructed data in the original input space ("x tilde").
            torch.Tensor: Data in the latent space ("z").
            torch.distributions.MultivariateNormal: Normal distribution of the encoded data ("distribution over z").
        """
        dist = self.encode(x, eps)
        z = self.reparameterize(dist)
        return self.decode(z), z, dist
