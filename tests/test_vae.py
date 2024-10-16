"""Variational autoencoder tests.

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

import pytest
import torch
from torch import nn
from chuchichaestli.models.vae import VAE


@pytest.mark.parametrize(
    "in_shape,n_channels,latent_dim,block_out_channel_mults,kernel_size,stride,padding",
    [
        ((3, 64, 64), 32, 256, (2, 2, 2, 2), 4, 2, 1),
        ((1, 28), 16, 128, (2, 2), 3, 1, 0),  # 1D data
        ((3, 16, 16, 16), 32, 256, (2, 2, 2), 3, 2, 1),  # 3D data
        ((1, 32, 32), 16, 128, (2, 2, 2), 3, 1, 1),
        ((3, 32, 32), 32, 256, (2, 2, 2, 2), 4, 2, 1),
        ((3, 64, 64, 32), 32, 256, (2, 2, 2, 2), 4, 2, 1),
        ((3, 64, 16, 32), 32, 256, (1, 2, 2, 4), 4, 2, 1),
    ],
)
def test_vae_initialization(
    in_shape,
    n_channels,
    latent_dim,
    block_out_channel_mults,
    kernel_size,
    stride,
    padding,
):
    """Test VAE initialization with different parameters."""
    vae = VAE(
        in_shape=in_shape,
        n_channels=n_channels,
        latent_dim=latent_dim,
        block_out_channel_mults=block_out_channel_mults,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    assert isinstance(vae.encoder, nn.Sequential)
    assert isinstance(vae.decoder, nn.Sequential)
    assert isinstance(vae.fc_mu, nn.Linear)
    assert isinstance(vae.fc_logvar, nn.Linear)
    assert isinstance(vae.fc_decode, nn.Linear)


@pytest.mark.parametrize(
    "in_shape,n_channels,latent_dim,block_out_channel_mults,kernel_size,stride,padding",
    [
        ((3, 64, 64), 32, 256, (2, 2, 2, 2), 4, 2, 1),
        ((1, 28), 16, 128, (2, 2), 3, 1, 0),  # 1D data
        ((3, 16, 16, 16), 32, 256, (2, 2, 2), 4, 2, 1),  # 3D data
        ((1, 32, 32), 16, 128, (2, 2, 2), 3, 1, 1),
        ((3, 32, 32), 32, 256, (2, 2, 2, 2), 4, 2, 1),
        ((3, 64, 64, 32), 32, 256, (2, 2, 2, 2), 4, 2, 1),
        ((3, 64, 16, 32), 32, 256, (1, 2, 2, 4), 4, 2, 1),
    ],
)
def test_vae_encode_shape(
    in_shape,
    n_channels,
    latent_dim,
    block_out_channel_mults,
    kernel_size,
    stride,
    padding,
):
    """Test the output shape of the VAE encoder with different parameters."""
    vae = VAE(
        in_shape=in_shape,
        n_channels=n_channels,
        latent_dim=latent_dim,
        block_out_channel_mults=block_out_channel_mults,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    x = torch.randn(1, *in_shape)  # A batch with a single input
    dist = vae.encode(x)
    assert isinstance(dist, torch.distributions.MultivariateNormal)
    assert dist.mean.shape == (1, vae.fc_mu.out_features)


@pytest.mark.parametrize(
    "in_shape,n_channels,latent_dim,block_out_channel_mults,kernel_size,stride,padding",
    [
        ((3, 64, 64), 32, 256, (2, 2, 2, 2), 4, 2, 1),
        ((1, 28), 16, 128, (2, 2), 3, 1, 0),  # 1D data
        ((3, 16, 16, 16), 32, 256, (2, 2, 2), 4, 2, 1),  # 3D data
        ((1, 32, 32), 16, 128, (2, 2, 2), 3, 1, 1),
        ((3, 32, 32), 32, 256, (2, 2, 2, 2), 4, 2, 1),
        ((3, 64, 64, 32), 32, 256, (2, 2, 2, 2), 4, 2, 1),
        ((3, 64, 16, 32), 32, 256, (1, 2, 2, 4), 4, 2, 1),
    ],
)
def test_vae_reparameterize(
    in_shape,
    n_channels,
    latent_dim,
    block_out_channel_mults,
    kernel_size,
    stride,
    padding,
):
    """Test the reparameterization process of the VAE with different parameters."""
    vae = VAE(
        in_shape=in_shape,
        n_channels=n_channels,
        latent_dim=latent_dim,
        block_out_channel_mults=block_out_channel_mults,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    x = torch.randn(1, *in_shape)
    dist = vae.encode(x)
    z = vae.reparameterize(dist)
    assert z.shape == (1, vae.fc_mu.out_features)


@pytest.mark.parametrize(
    "in_shape,n_channels,latent_dim,block_out_channel_mults,kernel_size,stride,padding",
    [
        ((3, 64, 64), 32, 256, (2, 2, 2, 2), 4, 2, 1),
        ((1, 28), 16, 128, (2, 2), 3, 1, 0),  # 1D data
        ((3, 16, 16, 16), 32, 256, (2, 2, 2), 4, 2, 1),  # 3D data
        ((1, 32, 32), 16, 128, (2, 2, 2), 3, 1, 1),
        ((3, 32, 32), 32, 256, (2, 2, 2, 2), 4, 2, 1),
        ((3, 64, 64, 32), 32, 256, (2, 2, 2, 2), 4, 2, 1),
        ((3, 64, 16, 32), 32, 256, (1, 2, 2, 4), 4, 2, 1),
    ],
)
def test_vae_decode_shape(
    in_shape,
    n_channels,
    latent_dim,
    block_out_channel_mults,
    kernel_size,
    stride,
    padding,
):
    """Test the output shape of the VAE decoder with different parameters."""
    vae = VAE(
        in_shape=in_shape,
        n_channels=n_channels,
        latent_dim=latent_dim,
        block_out_channel_mults=block_out_channel_mults,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    z = torch.randn(1, vae.fc_mu.out_features)
    decoded = vae.decode(z)
    assert decoded.shape == (1, *in_shape)


@pytest.mark.parametrize(
    "in_shape,n_channels,latent_dim,block_out_channel_mults,kernel_size,stride,padding",
    [
        ((3, 64, 64), 32, 256, (2, 2, 2, 2), 4, 2, 1),
        ((1, 28), 16, 128, (2, 2), 3, 1, 0),  # 1D data
        ((3, 16, 16, 16), 32, 256, (2, 2, 2), 4, 2, 1),  # 3D data
        ((1, 32, 32), 16, 128, (2, 2, 2), 3, 1, 1),
        ((3, 32, 32), 32, 256, (2, 2, 2, 2), 4, 2, 1),
        ((3, 64, 64, 32), 32, 256, (2, 2, 2, 2), 4, 2, 1),
        ((3, 64, 16, 32), 32, 256, (1, 2, 2, 4), 4, 2, 1),
    ],
)
def test_vae_forward(
    in_shape,
    n_channels,
    latent_dim,
    block_out_channel_mults,
    kernel_size,
    stride,
    padding,
):
    """Test the forward pass of the VAE model with different parameters."""
    vae = VAE(
        in_shape=in_shape,
        n_channels=n_channels,
        latent_dim=latent_dim,
        block_out_channel_mults=block_out_channel_mults,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    x = torch.randn(1, *in_shape)
    x_tilde, z, dist = vae.forward(x)
    assert x_tilde.shape == x.shape
    assert z.shape == (1, vae.fc_mu.out_features)
    assert isinstance(dist, torch.distributions.MultivariateNormal)


@pytest.mark.parametrize(
    "in_shape,n_channels,latent_dim,block_out_channel_mults,kernel_size,stride,padding",
    [
        ((3, 64, 64), 32, 256, (2, 2, 2, 2), 4, 2, 1),
        ((1, 28), 16, 128, (2, 2), 3, 1, 0),  # 1D data
        ((3, 16, 16, 16), 32, 256, (2, 2, 2), 4, 2, 1),  # 3D data
        ((1, 32, 32), 16, 128, (2, 2, 2), 3, 1, 1),
        ((3, 32, 32), 32, 256, (2, 2, 2, 2), 4, 2, 1),
        ((3, 64, 64, 32), 32, 256, (2, 2, 2, 2), 4, 2, 1),
        ((3, 64, 16, 32), 32, 256, (1, 2, 2, 4), 4, 2, 1),
    ],
)
def test_vae_gradient_flow(
    in_shape,
    n_channels,
    latent_dim,
    block_out_channel_mults,
    kernel_size,
    stride,
    padding,
):
    """Test that gradients flow properly during backpropagation with different parameters."""
    vae = VAE(
        in_shape=in_shape,
        n_channels=n_channels,
        latent_dim=latent_dim,
        block_out_channel_mults=block_out_channel_mults,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    x = torch.randn(1, *in_shape)
    x_tilde, _, _ = vae.forward(x)
    loss = nn.functional.mse_loss(x_tilde, x)
    loss.backward()
    for param in vae.parameters():
        if param.grad is not None:
            assert param.grad.abs().sum() > 0


if __name__ == "__main__":
    pytest.main(["-v", "test_vae.py"])
