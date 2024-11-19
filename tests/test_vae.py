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

import math


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,latent_dim,block_out_channel_mults,down_block_types,up_block_types",
    [
        (2, 3, 32, 256, (2,), ("EncoderDownBlock",), ("EncoderUpBlock",)),
        (
            1,
            1,
            16,
            128,
            (2, 2),
            ("EncoderDownBlock", "EncoderDownBlock"),
            ("EncoderUpBlock", "EncoderUpBlock"),
        ),
        (
            3,
            3,
            64,
            512,
            (2, 2, 2),
            ("EncoderDownBlock", "EncoderDownBlock", "EncoderDownBlock"),
            ("EncoderUpBlock", "EncoderUpBlock", "EncoderUpBlock"),
        ),
    ],
)
def test_vae_initialization(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    block_out_channel_mults,
    down_block_types,
    up_block_types,
):
    """Test VAE initialization with different parameters."""
    vae = VAE(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        block_out_channel_mults=block_out_channel_mults,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )
    assert isinstance(vae.encoder, nn.Module)
    assert isinstance(vae.decoder, nn.Module)


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,block_out_channel_mults,down_block_types,up_block_types,input_shape,encoder_out_block_type",
    [
        (
            2,
            3,
            4,
            (2,),
            ("EncoderDownBlock", "EncoderDownBlock"),
            ("EncoderUpBlock", "EncoderUpBlock"),
            (1, 3, 64, 64),
            "EncoderOutBlock",
        ),
        (
            2,
            3,
            4,
            (2,),
            ("EncoderDownBlock", "EncoderDownBlock"),
            ("EncoderUpBlock", "EncoderUpBlock"),
            (1, 3, 64, 64),
            "SDVAEEncoderOutBlock",
        ),
        (
            1,
            1,
            16,
            (2, 2),
            ("EncoderDownBlock", "EncoderDownBlock"),
            ("EncoderUpBlock", "EncoderUpBlock"),
            (1, 1, 28),
            "EncoderOutBlock",
        ),
        (
            3,
            3,
            64,
            (2, 2, 2),
            ("EncoderDownBlock", "EncoderDownBlock", "EncoderDownBlock"),
            ("EncoderUpBlock", "EncoderUpBlock", "EncoderUpBlock"),
            (1, 3, 16, 16, 16),
            "EncoderOutBlock",
        ),
        (
            1,
            1,
            16,
            (2, 2),
            ("EncoderDownBlock", "EncoderDownBlock"),
            ("EncoderUpBlock", "EncoderUpBlock"),
            (1, 1, 28),
            "SDVAEEncoderOutBlock",
        ),
        (
            3,
            3,
            64,
            (2, 2, 2),
            ("EncoderDownBlock", "EncoderDownBlock", "EncoderDownBlock"),
            ("EncoderUpBlock", "EncoderUpBlock", "EncoderUpBlock"),
            (1, 3, 16, 16, 16),
            "SDVAEEncoderOutBlock",
        ),
    ],
)
def test_vae_encode_shape(
    dimensions,
    in_channels,
    n_channels,
    block_out_channel_mults,
    down_block_types,
    up_block_types,
    input_shape,
    encoder_out_block_type,
):
    """Test the output shape of the VAE encoder with different parameters."""
    latent_dim = n_channels * math.prod(block_out_channel_mults)
    vae = VAE(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        latent_dim=latent_dim,
        block_out_channel_mults=block_out_channel_mults,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        encoder_out_block_type=encoder_out_block_type,
    )
    x = torch.randn(input_shape)
    dist = vae.encode(x)
    spatial_dims = (
        [dim / (2 ** (len(block_out_channel_mults) - 1)) for dim in input_shape[2:]]
        if len(block_out_channel_mults) > 1
        else input_shape[2:]
    )
    assert isinstance(dist, torch.distributions.MultivariateNormal)
    assert dist.mean.shape == (
        input_shape[0],
        latent_dim,
        *spatial_dims,
    )


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,block_out_channel_mults,down_block_types,up_block_types,input_shape,decoder_in_block_type",
    [
        (
            2,
            3,
            32,
            (2,),
            ("EncoderDownBlock",),
            ("EncoderUpBlock",),
            (1, 3, 64, 64),
            "DecoderInBlock",
        ),
        (
            1,
            1,
            16,
            (2, 2),
            ("EncoderDownBlock", "EncoderDownBlock"),
            ("EncoderUpBlock", "EncoderUpBlock"),
            (1, 1, 28),
            "DecoderInBlock",
        ),
        (
            3,
            3,
            64,
            (2, 2, 2),
            ("EncoderDownBlock", "EncoderDownBlock", "EncoderDownBlock"),
            ("EncoderUpBlock", "EncoderUpBlock", "EncoderUpBlock"),
            (1, 3, 16, 16, 16),
            "DecoderInBlock",
        ),
        (
            2,
            3,
            32,
            (2,),
            ("EncoderDownBlock",),
            ("EncoderUpBlock",),
            (1, 3, 64, 64),
            "SDVAEDecoderInBlock",
        ),
        (
            1,
            1,
            16,
            (2, 2),
            ("EncoderDownBlock", "EncoderDownBlock"),
            ("EncoderUpBlock", "EncoderUpBlock"),
            (1, 1, 28),
            "SDVAEDecoderInBlock",
        ),
        (
            3,
            3,
            64,
            (2, 2, 2),
            ("EncoderDownBlock", "EncoderDownBlock", "EncoderDownBlock"),
            ("EncoderUpBlock", "EncoderUpBlock", "EncoderUpBlock"),
            (1, 3, 16, 16, 16),
            "SDVAEDecoderInBlock",
        ),
    ],
)
def test_vae_decode_shape(
    dimensions,
    in_channels,
    n_channels,
    block_out_channel_mults,
    down_block_types,
    up_block_types,
    input_shape,
    decoder_in_block_type,
):
    """Test the output shape of the VAE decoder with different parameters."""
    latent_dim = 4
    print(latent_dim)
    vae = VAE(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        latent_dim=4,
        block_out_channel_mults=block_out_channel_mults,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        decoder_in_block_type=decoder_in_block_type,
    )
    spatial_dims = (
        [dim // (2 ** (len(block_out_channel_mults) - 1)) for dim in input_shape[2:]]
        if len(block_out_channel_mults) > 1
        else input_shape[2:]
    )
    z = torch.randn((input_shape[0], 4, *spatial_dims))
    decoded = vae.decode(z)
    assert decoded.shape == input_shape


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,block_out_channel_mults,down_block_types,up_block_types,input_shape",
    [
        (2, 3, 32, (2,), ("EncoderDownBlock",), ("EncoderUpBlock",), (1, 3, 64, 64)),
        (
            1,
            1,
            16,
            (2, 2),
            ("EncoderDownBlock", "EncoderDownBlock"),
            ("EncoderUpBlock", "EncoderUpBlock"),
            (1, 1, 28),
        ),
        (
            3,
            3,
            64,
            (2, 2, 2),
            ("EncoderDownBlock", "EncoderDownBlock", "EncoderDownBlock"),
            ("EncoderUpBlock", "EncoderUpBlock", "EncoderUpBlock"),
            (1, 3, 16, 16, 16),
        ),
    ],
)
def test_vae_forward(
    dimensions,
    in_channels,
    n_channels,
    block_out_channel_mults,
    down_block_types,
    up_block_types,
    input_shape,
):
    """Test the forward pass of the VAE model with different parameters."""
    vae = VAE(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        block_out_channel_mults=block_out_channel_mults,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )
    x = torch.randn(input_shape)
    dist = vae.encode(x)
    z = dist.sample()
    x_tilde = vae.decode(z)
    assert x_tilde.shape == x.shape


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,block_out_channel_mults,down_block_types,up_block_types,input_shape",
    [
        (2, 3, 32, (2,), ("EncoderDownBlock",), ("EncoderUpBlock",), (1, 3, 64, 64)),
        (
            1,
            1,
            16,
            (2, 2),
            ("EncoderDownBlock", "EncoderDownBlock"),
            ("EncoderUpBlock", "EncoderUpBlock"),
            (1, 1, 28),
        ),
        (
            3,
            3,
            64,
            (2, 2, 2),
            ("EncoderDownBlock", "EncoderDownBlock", "EncoderDownBlock"),
            ("EncoderUpBlock", "EncoderUpBlock", "EncoderUpBlock"),
            (1, 3, 16, 16, 16),
        ),
    ],
)
def test_vae_gradient_flow(
    dimensions,
    in_channels,
    n_channels,
    block_out_channel_mults,
    down_block_types,
    up_block_types,
    input_shape,
):
    """Test that gradients flow properly during backpropagation with different parameters."""
    vae = VAE(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        block_out_channel_mults=block_out_channel_mults,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )
    x = torch.randn(input_shape)
    dist = vae.encode(x)
    z = dist.sample()
    x_tilde = vae.decode(z)
    loss = nn.functional.mse_loss(x_tilde, x)
    loss.backward()
    for param in vae.parameters():
        if param.grad is not None:
            assert param.grad.abs().sum() > 0


def test_vae_large():
    """Test the VAE model with large input dimensions."""
    vae = VAE(
        dimensions=2,
        in_channels=1,
        n_channels=16,
        latent_dim=8,
        block_out_channel_mults=(2, 2),
        down_block_types=("EncoderDownBlock", "EncoderDownBlock"),
        mid_block_type="EncoderMidBlock",
        up_block_types=("EncoderUpBlock", "EncoderUpBlock"),
    )
    input_shape = (1, 1, 512, 512)
    x = torch.randn(input_shape)
    dist = vae.encode(x)
    z = dist.sample()
    x_tilde = vae.decode(z)
    assert x_tilde.shape == x.shape


def test_vae_kl():
    """Test the kl_div method of the VAE model."""
    vae = VAE(
        dimensions=2,
        in_channels=1,
        n_channels=16,
        latent_dim=8,
        block_out_channel_mults=(2, 2),
        down_block_types=("EncoderDownBlock", "EncoderDownBlock"),
        mid_block_type="EncoderMidBlock",
        up_block_types=("EncoderUpBlock", "EncoderUpBlock"),
    )
    x = torch.randn((1, 1, 64, 64))
    _, dist = vae(x)
    _ = vae.kl_div(dist)


if __name__ == "__main__":
    pytest.main(["-v", "test_vae.py"])
