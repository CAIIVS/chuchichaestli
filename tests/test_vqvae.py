"""Vector Quantized Variational Autoencoder tests.

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
from chuchichaestli.models.vae import VQVAE


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
def test_vqvae_initialization(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    block_out_channel_mults,
    down_block_types,
    up_block_types,
):
    """Test VQVAE initialization with different parameters."""
    vqvae = VQVAE(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        block_out_channel_mults=block_out_channel_mults,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )
    assert isinstance(vqvae.encoder, nn.Module)
    assert isinstance(vqvae.decoder, nn.Module)


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
def test_vqvae_encode_shape(
    dimensions,
    in_channels,
    n_channels,
    block_out_channel_mults,
    down_block_types,
    up_block_types,
    input_shape,
    encoder_out_block_type,
):
    """Test the output shape of the VQVAE encoder with different parameters."""
    vqvae = VQVAE(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        latent_dim=4,
        vq_embedding_dim=32,
        block_out_channel_mults=block_out_channel_mults,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        encoder_out_block_type=encoder_out_block_type,
    )
    x = torch.randn(input_shape)
    z = vqvae.encode(x)
    spatial_dims = (
        [dim / (2 ** (len(block_out_channel_mults) - 1)) for dim in input_shape[2:]]
        if len(block_out_channel_mults) > 1
        else input_shape[2:]
    )
    assert z.shape == (
        input_shape[0],
        32,
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
def test_vqvae_decode_shape(
    dimensions,
    in_channels,
    n_channels,
    block_out_channel_mults,
    down_block_types,
    up_block_types,
    input_shape,
    decoder_in_block_type,
):
    """Test the output shape of the VQVAE decoder with different parameters."""
    vqvae = VQVAE(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        latent_dim=4,
        vq_embedding_dim=32,
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
    z_shape = (input_shape[0], 32, *spatial_dims)
    z = torch.randn(z_shape)
    decoded, _ = vqvae.decode(z)
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
def test_vqvae_forward(
    dimensions,
    in_channels,
    n_channels,
    block_out_channel_mults,
    down_block_types,
    up_block_types,
    input_shape,
):
    """Test the forward pass of the VQVAE model with different parameters."""
    vqvae = VQVAE(
        dimensions=dimensions,
        in_channels=in_channels,
        n_channels=n_channels,
        block_out_channel_mults=block_out_channel_mults,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )
    x = torch.randn(input_shape)
    z = vqvae.encode(x)
    x_tilde, _ = vqvae.decode(z)
    assert x_tilde.shape == x.shape


def test_vqvae_large():
    """Test the VQVAE model with large input dimensions."""
    vqvae = VQVAE(
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
    z = vqvae.encode(x)
    x_tilde, _ = vqvae.decode(z)
    assert x_tilde.shape == x.shape


if __name__ == "__main__":
    pytest.main(["-v", "test_vqvae.py"])
