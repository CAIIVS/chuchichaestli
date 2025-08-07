"""Tests for the UNet Time Embedding modules.

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
from chuchichaestli.models.unet.time_embeddings import (
    GaussianFourierProjection,
    SinusoidalTimeEmbedding,
    TimestepEmbedding,
    DeepSinusoidalTimeEmbedding,
)


def test_gaussian_fourier_projection_default():
    """Test GaussianFourierProjection with default parameters."""
    emb_size = 16
    batch_size = 4
    gfp = GaussianFourierProjection(embedding_size=emb_size)
    x = torch.tensor(0.5, dtype=torch.long).expand(batch_size)
    out = gfp(x)
    assert out.shape == (batch_size, emb_size * 2)
    assert out.dtype == torch.float32


def test_gaussian_fourier_projection_flip_sin_to_cos():
    """Test GaussianFourierProjection with `flip_sin_to_cos=True`."""
    emb_size = 8
    batch_size = 2
    gfp = GaussianFourierProjection(embedding_size=emb_size, flip_sin_to_cos=True)
    x = torch.tensor(0.5, dtype=torch.long).expand(batch_size)
    out = gfp(x)
    assert out.shape == (batch_size, emb_size * 2)


def test_gaussian_fourier_projection_log_false():
    """Test GaussianFourierProjection with `log=False`."""
    emb_size = 10
    gfp = GaussianFourierProjection(embedding_size=emb_size, log=False)
    x = torch.tensor(0.5, dtype=torch.long).expand(3)
    out = gfp(x)
    assert out.shape == (3, emb_size * 2)


def test_sinusoidal_time_embedding_even_channels():
    """Test SinusoidalTimeEmbedding with even number of channels."""
    num_channels = 8
    batch_size = 5
    emb = SinusoidalTimeEmbedding(
        num_channels, flip_sin_to_cos=False, downscale_freq_shift=0
    )
    timestep = torch.tensor(0.5, dtype=torch.long).expand(batch_size)
    out = emb(timestep)
    assert out.shape == (batch_size, num_channels)
    assert out.dtype == torch.float32


def test_sinusoidal_time_embedding_odd_channels():
    """Test SinusoidalTimeEmbedding with odd number of channels."""
    num_channels = 9
    batch_size = 3
    emb = SinusoidalTimeEmbedding(
        num_channels, flip_sin_to_cos=True, downscale_freq_shift=1.0
    )
    timestep = torch.tensor(0.5, dtype=torch.long).expand(batch_size)
    out = emb(timestep)
    assert out.shape == (batch_size, num_channels)
    assert out.dtype == torch.float32


def test_sinusoidal_time_embedding_scale_and_max_period():
    """Test SinusoidalTimeEmbedding with scale and max_period."""
    num_channels = 8
    emb = SinusoidalTimeEmbedding(
        num_channels, flip_sin_to_cos=False, downscale_freq_shift=0
    )
    timestep = torch.tensor(0.5, dtype=torch.long).expand(2)
    scale = 2.5
    max_period = 100
    out = emb(timestep, scale=scale, max_period=max_period)
    assert out.shape == (2, num_channels)
    assert out.dtype == torch.float32


def test_timestep_embedding_basic():
    """Test TimestepEmbedding with basic parameters."""
    input_dim = num_channels = 8
    embedding_dim = 16
    out_dim = 32
    batch_size = 5
    tstep = TimestepEmbedding(
        input_dim, embedding_dim, out_dim=out_dim, activation="silu"
    )
    x = torch.rand((batch_size, num_channels))
    out = tstep(x)
    assert out.shape == (batch_size, out_dim)
    assert out.dtype == torch.float32


def test_timestep_embedding_no_out_dim():
    """Test TimestepEmbedding without specifying out_dim."""
    input_dim = num_channels = 8
    embedding_dim = 16
    batch_size = 3
    tstep = TimestepEmbedding(input_dim, embedding_dim, activation="silu")
    x = torch.rand((batch_size, num_channels))
    out = tstep(x)
    assert out.shape == (batch_size, embedding_dim)


def test_timestep_embedding_no_activation():
    """Test TimestepEmbedding without specifying out_dim."""
    input_dim = num_channels = 8
    embedding_dim = 16
    batch_size = 4
    x = torch.rand((batch_size, num_channels))
    tstep = TimestepEmbedding(input_dim, embedding_dim)
    assert tstep.activation.__class__.__name__ == "SiLU"
    out = tstep(x)
    assert out.shape == (batch_size, embedding_dim)


def test_timestep_embedding_with_condition():
    """Test TimestepEmbedding with condition."""
    input_dim = num_channels = 8
    embedding_dim = 16
    batch_size = 2
    condition_dim = 4
    tstep = TimestepEmbedding(
        input_dim, embedding_dim, activation="relu", condition_dim=condition_dim
    )
    x = torch.rand((batch_size, num_channels))
    cond = torch.randn((batch_size, condition_dim))
    out = tstep(x, cond)
    assert out.shape == (batch_size, embedding_dim)


def test_deep_sinusoidal_time_embedding():
    """Test DeepSinusoidalTimeEmbedding with default parameters."""
    num_channels = 8
    batch_size = 3
    emb = DeepSinusoidalTimeEmbedding(
        num_channels, flip_sin_to_cos=False, downscale_freq_shift=0
    )
    timestep = torch.tensor(0.5, dtype=torch.long).expand(batch_size)
    out = emb(timestep)
    assert out.shape == (batch_size, num_channels)
    assert out.dtype == torch.float32


def test_deep_sinusoidal_time_embedding_with_embedding_dim():
    """Test DeepSinusoidalTimeEmbedding with default parameters."""
    num_channels = 8
    embedding_dim = 16
    batch_size = 3
    emb = DeepSinusoidalTimeEmbedding(
        num_channels,
        embedding_dim=embedding_dim,
        flip_sin_to_cos=False,
        downscale_freq_shift=0,
    )
    timestep = torch.tensor(0.5, dtype=torch.long).expand(batch_size)
    out = emb(timestep)
    assert out.shape == (batch_size, num_channels)
    assert out.dtype == torch.float32
