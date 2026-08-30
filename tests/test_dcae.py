# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Standard autoencoder tests."""

import pytest
import torch
from torch import nn
from chuchichaestli.models.autoencoder import DCAE


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,latent_dim,out_channels",
    [
        (2, 1, 64, 4, 1),
        (2, 1, 32, 8, 1),
        (2, 1, 32, 4, 3),
        (1, 1, 32, 4, 1),
        (3, 1, 32, 4, 1),
    ],
)
def test_dcae_init(dimensions, in_channels, n_channels, latent_dim, out_channels):
    """Test the VAE module initialization."""
    model = DCAE.build(
        dimensions=dimensions,
        in_channels=in_channels,
        latent_dim=latent_dim,
        out_channels=out_channels,
        attn_groups=8,
        res_groups=8,
        encoder_args={"n_channels": n_channels},
        decoder_args={"num_groups": 8},
    )
    assert isinstance(model.encoder, nn.Module)
    assert isinstance(model.decoder, nn.Module)
    assert model.latent_proj is None
    assert model.latent_deproj is None
    assert len(model.encoder.mid_blocks) == 0
    assert len(model.decoder.mid_blocks) == 0


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,latent_dim,out_channels,down_block_types,up_block_types",
    [
        (
            2,
            1,
            128,
            64,
            1,
            ("DCAutoencoderDownBlock",) * 5,
            ("DCAutoencoderUpBlock",) * 5,
        ),
        (
            2,
            1,
            128,
            32,
            1,
            ("DCAutoencoderDownBlock",) * 3,
            ("DCAutoencoderUpBlock",) * 3,
        ),
        (
            2,
            1,
            256,
            256,
            1,
            ("DCAutoencoderDownBlock",) * 2,
            ("DCAutoencoderUpBlock",) * 2,
        ),
        (
            2,
            3,
            256,
            64,
            3,
            ("DCAutoencoderDownBlock",) * 1,
            ("DCAutoencoderUpBlock",) * 1,
        ),
    ],
)
def test_dcae_blocks(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    out_channels,
    down_block_types,
    up_block_types,
):
    """Test the DCAE module."""
    model = DCAE.build(
        dimensions=dimensions,
        in_channels=in_channels,
        # latent_dim=latent_dim,
        out_channels=out_channels,
        encoder_args={"n_channels": n_channels, "down_block_types": down_block_types},
        decoder_args={"up_block_types": up_block_types},
    )
    wh = 64
    shape = (1, in_channels) + (wh,) * dimensions
    sample = torch.randn(shape)
    out = model(sample)
    assert isinstance(out, torch.Tensor)


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,latent_dim,out_channels,down_block_types,up_block_types",
    [
        (
            2,
            1,
            32,
            8,
            1,
            ("DCAutoencoderDownBlock",) * 5,
            ("DCAutoencoderUpBlock",) * 5,
        ),
        (
            2,
            1,
            64,
            4,
            1,
            ("DCAutoencoderDownBlock",) * 3,
            ("DCAutoencoderUpBlock",) * 3,
        ),
        (
            2,
            1,
            16,
            8,
            1,
            ("DCAutoencoderDownBlock",) * 2,
            ("DCAutoencoderUpBlock",) * 2,
        ),
        (
            2,
            1,
            32,
            4,
            3,
            ("DCAutoencoderDownBlock",) * 1,
            ("DCAutoencoderUpBlock",) * 1,
        ),
    ],
)
def test_dcae_latent_dim(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    out_channels,
    down_block_types,
    up_block_types,
):
    """Test the DCAE module (latent dim)."""
    model = DCAE.build(
        dimensions=dimensions,
        in_channels=in_channels,
        latent_dim=latent_dim,
        out_channels=out_channels,
        encoder_args={"n_channels": n_channels, "down_block_types": down_block_types},
        decoder_args={"up_block_types": up_block_types},
    )
    wh = 64
    shape = (1, in_channels) + (wh,) * dimensions
    spatial_dims = (wh // model.f_comp,) * dimensions
    assert model.levels == (len(down_block_types), len(up_block_types))
    assert model.f_comp == 2 ** (len(down_block_types) - 1)
    assert model.f_exp == 2 ** (len(up_block_types) - 1)
    assert model.compute_latent_shape(shape) == (1, latent_dim, *spatial_dims)


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,latent_dim,out_channels,down_block_types,up_block_types",
    [
        (
            2,
            1,
            32,
            8,
            1,
            ("DCAutoencoderDownBlock",) * 5,
            ("DCAutoencoderUpBlock",) * 5,
        ),
        (
            2,
            1,
            64,
            4,
            1,
            ("DCAutoencoderDownBlock",) * 3,
            ("DCAutoencoderUpBlock",) * 3,
        ),
        (
            2,
            1,
            16,
            8,
            1,
            ("DCAutoencoderDownBlock",) * 2,
            ("DCAutoencoderUpBlock",) * 2,
        ),
        (
            2,
            1,
            32,
            4,
            3,
            ("DCAutoencoderDownBlock",) * 1,
            ("DCAutoencoderUpBlock",) * 1,
        ),
    ],
)
def test_dcae_forward(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    out_channels,
    down_block_types,
    up_block_types,
):
    """Test the DCAE module (forward pass)."""
    model = DCAE.build(
        dimensions=dimensions,
        in_channels=in_channels,
        latent_dim=latent_dim,
        out_channels=out_channels,
        encoder_args={"n_channels": n_channels, "down_block_types": down_block_types},
        decoder_args={"up_block_types": up_block_types},
    )
    wh = 64
    shape = (1, in_channels) + (wh,) * dimensions
    sample = torch.randn(shape)
    out = model(sample)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (
        1,
        out_channels,
        *(model.f_exp / model.f_comp * wh,) * dimensions,
    )
    print(out.shape)


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,latent_dim,out_channels,down_block_types,up_block_types",
    [
        (
            2,
            1,
            32,
            8,
            1,
            ("DCAutoencoderDownBlock",) * 5,
            ("DCAutoencoderUpBlock",) * 5,
        ),
        (
            2,
            1,
            64,
            4,
            1,
            ("DCAutoencoderDownBlock",) * 3,
            ("DCAutoencoderUpBlock",) * 3,
        ),
        (
            2,
            3,
            16,
            8,
            3,
            ("DCAutoencoderDownBlock",) * 2,
            ("DCAutoencoderUpBlock",) * 2,
        ),
        (
            2,
            1,
            32,
            4,
            1,
            ("DCAutoencoderDownBlock",) * 1,
            ("DCAutoencoderUpBlock",) * 1,
        ),
    ],
)
def test_dcae_backward(
    dimensions,
    in_channels,
    n_channels,
    latent_dim,
    out_channels,
    down_block_types,
    up_block_types,
):
    """Test the DCAE module (backward pass)."""
    model = DCAE.build(
        dimensions=dimensions,
        in_channels=in_channels,
        latent_dim=latent_dim,
        out_channels=out_channels,
        encoder_args={"n_channels": n_channels, "down_block_types": down_block_types},
        decoder_args={"up_block_types": up_block_types},
    )
    wh = 64
    shape = (1, in_channels) + (wh,) * dimensions
    sample = torch.randn(shape)
    out = model(sample)
    loss = nn.functional.mse_loss(out, sample)
    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            assert param.grad.abs().sum() > 0


def test_dcae_inspect():
    """Test the DCAE module inspection."""
    model = DCAE.build(
        dimensions=2,
        in_channels=1,
        latent_dim=32,
        out_channels=1,
        encoder_args={"n_channels": 128},
        # down_block_types=("DCAutoencoderDownBlock",) * 3 + ("EfficientViTBlock",) * 4,
        # up_block_types=("EfficientViTBlock",) * 4 + ("DCAutoencoderUpBlock",) * 3,
        # block_out_channel_mults=(2, 2, 1, 2, 1, 2, 1),
        # attn_scales=(3,5,7),
    )
    try:
        from torchinfo import summary

        summary(
            model,
            (1, 1, 256, 256),
            col_names=["input_size", "output_size", "kernel_size", "num_params"],
            depth=8,
        )
    except ImportError:
        print(model)
    print()


if __name__ == "__main__":
    pytest.main(["-v", "test_dcae.py"])


def test_attn_norm_type_sequence_keeps_its_intra_block_meaning():
    """Test that a norm_type sequence is not read as a per-level list."""
    model = DCAE.build(attn_norm_type=("rms", "rms"))
    assert model.levels == (6, 6)


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_dcae_reconstructs_at_every_rank(dimensions):
    """Test that deep-compression autoencoding works beyond 2D data."""
    wh = 64
    model = DCAE.build(
        dimensions=dimensions,
        in_channels=1,
        latent_dim=4,
        out_channels=1,
        attn_groups=8,
        res_groups=8,
        encoder_args={"n_channels": 32},
        decoder_args={"num_groups": 8},
    )
    shape = (1, 1) + (wh,) * dimensions
    sample = torch.randn(shape)
    out = model(sample)
    assert out.shape == shape
    assert (
        model.compute_latent_shape(shape) == (1, 4) + (wh // model.f_comp,) * dimensions
    )
    out.sum().backward()


DCAE_REDUCED = dict(
    latent_dim=8,
    attn_groups=8,
    res_groups=8,
    encoder_args={"n_channels": 32},
    decoder_args={"num_groups": 8},
)


def _assert_dc_structure(model):
    """Assert the deep-compression architecture the components are meant to carry.

    Args:
        model: Model to inspect.
    """
    from chuchichaestli.models.downsampling import DownsampleUnshuffle
    from chuchichaestli.models.upsampling import UpsampleShuffle

    assert len(model.state_dict()) == 302
    assert model.levels == (6, 6)
    assert model.f_comp == 32 and model.f_exp == 32
    assert model.latent_proj is None and model.latent_deproj is None
    assert len(model.encoder.mid_blocks) == 0 and len(model.decoder.mid_blocks) == 0
    assert [type(s[0]).__name__ for s in model.encoder.down_blocks[::2]] == [
        "DCAutoencoderDownBlock"
    ] * 3 + ["EfficientViTBlock"] * 3
    # data flow runs deepest level first, so the decoder mirrors the encoder
    assert [type(s[0]).__name__ for s in model.decoder.up_blocks[::2]] == [
        "EfficientViTBlock"
    ] * 3 + ["DCAutoencoderUpBlock"] * 3
    assert all(
        isinstance(b, DownsampleUnshuffle) for b in model.encoder.down_blocks[1::2]
    )
    assert all(isinstance(b, UpsampleShuffle) for b in model.decoder.up_blocks[1::2])


def test_dcae_build_carries_the_deep_compression_architecture():
    """Test that the flat constructor still yields the DC architecture."""
    _assert_dc_structure(DCAE.build(**DCAE_REDUCED))


def test_dcae_components_carry_the_same_architecture_when_injected():
    """Test that injecting the components gives the same model as building it."""
    from chuchichaestli.models.autoencoder import DCDecoder, DCEncoder

    encoder = DCEncoder(
        n_channels=32,
        out_channels=8,
        num_groups=8,
        attn_args={"groups": 8},
        res_args={"res_groups": 8},
    )
    decoder = DCDecoder(
        in_channels=8,
        n_channels=encoder.bottleneck_channels,
        num_groups=8,
        attn_args={"groups": 8},
        res_args={"res_groups": 8},
    )
    injected = DCAE(encoder, decoder)
    built = DCAE.build(**DCAE_REDUCED)
    _assert_dc_structure(injected)
    assert list(injected.state_dict()) == list(built.state_dict())
    assert sum(p.numel() for p in injected.parameters()) == sum(
        p.numel() for p in built.parameters()
    )


def test_dcae_components_keep_the_default_widths():
    """Test that the numeric defaults survived moving into the components."""
    import inspect

    from chuchichaestli.models.autoencoder import DCDecoder, DCEncoder

    enc = inspect.signature(DCEncoder.__init__).parameters
    dec = inspect.signature(DCDecoder.__init__).parameters
    assert enc["n_channels"].default == 128
    assert enc["out_channels"].default == 32
    assert enc["block_out_channel_mults"].default == (2, 2, 1, 2, 1)
    assert enc["norm_type"].default == "rms"
    assert dec["n_channels"].default == 1024
    assert dec["block_out_channel_mults"].default == (1, 2, 1, 2, 2)
    assert (
        dec["up_block_types"].default
        == ("EfficientViTBlock",) * 3 + ("DCAutoencoderUpBlock",) * 3
    )
    assert dec["act_fn"].default == "relu"
    assert inspect.signature(DCAE.build).parameters["latent_dim"].default == 32


def test_dcae_runs_attention_on_the_compressed_levels():
    """Test that attention sits at the low resolutions in both components."""
    model = DCAE.build(**DCAE_REDUCED)
    seen = []
    handles = [
        stage[0].register_forward_hook(
            lambda mod, inp, out: seen.append((type(mod).__name__, inp[0].shape[-1]))
        )
        for stage in (*model.encoder.down_blocks[::2], *model.decoder.up_blocks[::2])
    ]
    with torch.no_grad():
        model(torch.randn(1, 1, 64, 64))
    for handle in handles:
        handle.remove()

    attn = [size for name, size in seen if name == "EfficientViTBlock"]
    conv = [size for name, size in seen if name != "EfficientViTBlock"]
    assert attn and conv
    assert max(attn) <= min(conv)
