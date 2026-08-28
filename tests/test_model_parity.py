# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Parity guard for scalar model configurations.

Per-level arguments must not change how a model built from plain scalar
arguments is assembled. These fingerprints pin the module structure, the
parameter shapes and the level layout of a few canonical configurations.

Regenerate deliberately (and review the diff) when a change to the built
modules is intended.
"""

import hashlib
import pytest
from chuchichaestli.models.autoencoder import DCAE, VAE, Autoencoder
from chuchichaestli.models.unet import UNet


def fingerprint(model) -> tuple[int, int, str]:
    """Return the entry count, parameter count and digest of a model's state dict."""
    lines = [f"{k}:{tuple(v.shape)}" for k, v in sorted(model.state_dict().items())]
    return (
        len(lines),
        sum(p.numel() for p in model.parameters()),
        hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16],
    )


def unet_default():
    """Build a UNet with all-default arguments."""
    return UNet()


def unet_multiblock_attn():
    """Build a UNet with repeated blocks per level, attention and a time embedding."""
    return UNet(
        dimensions=2,
        n_channels=16,
        down_block_types=("DownBlock", "DownBlock", "AttnDownBlock"),
        up_block_types=("AttnUpBlock", "UpBlock", "UpBlock"),
        block_out_channel_mults=(1, 2, 4),
        num_blocks_per_level=2,
        res_groups=4,
        skip_connection_to_all_blocks=True,
        time_embedding=True,
    )


@pytest.mark.parametrize(
    "build,expected",
    [
        (unet_default, (128, 10436705, "a6a29a870bce2d2a")),
        (unet_multiblock_attn, (184, 1075089, "108dfa3ea29aca67")),
        (Autoencoder, (244, 53867309, "67f854b17a9e9175")),
        (VAE, (244, 53885797, "8944adbc4e0bd4a0")),
        (DCAE, (302, 357219041, "cbc0fadb90cab21e")),
    ],
    ids=["unet", "unet_multiblock", "autoencoder", "vae", "dcae"],
)
def test_state_dict_parity(build, expected):
    """Test that a scalar configuration builds the same parameters as it always has."""
    assert fingerprint(build()) == expected


@pytest.mark.parametrize(
    "build,down,up",
    [
        (
            unet_default,
            ["DownBlock", "DownBlock", "AttnDownBlock", "AttnDownBlock"],
            ["AttnUpBlock", "AttnUpBlock", "UpBlock", "UpBlock"],
        ),
        (
            unet_multiblock_attn,
            ["DownBlock"] * 4 + ["AttnDownBlock"] * 2,
            ["AttnUpBlock"] * 2 + ["UpBlock"] * 4,
        ),
    ],
    ids=["unet", "unet_multiblock"],
)
def test_unet_block_layout(build, down, up):
    """Test that the UNet block sequence is unchanged, ignoring the sampling blocks."""
    model = build()
    samplers = ("Downsample", "Upsample")
    assert [
        type(b).__name__ for b in model.down_blocks if type(b).__name__ not in samplers
    ] == down
    assert [
        type(b).__name__ for b in model.up_blocks if type(b).__name__ not in samplers
    ] == up


@pytest.mark.parametrize(
    "build,levels",
    [(Autoencoder, 4), (VAE, 4), (DCAE, 6)],
    ids=["autoencoder", "vae", "dcae"],
)
def test_autoencoder_level_layout(build, levels):
    """Test that encoder and decoder interleave one stage per level with samplers."""
    model = build()
    assert model.levels == (levels, levels)
    for blocks in (model.encoder.down_blocks, model.decoder.up_blocks):
        assert len(blocks) == 2 * levels - 1
        assert [type(b).__name__ for b in blocks][::2] == ["Sequential"] * levels
