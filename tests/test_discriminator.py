"""Tests for the discriminator module.

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
from chuchichaestli.models.adversarial.blocks import BaseConvBlock
from chuchichaestli.models.adversarial.discriminator import (
    BlockDiscriminator,
    PixelDiscriminator,
    PatchDiscriminator,
    AttnPatchDiscriminator,
    AntialiasingDiscriminator,
    AntialiasingPatchDiscriminator,
)


@pytest.mark.parametrize(
    "dimensions,in_channels,out_channels,act_fn,norm_type",
    [
        (1, 32, 64, "leakyrelu,0.2", "batch"),
        (2, 64, 128, "leakyrelu,0.2", "batch"),
        (3, 64, 128, "leakyrelu,0.2", "batch"),
    ],
)
def test_BaseConvBlock_forward_pass(
    dimensions,
    in_channels,
    out_channels,
    act_fn,
    norm_type,
    num_groups=32,
    img_wh=32,
):
    """Test forward pass of a BaseConvBlock module."""
    block = BaseConvBlock(
        dimensions,
        in_channels,
        out_channels,
        act_fn=act_fn,
        norm_type=norm_type,
        num_groups=num_groups,
    )
    x_shape = (1, in_channels) + (img_wh,) * dimensions
    x = torch.randn(*x_shape)
    out = block(x)
    assert block.norm is not None
    assert block.act is not None
    assert block.attn is None
    assert len(out.shape[2:]) == dimensions
    assert out.shape[1] == out_channels
    assert out.shape[-1] == img_wh // 2


@pytest.mark.parametrize(
    "dimensions,in_channels,out_channels,act_fn,norm_type,attention,attn_args",
    [
        (
            2,
            64,
            128,
            "leakyrelu,0.2",
            "batch",
            "self_attention",
            {
                "num_heads": 1,
                "head_dim": 32,
            },
        ),
        (
            2,
            64,
            128,
            "leakyrelu,0.2",
            "batch",
            "self_attention",
            {
                "num_heads": 2,
                "head_dim": 32,
            },
        ),
    ],
)
def test_BaseConvBlock_forward_pass_with_attn(
    dimensions,
    in_channels,
    out_channels,
    act_fn,
    norm_type,
    attention,
    attn_args,
    num_groups=32,
    img_wh=32,
):
    """Test forward pass of a BaseConvBlock module."""
    block = BaseConvBlock(
        dimensions,
        in_channels,
        out_channels,
        act_fn=act_fn,
        norm_type=norm_type,
        num_groups=num_groups,
        attention=attention,
        attn_args=attn_args,
    )
    x_shape = (1, in_channels) + (img_wh,) * dimensions
    x = torch.randn(*x_shape)
    out = block(x)
    assert block.attn is not None
    assert len(out.shape[2:]) == dimensions
    assert out.shape[1] == out_channels
    assert out.shape[-1] == img_wh // 2


def test_BaseConvBlock_info(
    dimensions=2,
    in_channels=64,
    out_channels=128,
    act_fn="leakyrelu,0.2",
    norm_type="batch",
    num_groups=32,
    dropout=0.2,
    attention="self_attention",
    attn_args={"num_heads": 1, "head_dim": 32},
    img_wh=64,
):
    """Test print a torchinfo pass of a BaseConvBlock module."""
    block = BaseConvBlock(
        dimensions,
        in_channels,
        out_channels,
        act_fn=act_fn,
        norm_type=norm_type,
        num_groups=num_groups,
        dropout=dropout,
        attention=attention,
        attn_args=attn_args,
    )
    print("\n# BaseConvBlock")
    try:
        from torchinfo import summary

        summary(
            block,
            (4, in_channels) + (img_wh,) * dimensions,
            col_names=["input_size", "output_size", "num_params"],
        )
    except ImportError:
        print(block)
    print()


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,out_channels,block_types,channel_mults,act_fn,norm_type",
    [
        (
            1,
            1,
            64,
            1,
            (
                "ConvDownBlock",
                "ActConvDownBlock",
                "NormActConvDownBlock",
                "NormActConvBlock",
                "NormActConvBlock",
            ),
            (2,) * 3,
            "leakyrelu,0.2",
            "batch",
        ),
        (
            1,
            1,
            64,
            1,
            (
                "ConvDownBlock",
                "ActConvDownBlock",
                "NormActConvDownBlock",
                "NormActConvBlock",
                "NormActConvBlock",
            ),
            (2,) * 3,
            "leakyrelu,0.2",
            "group",
        ),
        (
            2,
            1,
            64,
            1,
            (
                "ConvDownBlock",
                "ActConvDownBlock",
                "NormActConvDownBlock",
                "NormActConvBlock",
                "NormActConvBlock",
            ),
            (2,) * 3,
            "leakyrelu,0.2",
            "batch",
        ),
        (
            2,
            1,
            128,
            1,
            (
                "ConvDownBlock",
                "ActConvDownBlock",
                "NormActConvDownBlock",
                "NormActConvBlock",
                "NormActConvBlock",
            ),
            (2,) * 3,
            "leakyrelu,0.2",
            "group",
        ),
    ],
)
def test_BlockDiscriminator_forward_pass(
    dimensions,
    in_channels,
    n_channels,
    out_channels,
    block_types,
    channel_mults,
    act_fn,
    norm_type,
    img_wh=128,
):
    """Test forward pass of a BlockDiscriminator."""
    model = BlockDiscriminator(
        dimensions,
        in_channels,
        n_channels,
        block_types=block_types,
        channel_mults=channel_mults,
        out_channels=out_channels,
        act_fn=act_fn,
        norm_type=norm_type,
    )
    x_shape = (1, in_channels) + (img_wh,) * dimensions
    x = torch.randn(*x_shape)
    out = model(x)
    assert len(out.shape[2:]) == dimensions
    assert out.shape[1] == out_channels
    assert (
        out.shape[2] == img_wh // 2 ** (sum("Down" in bstr for bstr in block_types)) - 2
    )


@pytest.mark.parametrize(
    "dimensions,in_channels,n_channels,block_types,expected_rfs",
    [
        (
            3,
            1,
            64,
            (
                "ConvDownBlock",
                "ActConvDownBlock",
                "NormActConvDownBlock",
                "NormActConvBlock",
                "NormActConvBlock",
            ),
            70,
        ),
        (
            2,
            1,
            64,
            (
                "ConvDownBlock",
                "ActConvDownBlock",
                "NormActConvDownBlock",
                "NormActConvBlock",
                "NormActConvBlock",
            ),
            70,
        ),
        (
            1,
            1,
            128,
            (
                "ConvDownBlock",
                "ActConvDownBlock",
                "NormActConvDownBlock",
                "NormActConvBlock",
                "NormActConvBlock",
            ),
            70,
        ),
        (2, 1, 128, ("ConvDownBlock", "NormActConvBlock", "NormActConvBlock"), 16),
        (
            2,
            1,
            64,
            (
                "ConvDownBlock",
                "ActConvDownBlock",
                *(("NormActConvDownBlock",) * 3),
                "NormActConvBlock",
                "NormActConvBlock",
            ),
            286,
        ),
    ],
)
def test_BlockDiscriminator_receptive_field(
    dimensions,
    in_channels,
    n_channels,
    block_types,
    expected_rfs,
):
    """Test forward pass of a BlockDiscriminator."""
    model = BlockDiscriminator(
        dimensions, in_channels, n_channels, block_types=block_types
    )
    rfs = model.receptive_field()
    assert rfs[0] == expected_rfs


def test_BlockDiscriminator_info(
    dimensions=2,
    in_channels=1,
    n_channels=64,
    block_types=(
        "ConvDownBlock",
        "ActConvDownBlock",
        "NormActConvDownBlock",
        "NormActConvBlock",
        "NormActConvBlock",
    ),
    channel_mults=(2,) * 3,
    norm_type="group",
    out_channels=1,
    img_wh=128,
):
    """Test print a torchinfo pass of a BlockDiscriminator module."""
    model = BlockDiscriminator(
        dimensions,
        in_channels,
        n_channels,
        block_types=block_types,
        channel_mults=channel_mults,
        norm_type="group",
        out_channels=out_channels,
    )
    print("\n# BlockDiscriminator")
    try:
        from torchinfo import summary

        summary(
            model,
            (4, in_channels) + (img_wh,) * dimensions,
            col_names=["input_size", "output_size", "num_params"],
        )
    except ImportError:
        print(model)
    print()


def test_BlockDiscriminator_with_ResidualBlock_forward(
    dimensions=2,
    in_channels=1,
    n_channels=64,
    block_types=(
        "ConvDownBlock",
        "ResidualBlock",
        "ConvDownBlock",
        "NormConvBlock",
    ),
    channel_mults=(2, 1),
    norm_type="batch",
    out_channels=1,
    img_wh=128,
):
    """Test print a torchinfo pass of a BlockDiscriminator module with a ResidualBlock."""
    model = BlockDiscriminator(
        dimensions,
        in_channels,
        n_channels,
        block_types=block_types,
        channel_mults=channel_mults,
        norm_type=norm_type,
        out_channels=out_channels,
    )
    print("\n# BlockDiscriminator (with ResidualBlock)")
    x_shape = (4, in_channels) + (img_wh,) * dimensions
    try:
        from torchinfo import summary

        summary(
            model,
            x_shape,
            col_names=["input_size", "output_size", "num_params"],
        )
    except ImportError:
        x = torch.randn(*x_shape)
        out = model(x)
        assert len(out.shape[2:]) == dimensions
        assert out.shape[1] == out_channels
        print(model)
    assert list(model.modules())[3].__class__.__name__ == "ResidualBlock"
    print()


def test_PixelDiscriminator_info(
    dimensions=2,
    in_channels=1,
    n_channels=64,
    img_wh=128,
):
    """Test print a torchinfo pass of a PixelDiscriminator module."""
    model = PixelDiscriminator(
        dimensions,
        in_channels,
        n_channels,
    )
    print("\n# PixelDiscriminator")
    x_shape = (4, in_channels) + (img_wh,) * dimensions
    try:
        from torchinfo import summary

        summary(
            model,
            x_shape,
            col_names=["input_size", "output_size", "num_params"],
        )
    except ImportError:
        x = torch.randn(*x_shape)
        out = model(x)
        assert len(out.shape[2:]) == dimensions
        assert out.shape[1] == 1
        print(model)

    print()


def test_PatchDiscriminator_info(
    dimensions=2,
    in_channels=1,
    n_channels=64,
    n_hidden=3,
    img_wh=128,
):
    """Test print a torchinfo pass of a PatchDiscriminator module."""
    model = PatchDiscriminator(
        dimensions,
        in_channels,
        n_channels,
        n_hidden=n_hidden,
    )
    print("\n# PatchDiscriminator")
    x_shape = (4, in_channels) + (img_wh,) * dimensions
    try:
        from torchinfo import summary

        summary(
            model,
            x_shape,
            col_names=["input_size", "output_size", "num_params"],
        )
    except ImportError:
        x = torch.randn(*x_shape)
        out = model(x)
        assert len(out.shape[2:]) == dimensions
        assert out.shape[1] == 1
        print(model)
    print()


def test_AttnPatchDiscriminator_info(
    dimensions=2,
    in_channels=1,
    n_channels=64,
    n_hidden=3,
    attn_n_heads=4,
    attn_head_dim=64,
    img_wh=128,
):
    """Test print a torchinfo pass of a AttnPatchDiscriminator module."""
    model = AttnPatchDiscriminator(
        dimensions,
        in_channels,
        n_channels,
        n_hidden=n_hidden,
        attn_n_heads=attn_n_heads,
        attn_head_dim=attn_head_dim,
    )
    print("\n# AttnPatchDiscriminator")
    x_shape = (4, in_channels) + (img_wh,) * dimensions
    try:
        from torchinfo import summary

        summary(
            model,
            x_shape,
            col_names=["input_size", "output_size", "num_params"],
        )
    except ImportError:
        x = torch.randn(*x_shape)
        out = model(x)
        assert len(out.shape[2:]) == dimensions
        assert out.shape[1] == 1
        print(model)
    print()


def test_AntialiasingDiscriminator_info(
    dimensions=2,
    in_channels=1,
    n_channels=64,
    patch_size=32,
    n_hidden=3,
    img_wh=128,
):
    """Test print a torchinfo pass of a AntialiasingDiscriminator module."""
    model = AntialiasingDiscriminator(
        dimensions,
        in_channels,
        n_channels,
        patch_size=patch_size,
        n_hidden=n_hidden,
    )
    print("\n# AntialiasingDiscriminator")
    x_shape = (4, in_channels) + (img_wh,) * dimensions
    try:
        from torchinfo import summary

        summary(
            model,
            x_shape,
            col_names=["input_size", "output_size", "num_params"],
        )
    except ImportError:
        x = torch.randn(*x_shape)
        out = model(x)
        assert len(out.shape[2:]) == dimensions
        assert out.shape[1] == 1
        print(model)
    print()


def test_AntialiasingPatchDiscriminator_info(
    dimensions=2,
    in_channels=1,
    n_channels=64,
    patch_size=32,
    n_hidden=3,
    img_wh=128,
):
    """Test print a torchinfo pass of a AntialiasingPatchDiscriminator module."""
    model = AntialiasingPatchDiscriminator(
        dimensions,
        in_channels,
        n_channels,
        patch_size=patch_size,
        n_hidden=n_hidden,
    )
    print("\n# AntialiasingPatchDiscriminator")
    x_shape = (4, in_channels) + (img_wh,) * dimensions
    try:
        from torchinfo import summary

        summary(
            model,
            x_shape,
            col_names=["input_size", "output_size", "num_params"],
        )
    except ImportError:
        x = torch.randn(*x_shape)
        out = model(x)
        assert len(out.shape[2:]) == dimensions
        assert out.shape[1] == 1
        print(model)
    print()
