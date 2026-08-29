# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the UNet model."""

import warnings

import pytest
import torch
from chuchichaestli.models.attention.attention_gate import AttentionGate
from chuchichaestli.models.norm import AdaNorm
from chuchichaestli.models.unet import UNet
from chuchichaestli.models.downsampling import DOWNSAMPLE_FUNCTIONS
from chuchichaestli.models.upsampling import UPSAMPLE_FUNCTIONS


def test_throws_error_on_invalid_dimension():
    """Test that the UNet model throws an error when an invalid dimension is passed."""
    with pytest.raises(ValueError):
        UNet(dimensions=4)


def test_throws_error_on_empty_levels():
    """Test that the UNet model throws an error when a level would hold no blocks."""
    with pytest.raises(ValueError, match="at least one block"):
        UNet(num_blocks_per_level=0)


def test_throws_error_on_sampler_sequence_without_level_transitions():
    """Test that a per-level sampler sequence is rejected when there is no transition."""
    with pytest.raises(ValueError, match="no values at all"):
        UNet(
            down_block_types=("DownBlock",),
            up_block_types=("UpBlock",),
            block_out_channel_mults=(1,),
            downsample_type=("Downsample",),
        )


def test_throws_error_on_mismatched_lengths():
    """Test that the UNet model throws an error when the down and up block types have different lengths."""
    with pytest.raises(ValueError):
        UNet(
            down_block_types=("DownBlock", "AttnDownBlock"),
            up_block_types=("UpBlock", "AttnUpBlock", "AttnUpBlock"),
        )


def test_throws_error_on_mismatched_lengths_2():
    """Test that the UNet model throws an error when the down block types and out channels have different lengths."""
    with pytest.raises(ValueError):
        UNet(
            down_block_types=("DownBlock", "AttnDownBlock"),
            up_block_types=("UpBlock", "AttnUpBlock"),
            block_out_channel_mults=(1, 2, 3),
        )


def test_throws_warning_for_group_divisibility():
    """Test that the UNet model throws a warning when the number of channels is not divisible by the number of groups."""
    with pytest.warns():
        UNet(n_channels=16, res_groups=32)


def test_block_types_follow_data_flow():
    """Test that down and up blocks are built in the order they are listed in."""
    model = UNet(
        down_block_types=("DownBlock", "DownBlock", "AttnDownBlock"),
        up_block_types=("AttnUpBlock", "UpBlock", "UpBlock"),
        block_out_channel_mults=(1, 2, 4),
        res_groups=4,
    )
    assert [type(block).__name__ for block in model.down_blocks] == [
        "DownBlock",
        "Downsample",
        "DownBlock",
        "Downsample",
        "AttnDownBlock",
    ]
    assert [type(block).__name__ for block in model.up_blocks] == [
        "AttnUpBlock",
        "Upsample",
        "UpBlock",
        "Upsample",
        "UpBlock",
    ]


PER_LEVEL_CONF = {
    "n_channels": 16,
    "res_groups": 4,
    "down_block_types": ("DownBlock", "DownBlock", "AttnDownBlock"),
    "mid_block_type": "AttnMidBlock",
    "up_block_types": ("AttnUpBlock", "UpBlock", "UpBlock"),
    "block_out_channel_mults": (1, 2, 4),
}


def res_dropouts(model):
    """Collect the dropout probability of every residual block, in build order."""
    blocks = [*model.down_blocks, model.mid_block, *model.up_blocks]
    return [b.res_block.dropout.p for b in blocks if hasattr(b, "res_block")]


def attn_heads(model):
    """Collect the head count of every block that has attention, in build order."""
    blocks = [*model.down_blocks, model.mid_block, *model.up_blocks]
    return [b.attn.n_heads for b in blocks if getattr(b, "attn", None) is not None]


def test_per_level_arguments_follow_the_block_path():
    """Test that a per-level sequence is read as down levels, mid block, up levels."""
    model = UNet(**PER_LEVEL_CONF, res_dropout=(0.1, 0.2, 0.3, 0.5, 0.4, 0.2, 0.1))
    assert res_dropouts(model) == [0.1, 0.2, 0.3, 0.5, 0.4, 0.2, 0.1]


def test_per_level_arguments_broadcast_a_single_value():
    """Test that a single value still reaches every block position."""
    model = UNet(**PER_LEVEL_CONF, res_dropout=0.25)
    assert res_dropouts(model) == [0.25] * 7


def test_attention_arguments_accept_both_lengths():
    """Test that the block-path and attention-block spellings build the same model."""
    by_path = UNet(**PER_LEVEL_CONF, attn_n_heads=(1, 1, 2, 4, 8, 1, 1))
    by_attn = UNet(**PER_LEVEL_CONF, attn_n_heads=(2, 4, 8))
    assert attn_heads(by_path) == attn_heads(by_attn) == [2, 4, 8]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"res_dropout": (0.1, 0.2)},
        {"res_norm_type": ("group",) * 4},
        {"attn_n_heads": (1, 2)},
    ],
    ids=["res_short", "res_wrong", "attn_wrong"],
)
def test_throws_error_on_wrong_per_level_length(kwargs):
    """Test that a sequence matching no accepted position count is rejected."""
    with pytest.raises(ValueError):
        UNet(**PER_LEVEL_CONF, **kwargs)


def test_group_divisibility_is_checked_per_block_position():
    """Test that groups valid at a position's own width are kept."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = UNet(
            n_channels=16,
            down_block_types=("DownBlock",) * 3,
            up_block_types=("UpBlock",) * 3,
            block_out_channel_mults=(1, 2, 2),
            res_groups=(8, 8, 32, 32, 32, 8, 8),
            groups=8,
        )
    # the mid block carries 64 channels, so 32 groups divide it
    assert model.mid_block.res_block.norm1.norm.num_channels == 64
    assert model.mid_block.res_block.norm1.norm.num_groups == 32


def test_group_divisibility_warns_once_for_a_single_value():
    """Test that the group clamp reports one warning, not one per block position."""
    with pytest.warns(UserWarning, match="Number of channels") as record:
        UNet(n_channels=16, res_groups=32)
    assert len(record) == 1


@pytest.mark.parametrize(
    "dimensions,down_block_types,up_block_types,n_channels,block_out_channel_mults",
    [
        (1, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 32, (1, 2)),
        (2, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 32, (1, 2)),
        (3, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 32, (1, 2)),
        # Attention test cases in 2D
        (2, ("AttnDownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 32, (1, 2)),
        (2, ("DownBlock", "DownBlock"), ("AttnUpBlock", "UpBlock"), 32, (1, 2)),
        (2, ("DownBlock", "AttnDownBlock"), ("UpBlock", "UpBlock"), 32, (1, 2)),
        (2, ("DownBlock", "DownBlock"), ("UpBlock", "AttnUpBlock"), 32, (1, 2)),
        (2, ("AttnDownBlock", "DownBlock"), ("UpBlock", "AttnUpBlock"), 32, (1, 2)),
        (2, ("AttnDownBlock", "DownBlock"), ("UpBlock", "AttnUpBlock"), 32, (1, 2)),
        # Attention test cases in 3D
        (3, ("AttnDownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 32, (1, 2)),
        (3, ("DownBlock", "DownBlock"), ("AttnUpBlock", "UpBlock"), 32, (1, 2)),
        (3, ("DownBlock", "AttnDownBlock"), ("UpBlock", "UpBlock"), 32, (1, 2)),
        (3, ("DownBlock", "DownBlock"), ("UpBlock", "AttnUpBlock"), 32, (1, 2)),
        (3, ("AttnDownBlock", "DownBlock"), ("UpBlock", "AttnUpBlock"), 32, (1, 2)),
        (
            2,
            ("DownBlock", "DownBlock", "DownBlock"),
            ("UpBlock", "UpBlock", "UpBlock"),
            16,
            (1, 2, 2),
        ),
        (
            2,
            ("DownBlock", "DownBlock", "DownBlock", "DownBlock"),
            ("UpBlock", "UpBlock", "UpBlock", "UpBlock"),
            16,
            (1, 2, 2, 4),
        ),
        (
            3,
            ("DownBlock", "DownBlock", "DownBlock"),
            ("UpBlock", "UpBlock", "UpBlock"),
            16,
            (1, 2, 2),
        ),
        (
            3,
            ("DownBlock", "DownBlock", "DownBlock", "DownBlock"),
            ("UpBlock", "UpBlock", "UpBlock", "UpBlock"),
            16,
            (1, 2, 2, 4),
        ),
        # AttentionGate test cases
        (
            2,
            ("DownBlock", "DownBlock"),
            ("AttnGateUpBlock", "AttnGateUpBlock"),
            32,
            (1, 2),
        ),
        (
            3,
            ("DownBlock", "DownBlock"),
            ("AttnGateUpBlock", "AttnGateUpBlock"),
            32,
            (1, 2),
        ),
    ],
)
def test_forward_pass(
    dimensions, down_block_types, up_block_types, n_channels, block_out_channel_mults
):
    """Test the forward pass of the UNet model."""
    model = UNet(
        dimensions=dimensions,
        n_channels=n_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channel_mults=block_out_channel_mults,
        res_groups=4,
        num_blocks_per_level=1,
    )
    input_dims = (1, 1) + (32,) * dimensions
    sample = torch.randn(*input_dims)  # Example input
    model.eval()
    output = model(sample)
    assert output.shape == input_dims  # Check output shape
    output2 = model(sample, 0.5)
    assert torch.equal(output, output2)


@pytest.mark.parametrize(
    "dimensions,down_block_types,up_block_types,n_channels,block_out_channel_mults",
    [
        (1, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 32, (1, 2)),
        (2, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 32, (1, 2)),
        (3, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 32, (1, 2)),
        (1, ("DownBlock", "AttnDownBlock"), ("UpBlock", "UpBlock"), 32, (1, 2)),
        (2, ("DownBlock", "AttnDownBlock"), ("AttnUpBlock", "UpBlock"), 32, (1, 2)),
        (3, ("DownBlock", "AttnDownBlock"), ("AttnUpBlock", "UpBlock"), 32, (1, 2)),
        # only the first block in a level is handed a skip connection to gate
        (
            2,
            ("DownBlock", "DownBlock"),
            ("AttnGateUpBlock", "AttnGateUpBlock"),
            32,
            (1, 2),
        ),
        (
            3,
            ("DownBlock", "DownBlock"),
            ("AttnGateUpBlock", "AttnGateUpBlock"),
            32,
            (1, 2),
        ),
    ],
)
def test_forward_with_2_layers_per_block(
    dimensions, down_block_types, up_block_types, n_channels, block_out_channel_mults
):
    """Test the forward pass of the UNet model with a specified number of layers per block."""
    model = UNet(
        dimensions=dimensions,
        n_channels=n_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channel_mults=block_out_channel_mults,
        res_groups=4,
        num_blocks_per_level=2,
    )
    input_dims = (1, 1) + (32,) * dimensions
    sample = torch.randn(*input_dims)  # Example input
    output = model(sample)
    assert output.shape == input_dims  # Check output shape


def test_skip_sources_are_level_terminal():
    """Test that a skip connection starts at the last block of each level."""
    model = UNet(
        n_channels=8,
        down_block_types=("DownBlock",) * 3,
        up_block_types=("UpBlock",) * 3,
        block_out_channel_mults=(1, 2, 2),
        num_blocks_per_level=2,
        res_groups=8,
    )
    # [block, block, sampler] per level, the last level without a sampler
    assert [type(b).__name__ for b in model.down_blocks] == (
        ["DownBlock", "DownBlock", "Downsample"] * 2 + ["DownBlock"] * 2
    )
    sources = [i for i, is_source in enumerate(model.skip_sources) if is_source]
    assert sources == [1, 4, 7]
    assert model.up_samplers == [False, False, True] * 2 + [False, False]
    assert model.up_level_starts == [True, False, False] * 2 + [True, False]


@pytest.mark.parametrize(
    "up_block_type", ["UpBlock", "AttnUpBlock", "AttnGateUpBlock", "ConvAttnUpBlock"]
)
def test_forward_multiblock_levels_without_skips_to_all(up_block_type):
    """Test that later blocks in a level still run when they merge no skip."""
    model = UNet(
        n_channels=8,
        down_block_types=("DownBlock",) * 3,
        up_block_types=(up_block_type,) * 3,
        block_out_channel_mults=(1, 2, 2),
        num_blocks_per_level=2,
        res_groups=8,
        attn_groups=8,
        groups=8,
    )
    input_dims = (2, 1, 32, 32)
    assert model(torch.randn(*input_dims)).shape == input_dims


@pytest.mark.parametrize(
    "up_block_type", ["UpBlock", "AttnUpBlock", "AttnGateUpBlock", "ConvAttnUpBlock"]
)
@pytest.mark.parametrize("block_out_channel_mults", [(1, 1, 1), (1, 2, 2)])
def test_forward_skip_to_all_blocks(up_block_type, block_out_channel_mults):
    """Test the forward pass with a skip connection to every block in a level."""
    model = UNet(
        n_channels=8,
        down_block_types=("DownBlock",) * 3,
        up_block_types=(up_block_type,) * 3,
        block_out_channel_mults=block_out_channel_mults,
        num_blocks_per_level=2,
        skip_connection_to_all_blocks=True,
        res_groups=8,
        attn_groups=8,
        groups=8,
    )
    input_dims = (2, 1, 32, 32)
    output = model(torch.randn(*input_dims))
    assert output.shape == input_dims


@pytest.mark.parametrize("skip_connection_action", ["avg", "add"])
def test_skip_to_all_blocks_rejects_wider_skip(skip_connection_action):
    """Test that a skip too wide to replicate onto the input is rejected."""
    with pytest.raises(ValueError, match="channels that divides"):
        UNet(
            n_channels=8,
            down_block_types=("DownBlock",) * 3,
            up_block_types=("UpBlock",) * 3,
            block_out_channel_mults=(1, 2, 2),
            num_blocks_per_level=2,
            skip_connection_to_all_blocks=True,
            skip_connection_action=skip_connection_action,
            res_groups=8,
        )


def test_info_conv_attn(
    dimensions=2,
    down_block_types=("ConvAttnDownBlock",) * 2,
    up_block_types=("ConvAttnUpBlock",) * 2,
    n_channels=64,
    block_out_channel_mults=(1,) + (2,) * (2 - 1),
    img_wh=128,
):
    """Test print a torchinfo pass of a UNet with Conv-Attention blocks."""
    model = UNet(
        dimensions=dimensions,
        in_channels=1,
        n_channels=n_channels,
        out_channels=1,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channel_mults=block_out_channel_mults,
        time_embedding=False,
        res_groups=8,
        num_blocks_per_level=2,
        attn_groups=16,
    )
    print(f"\n# UNet({down_block_types=}, {up_block_types=})")
    try:
        from torchinfo import summary

        summary(
            model,
            (2, 1) + (img_wh,) * dimensions,
            col_names=["input_size", "output_size", "num_params"],
            # device=torch.device("cpu"),
        )
    except ImportError:
        print(model)
    print()


@pytest.mark.parametrize(
    "dimensions,down_block_types,up_block_types,block_out_channel_mults,time_embedding",
    [
        (1, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), (1, 2), True),
        (
            2,
            ("DownBlock", "AttnDownBlock"),
            ("AttnUpBlock", "UpBlock"),
            (1, 2),
            "SinusoidalTimeEmbedding",
        ),
        (
            3,
            ("DownBlock", "AttnDownBlock"),
            ("AttnUpBlock", "UpBlock"),
            (1, 2),
            "SinusoidalTimeEmbedding",
        ),
        (
            1,
            ("DownBlock", "DownBlock"),
            ("UpBlock", "UpBlock"),
            (1, 2),
            "DeepSinusoidalTimeEmbedding",
        ),
        (
            2,
            ("DownBlock", "AttnDownBlock"),
            ("AttnUpBlock", "UpBlock"),
            (1, 2),
            "DeepSinusoidalTimeEmbedding",
        ),
        (
            3,
            ("DownBlock", "AttnDownBlock"),
            ("AttnUpBlock", "UpBlock"),
            (1, 2),
            "DeepSinusoidalTimeEmbedding",
        ),
    ],
)
def test_with_timestep(
    dimensions,
    down_block_types,
    up_block_types,
    block_out_channel_mults,
    time_embedding,
):
    """Test the forward pass of the UNet model without a timestep."""
    model = UNet(
        dimensions=dimensions,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        n_channels=32,
        block_out_channel_mults=block_out_channel_mults,
        time_embedding=time_embedding,
        res_groups=8,
    )
    input_dims = (1, 1) + (32,) * dimensions
    sample = torch.randn(*input_dims)  # Example input
    timestep = 0.5  # Example timestep
    output = model(sample, timestep)
    assert output.shape == input_dims  # Check output shape
    tensor_timestep = torch.Tensor([0.5])  # same as tensor
    output = model(sample, tensor_timestep)


@pytest.mark.parametrize(
    "dimensions,down_block_types,up_block_types,n_channels,block_out_channel_mults,in_channels,out_channels",
    [
        (1, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 32, (2, 4), 1, 3),
        (2, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 32, (2, 4), 1, 3),
        (3, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 32, (2, 4), 1, 3),
        (1, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 16, (2, 4), 6, 3),
        (2, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 16, (2, 4), 6, 3),
        (3, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 16, (2, 4), 6, 3),
    ],
)
def test_out_channels(
    dimensions,
    down_block_types,
    up_block_types,
    n_channels,
    block_out_channel_mults,
    in_channels,
    out_channels,
):
    """Test the forward pass of the UNet model with a different number of output channels."""
    model = UNet(
        dimensions=dimensions,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        n_channels=n_channels,
        block_out_channel_mults=block_out_channel_mults,
        in_channels=in_channels,
        out_channels=out_channels,
        res_groups=16,
    )
    input_dims = (1, in_channels) + (32,) * dimensions
    sample = torch.randn(*input_dims)
    timestep = 0.5
    output = model(sample, timestep)
    assert output.shape == (1, out_channels) + (32,) * dimensions


@pytest.mark.parametrize(
    "in_kernel_size,out_kernel_size,res_kernel_size",
    [
        (1, 1, 1),
        (3, 3, 3),
        (5, 5, 5),
        (7, 7, 7),
        (9, 9, 9),
        (1, 3, 5),
        (5, 3, 1),
        (3, 1, 5),
        (5, 1, 3),
        (1, 5, 3),
    ],
)
def test_kernel_sizes(in_kernel_size, out_kernel_size, res_kernel_size):
    """Test the forward pass of the UNet model with different kernel sizes."""
    model = UNet(
        dimensions=2,
        down_block_types=("DownBlock", "DownBlock"),
        up_block_types=("UpBlock", "UpBlock"),
        n_channels=16,
        block_out_channel_mults=(1, 2),
        in_kernel_size=in_kernel_size,
        out_kernel_size=out_kernel_size,
        res_kernel_size=res_kernel_size,
        res_groups=16,
    )
    input_dims = (1, 1, 64, 64)
    sample = torch.randn(*input_dims)
    timestep = 0.5
    output = model(sample, timestep)
    assert output.shape == input_dims


@pytest.mark.parametrize(
    "dimensions,down_block_types,up_block_types,n_channels,block_out_channel_mults,skip_connection_action,skip_connection_to_all_blocks",
    [
        (
            1,
            ("DownBlock", "DownBlock"),
            ("UpBlock", "UpBlock"),
            8,
            (1, 2),
            "concat",
            True,
        ),
        (
            1,
            ("DownBlock", "DownBlock"),
            ("UpBlock", "UpBlock"),
            8,
            (1, 2),
            "concat",
            False,
        ),
        (
            2,
            ("DownBlock", "AttnDownBlock"),
            ("AttnUpBlock", "UpBlock"),
            8,
            (1, 2),
            "concat",
            True,
        ),
        (
            1,
            ("DownBlock", "DownBlock"),
            ("UpBlock", "UpBlock"),
            8,
            (1, 2),
            "concat",
            True,
        ),
        (
            1,
            ("DownBlock", "DownBlock"),
            ("UpBlock", "UpBlock"),
            8,
            (1, 2),
            "concat",
            False,
        ),
        (
            2,
            ("DownBlock", "AttnDownBlock"),
            ("AttnUpBlock", "UpBlock"),
            8,
            (1, 2),
            "concat",
            False,
        ),
        (
            3,
            ("DownBlock", "AttnDownBlock"),
            ("AttnUpBlock", "UpBlock"),
            8,
            (1, 2),
            "concat",
            True,
        ),
        (
            3,
            ("DownBlock", "AttnDownBlock"),
            ("AttnUpBlock", "UpBlock"),
            8,
            (1, 2),
            "concat",
            False,
        ),
        (1, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 8, (1, 2), "avg", True),
        (
            1,
            ("DownBlock", "DownBlock"),
            ("UpBlock", "UpBlock"),
            8,
            (1, 2),
            "avg",
            False,
        ),
        (
            2,
            ("DownBlock", "AttnDownBlock", "DownBlock"),
            ("AttnUpBlock", "UpBlock", "AttnUpBlock"),
            8,
            (1, 2, 3),
            "avg",
            True,
        ),
        (
            2,
            ("DownBlock", "AttnDownBlock", "DownBlock"),
            ("AttnUpBlock", "UpBlock", "AttnUpBlock"),
            8,
            (1, 2, 3),
            "avg",
            False,
        ),
        (
            3,
            ("DownBlock", "AttnDownBlock"),
            ("AttnUpBlock", "UpBlock"),
            8,
            (1, 2),
            "avg",
            True,
        ),
        (
            3,
            ("DownBlock", "AttnDownBlock"),
            ("AttnUpBlock", "UpBlock"),
            8,
            (1, 2),
            "avg",
            False,
        ),
        (1, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 8, (1, 2), "add", True),
        (
            1,
            ("DownBlock", "DownBlock"),
            ("UpBlock", "UpBlock"),
            8,
            (1, 2),
            "add",
            False,
        ),
        (
            2,
            ("DownBlock", "AttnDownBlock", "DownBlock"),
            ("AttnUpBlock", "UpBlock", "AttnUpBlock"),
            8,
            (1, 2, 3),
            "add",
            True,
        ),
        (
            2,
            ("DownBlock", "AttnDownBlock", "DownBlock"),
            ("AttnUpBlock", "UpBlock", "AttnUpBlock"),
            8,
            (1, 2, 3),
            "add",
            False,
        ),
        (
            3,
            ("DownBlock", "AttnDownBlock"),
            ("AttnUpBlock", "UpBlock"),
            8,
            (1, 2),
            "add",
            True,
        ),
        (
            3,
            ("DownBlock", "AttnDownBlock"),
            ("AttnUpBlock", "UpBlock"),
            8,
            (1, 2),
            "add",
            False,
        ),
    ],
)
def test_skip_connection_action(
    dimensions,
    down_block_types,
    up_block_types,
    n_channels,
    block_out_channel_mults,
    skip_connection_action,
    skip_connection_to_all_blocks,
):
    """Test the forward pass of the UNet model without a timestep."""
    model = UNet(
        dimensions=dimensions,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        n_channels=n_channels,
        block_out_channel_mults=block_out_channel_mults,
        time_embedding=False,
        res_groups=8,
        attn_groups=8,
        skip_connection_action=skip_connection_action,
        skip_connection_to_all_blocks=skip_connection_to_all_blocks,
    )
    input_dims = (3, 1) + (32,) * dimensions
    sample = torch.randn(*input_dims)  # Example input
    output = model(sample)
    assert output.shape == input_dims


@pytest.mark.parametrize(
    "dimensions,down_block_types,up_block_types,n_channels,block_out_channel_mults,add_noise,noise_sigma",
    [
        (1, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 32, (1, 2), "up", 0.1),
        (2, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 32, (1, 2), "up", 0.1),
        (3, ("DownBlock", "DownBlock"), ("UpBlock", "UpBlock"), 32, (1, 2), "up", 0.1),
        # Attention test cases in 2D
        (
            2,
            ("AttnDownBlock", "DownBlock"),
            ("UpBlock", "UpBlock"),
            32,
            (1, 2),
            "up",
            0.1,
        ),
        (
            2,
            ("DownBlock", "DownBlock"),
            ("AttnUpBlock", "UpBlock"),
            32,
            (1, 2),
            "up",
            0.2,
        ),
        (
            2,
            ("DownBlock", "AttnDownBlock"),
            ("UpBlock", "UpBlock"),
            32,
            (1, 2),
            "down",
            0.1,
        ),
        (
            2,
            ("DownBlock", "DownBlock"),
            ("UpBlock", "AttnUpBlock"),
            32,
            (1, 2),
            "down",
            0.2,
        ),
        (
            2,
            ("AttnDownBlock", "DownBlock"),
            ("UpBlock", "AttnUpBlock"),
            32,
            (1, 2),
            "down",
            0.1,
        ),
        (
            2,
            ("AttnDownBlock", "DownBlock"),
            ("UpBlock", "AttnUpBlock"),
            32,
            (1, 2),
            "down",
            0.1,
        ),
        # Attention test cases in 3D
        (
            3,
            ("AttnDownBlock", "DownBlock"),
            ("UpBlock", "UpBlock"),
            32,
            (1, 2),
            "up",
            0.1,
        ),
        (
            3,
            ("DownBlock", "DownBlock"),
            ("AttnUpBlock", "UpBlock"),
            32,
            (1, 2),
            "up",
            0.1,
        ),
        (
            3,
            ("DownBlock", "AttnDownBlock"),
            ("UpBlock", "UpBlock"),
            32,
            (1, 2),
            "down",
            0.1,
        ),
        (
            3,
            ("DownBlock", "DownBlock"),
            ("UpBlock", "AttnUpBlock"),
            32,
            (1, 2),
            "down",
            0.1,
        ),
        (
            3,
            ("AttnDownBlock", "DownBlock"),
            ("UpBlock", "AttnUpBlock"),
            32,
            (1, 2),
            "up",
            0.1,
        ),
        (
            2,
            ("DownBlock", "DownBlock", "DownBlock"),
            ("UpBlock", "UpBlock", "UpBlock"),
            16,
            (1, 2, 2),
            "up",
            0.1,
        ),
        (
            2,
            ("DownBlock", "DownBlock", "DownBlock", "DownBlock"),
            ("UpBlock", "UpBlock", "UpBlock", "UpBlock"),
            16,
            (1, 2, 2, 4),
            "up",
            0.1,
        ),
        (
            3,
            ("DownBlock", "DownBlock", "DownBlock"),
            ("UpBlock", "UpBlock", "UpBlock"),
            16,
            (1, 2, 2),
            "down",
            0.1,
        ),
        (
            3,
            ("DownBlock", "DownBlock", "DownBlock", "DownBlock"),
            ("UpBlock", "UpBlock", "UpBlock", "UpBlock"),
            16,
            (1, 2, 2, 4),
            "down",
            0.1,
        ),
        # AttentionGate test cases
        (
            2,
            ("DownBlock", "DownBlock"),
            ("AttnGateUpBlock", "AttnGateUpBlock"),
            32,
            (1, 2),
            "up",
            0.1,
        ),
        (
            3,
            ("DownBlock", "DownBlock"),
            ("AttnGateUpBlock", "AttnGateUpBlock"),
            32,
            (1, 2),
            "up",
            0.1,
        ),
    ],
)
def test_forward_pass_with_noise(
    dimensions,
    down_block_types,
    up_block_types,
    n_channels,
    block_out_channel_mults,
    add_noise,
    noise_sigma,
):
    """Test the forward pass of the UNet model."""
    model = UNet(
        dimensions=dimensions,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channel_mults=block_out_channel_mults,
        n_channels=n_channels,
        res_groups=4,
        num_blocks_per_level=1,
        add_noise=add_noise,
        noise_sigma=noise_sigma,
    )
    input_dims = (1, 1) + (32,) * dimensions
    sample = torch.randn(*input_dims)  # Example input
    timestep = 0.5  # Example timestep
    output1 = model(sample, timestep)
    output2 = model(sample, timestep)
    assert output1.shape == input_dims  # Check output shape
    assert not torch.equal(output1, output2)


def test_forward_pass_with_noise_at_inference(
    dimensions=2,
    down_block_types=("DownBlock", "DownBlock"),
    up_block_types=("UpBlock", "UpBlock"),
    n_channels=64,
    block_out_channel_mults=(2, 2),
    add_noise="up",
    noise_sigma=0.1,
):
    """Test the forward pass of the UNet model with noise at inference."""
    model = UNet(
        dimensions=dimensions,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channel_mults=block_out_channel_mults,
        n_channels=n_channels,
        res_groups=4,
        num_blocks_per_level=1,
        add_noise=add_noise,
        noise_sigma=noise_sigma,
    )
    model.eval()
    input_dims = (1, 1) + (32,) * dimensions
    sample = torch.randn(*input_dims)  # Example input
    timestep = 0.5  # Example timestep
    output1 = model(sample, timestep)
    output2 = model(sample, timestep)
    assert output1.shape == input_dims  # Check output shape
    assert torch.equal(output1, output2)


def test_per_level_sampling_types():
    """Test that each level transition can use a different sampling block."""
    model = UNet(
        **PER_LEVEL_CONF,
        downsample_type=("Downsample", "DownsampleInterpolate"),
        upsample_type=("UpsampleInterpolate", "Upsample"),
    )
    assert [type(b).__name__ for b in model.down_blocks][1::2] == [
        "Downsample",
        "DownsampleInterpolate",
    ]
    assert [type(b).__name__ for b in model.up_blocks][1::2] == [
        "UpsampleInterpolate",
        "Upsample",
    ]
    assert model(torch.randn(1, 1, 32, 32)).shape == (1, 1, 32, 32)


def test_per_level_skip_connection_actions():
    """Test that each up level can merge its skip connection differently."""
    model = UNet(**PER_LEVEL_CONF, skip_connection_action=("concat", "add", None))
    actions = [
        b.skip_connection_action
        for b in model.up_blocks
        if hasattr(b, "skip_connection_action")
    ]
    assert actions == ["concat", "add", None]
    assert model(torch.randn(1, 1, 32, 32)).shape == (1, 1, 32, 32)


@pytest.mark.parametrize(
    "kwargs",
    [{"downsample_type": "NotASampler"}, {"upsample_type": "NotASampler"}],
    ids=["down", "up"],
)
def test_throws_error_on_unsupported_sampling_type(kwargs):
    """Test that a sampling type the UNet cannot build is rejected up front."""
    with pytest.raises(ValueError, match="Unsupported sampling type"):
        UNet(**PER_LEVEL_CONF, **kwargs)


@pytest.mark.parametrize("downsample_type", sorted(DOWNSAMPLE_FUNCTIONS))
def test_every_registered_downsampling_type_runs(downsample_type):
    """Test that every registered downsampling type builds and preserves the shape."""
    # a sampler that changes the channel count has to be mirrored in the other half
    upsample_type = (
        "UpsampleShuffle"
        if DOWNSAMPLE_FUNCTIONS[downsample_type].changes_channels
        else "Upsample"
    )
    model = UNet(
        **PER_LEVEL_CONF,
        downsample_type=downsample_type,
        upsample_type=upsample_type,
    )
    assert model(torch.randn(1, 1, 32, 32)).shape == (1, 1, 32, 32)


@pytest.mark.parametrize("upsample_type", sorted(UPSAMPLE_FUNCTIONS))
def test_every_registered_upsampling_type_runs(upsample_type):
    """Test that every registered upsampling type builds and preserves the shape."""
    downsample_type = (
        "DownsampleUnshuffle"
        if UPSAMPLE_FUNCTIONS[upsample_type].changes_channels
        else "Downsample"
    )
    model = UNet(
        **PER_LEVEL_CONF,
        downsample_type=downsample_type,
        upsample_type=upsample_type,
    )
    assert model(torch.randn(1, 1, 32, 32)).shape == (1, 1, 32, 32)


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_shuffle_sampling_types_at_every_rank(dimensions):
    """Test that the spatial-to-channel samplers keep the skip connections aligned."""
    model = UNet(
        dimensions=dimensions,
        n_channels=16,
        res_groups=4,
        block_out_channel_mults=(1, 2, 4),
        down_block_types=("DownBlock",) * 3,
        up_block_types=("UpBlock",) * 3,
        downsample_type="DownsampleUnshuffle",
        upsample_type="UpsampleShuffle",
    )
    shape = (1, 1) + (32,) * dimensions
    out = model(torch.randn(shape))
    assert out.shape == shape
    out.sum().backward()


def test_shuffle_and_conv_sampling_types_can_be_mixed():
    """Test that a level may spend its multiplier in the sampler and the next in blocks."""
    model = UNet(
        **PER_LEVEL_CONF,
        downsample_type=("Downsample", "DownsampleUnshuffle"),
        upsample_type=("UpsampleShuffle", "Upsample"),
    )
    assert [type(b).__name__ for b in model.down_blocks][1::2] == [
        "Downsample",
        "DownsampleUnshuffle",
    ]
    assert [type(b).__name__ for b in model.up_blocks][1::2] == [
        "UpsampleShuffle",
        "Upsample",
    ]
    assert model(torch.randn(1, 1, 32, 32)).shape == (1, 1, 32, 32)


def test_throws_error_on_mirrored_sampling_type_mismatch():
    """Test that the halves must agree on which levels spend the channel multiplier."""
    with pytest.raises(ValueError, match="must agree"):
        UNet(**PER_LEVEL_CONF, upsample_type=("Upsample", "UpsampleShuffle"))


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("pool", ["AdaptiveMaxPool", "AdaptiveAvgPool"])
def test_adaptive_pooling_downsamples_a_unet(dimensions, pool):
    """Test that adaptive pooling halves each level rather than collapsing it."""
    model = UNet(
        dimensions=dimensions,
        n_channels=16,
        res_groups=4,
        block_out_channel_mults=(1, 2, 4),
        down_block_types=("DownBlock",) * 3,
        up_block_types=("UpBlock",) * 3,
        downsample_type=pool,
    )
    shape = (1, 1) + (32,) * dimensions
    assert model(torch.randn(shape)).shape == shape


def test_adaptive_pooling_tolerates_indivisible_input_sizes():
    """Test that adaptive pooling accepts a size the strided samplers cannot halve."""
    model = UNet(
        n_channels=16,
        res_groups=4,
        block_out_channel_mults=(1, 2),
        down_block_types=("DownBlock",) * 2,
        up_block_types=("UpBlock",) * 2,
        downsample_type="AdaptiveMaxPool",
    )
    assert model(torch.randn(1, 1, 30, 30)).shape == (1, 1, 30, 30)


GATE_CONF = {
    "n_channels": 16,
    "res_groups": 4,
    "block_out_channel_mults": (1, 2, 4),
    "down_block_types": ("DownBlock",) * 3,
    "up_block_types": ("AttnGateUpBlock",) * 3,
}


def attention_gates(model):
    """Collect the (input, intermediate) channel counts of every attention gate."""
    return [
        (b.attn.W_g.in_channels, b.attn.W_g.out_channels)
        for b in model.up_blocks
        if getattr(b, "attn", None) is not None
    ]


def test_attention_gate_halves_the_channels_by_default():
    """Test that the gate derives its intermediate width from the block's channels."""
    gates = attention_gates(UNet(**GATE_CONF))
    assert gates == [(channels, channels // 2) for channels, _ in gates]
    assert [inter for _, inter in gates] == [64, 16, 8]


def test_attention_gate_inter_channels_reaches_the_gate():
    """Test that an explicit intermediate width is applied rather than ignored."""
    gates = attention_gates(UNet(**GATE_CONF, attn_gate_inter_channels=64))
    assert [inter for _, inter in gates] == [64, 64, 64]


def test_attention_gate_inter_channels_can_vary_per_level():
    """Test that the gate width is a per-level argument like the other attn_ ones."""
    gates = attention_gates(UNet(**GATE_CONF, attn_gate_inter_channels=(8, 16, 32)))
    assert [inter for _, inter in gates] == [8, 16, 32]


def test_attention_gate_gates_the_skip_connection():
    """Test that an attention gate prunes the skip connection, not the decoder."""
    model = UNet(
        n_channels=8,
        down_block_types=("DownBlock",) * 2,
        up_block_types=("AttnGateUpBlock",) * 2,
        block_out_channel_mults=(1, 2),
        res_groups=4,
    )
    up_block = next(b for b in model.up_blocks if getattr(b, "attn", None) is not None)
    gate = up_block.attn
    channels = gate.W_out.in_channels
    with torch.no_grad():
        gate.psi.weight.zero_()
        gate.psi.bias.fill_(-50.0)  # alpha -> 0, so the gate prunes all of it
        # the identity output transform lets the pruned skip stay at zero
        gate.W_out.weight.copy_(torch.eye(channels).reshape(channels, channels, 1, 1))
        gate.W_out.bias.zero_()
    model.eval()

    captured = {}
    up_block.res_block.register_forward_pre_hook(
        lambda module, args: captured.update(xh=args[0])
    )
    model(torch.randn(1, 1, 16, 16))

    # concatenation puts the decoder features first and the gated skip behind them
    xh = captured["xh"]
    decoder, skip = xh[:, : xh.shape[1] // 2], xh[:, xh.shape[1] // 2 :]
    assert not torch.allclose(decoder, torch.zeros_like(decoder))
    assert torch.allclose(skip, torch.zeros_like(skip), atol=1e-4)


@pytest.mark.parametrize("subsample_factor, stride", [(1, 1), (2, 2)])
def test_attn_gate_subsample_factor(subsample_factor: int, stride: int):
    """Test that the attention gate subsample factor reaches the gate."""
    model = UNet(
        n_channels=8,
        down_block_types=("DownBlock",) * 2,
        up_block_types=("AttnGateUpBlock",) * 2,
        block_out_channel_mults=(1, 2),
        res_groups=4,
        attn_gate_subsample_factor=subsample_factor,
    )
    gates = [b.attn for b in model.up_blocks if getattr(b, "attn", None) is not None]
    assert gates
    assert all(g.W_x.stride == (stride, stride) for g in gates)
    sample = torch.randn(1, 1, 16, 16)
    assert model(sample).shape == sample.shape


def test_attention_gate_not_built_where_the_skip_is_unused():
    """Test that a level dropping its skip connection carries no attention gate."""
    args = {
        "n_channels": 8,
        "down_block_types": ("DownBlock",) * 2,
        "up_block_types": ("AttnGateUpBlock",) * 2,
        "block_out_channel_mults": (1, 2),
        "res_groups": 4,
    }
    gated = UNet(**args)
    ungated = UNet(**args, skip_connection_action=None)

    def gates(model):
        return [m for m in model.modules() if isinstance(m, AttentionGate)]

    assert len(gates(gated)) == 2
    assert not gates(ungated)  # a gate here would only ever be dead weight
    assert sum(p.numel() for p in ungated.parameters()) < sum(
        p.numel() for p in gated.parameters()
    )
    sample = torch.randn(1, 1, 16, 16)
    assert ungated(sample).shape == sample.shape


def test_attention_gate_only_on_blocks_that_take_a_skip():
    """Test that only the skip-consuming block of a level carries a gate."""
    model = UNet(
        n_channels=8,
        down_block_types=("DownBlock",) * 2,
        up_block_types=("AttnGateUpBlock",) * 2,
        block_out_channel_mults=(1, 2),
        res_groups=4,
        num_blocks_per_level=2,
    )
    carries_gate = [
        getattr(b, "attn", None) is not None
        for b in model.up_blocks
        if hasattr(b, "skip_connection_action")
    ]
    consumes_skip = [
        b.skip_connection_action is not None
        for b in model.up_blocks
        if hasattr(b, "skip_connection_action")
    ]
    assert carries_gate == consumes_skip
    assert any(carries_gate) and not all(carries_gate)


def test_gated_unet_trains_at_batch_one_down_to_a_single_pixel():
    """Test that a single-pixel gated level trains once its output norm allows it."""
    args = {
        "n_channels": 8,
        "down_block_types": ("DownBlock",) * 5,
        "up_block_types": ("AttnGateUpBlock",) * 5,
        "block_out_channel_mults": (1,) * 5,
        "res_groups": 4,
    }
    sample = torch.randn(1, 1, 16, 16)  # 16 -> a single pixel at the bottleneck

    # the reference default cannot normalize one sample over one pixel
    default = UNet(**args)
    default.train()
    with pytest.raises(ValueError, match="more than 1 value per channel"):
        default(sample)

    model = UNet(**args, attn_gate_out_norm_type=None)
    model.train()
    out = model(sample)
    assert out.shape == sample.shape
    out.sum().backward()


def time_injections(model):
    """Collect the time injection mode of every residual block, in build order."""
    blocks = [*model.down_blocks, model.mid_block, *model.up_blocks]
    return [b.res_block.time_injection for b in blocks if hasattr(b, "res_block")]


TIME_CONF = {
    "dimensions": 2,
    "n_channels": 16,
    "res_groups": 4,
    "groups": 4,
    "down_block_types": ("DownBlock", "DownBlock"),
    "up_block_types": ("UpBlock", "UpBlock"),
    "block_out_channel_mults": (1, 2),
    "time_embedding": "DeepSinusoidalTimeEmbedding",
    "time_channels": 16,
}


@pytest.mark.parametrize("res_norm_type", ["group", "layer", "rms"])
def test_forward_with_adaptive_normalization(res_norm_type):
    """Test the forward and backward pass with time-modulated normalization."""
    model = UNet(
        **TIME_CONF, res_norm_type=res_norm_type, res_time_injection="scale_shift"
    )
    sample = torch.randn(2, 1, 32, 32)
    output = model(sample, torch.tensor([3, 7]))
    assert output.shape == sample.shape
    assert torch.isfinite(output).all()

    output.sum().backward()
    projections = [m.proj for m in model.modules() if isinstance(m, AdaNorm)]
    assert projections
    assert all(p.weight.grad is not None for p in projections)


def test_adaptive_normalization_replaces_the_additive_projection():
    """Test that only the additive mode carries a time projection."""
    added = UNet(**TIME_CONF, res_time_injection="add")
    modulated = UNet(**TIME_CONF, res_time_injection="scale_shift")

    assert any("time_proj" in key for key in added.state_dict())
    assert not any(isinstance(m, AdaNorm) for m in added.modules())

    assert not any("time_proj" in key for key in modulated.state_dict())
    assert any("norm2.proj" in key for key in modulated.state_dict())


def test_time_injection_is_per_block_position():
    """Test that a per-position sequence is read as down levels, mid block, up levels."""
    modes = ("add", "scale_shift", "add", "scale_shift", "add")
    model = UNet(**TIME_CONF, res_time_injection=modes)
    assert time_injections(model) == list(modes)
    assert sum(isinstance(m, AdaNorm) for m in model.modules()) == 2


def test_time_injection_broadcasts_a_single_value():
    """Test that a single mode still reaches every block position."""
    model = UNet(**TIME_CONF, res_time_injection="scale_shift")
    assert time_injections(model) == ["scale_shift"] * 5


def test_time_injection_without_a_time_embedding():
    """Test that the mode is inert on a U-Net built without a time embedding."""
    conf = {**TIME_CONF, "time_embedding": None}
    model = UNet(**conf, res_time_injection="scale_shift")
    assert time_injections(model) == [None] * 5
    assert not any(isinstance(m, AdaNorm) for m in model.modules())


def test_adaptive_normalization_starts_out_ignoring_the_timestep():
    """Test that the zero-initialised projections leave the U-Net unconditioned."""
    torch.manual_seed(0)
    model = UNet(**TIME_CONF, res_time_injection="scale_shift").eval()
    sample = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        early = model(sample, torch.tensor([0, 0]))
        late = model(sample, torch.tensor([500, 900]))
        assert torch.allclose(early, late, atol=1e-6)

        for module in model.modules():
            if isinstance(module, AdaNorm):
                torch.nn.init.normal_(module.proj.weight, std=0.5)
        assert not torch.allclose(
            early, model(sample, torch.tensor([500, 900])), atol=1e-6
        )
