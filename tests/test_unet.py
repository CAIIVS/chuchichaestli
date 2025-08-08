# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the UNet model."""

import pytest
import torch
from chuchichaestli.models.unet import UNet


def test_throws_error_on_invalid_dimension():
    """Test that the UNet model throws an error when an invalid dimension is passed."""
    with pytest.raises(ValueError):
        UNet(dimensions=4)


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
