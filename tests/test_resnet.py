# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the ResNet module."""

import pytest
import torch
from chuchichaestli.models.blocks import ResidualBlock, ResidualBottleneck
from chuchichaestli.models.resnet import (
    ResNet,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
)


@pytest.mark.parametrize(
    "dimensions, in_channels, out_channels, time_channels, res_groups, res_act_fn, res_dropout",
    [
        (1, 32, 32, 32, 16, "silu", 0.1),
        (2, 32, 32, 32, 16, "silu", 0.1),
        (3, 32, 32, 32, 16, "silu", 0.1),
        (1, 48, 32, 32, 16, "silu", 0.1),
        (2, 48, 32, 32, 16, "silu", 0.1),
        (3, 48, 32, 32, 16, "silu", 0.1),
        (1, 8, 16, 16, 4, "relu", 0.1),
        (2, 8, 16, 16, 4, "relu", 0.1),
        (3, 8, 16, 16, 4, "relu", 0.1),
        (1, 32, 64, 16, 16, "mish", 0.1),
        (2, 32, 64, 16, 16, "mish", 0.1),
        (3, 32, 64, 16, 16, "mish", 0.1),
    ],
)
def test_forward_residual_block(
    dimensions,
    in_channels,
    out_channels,
    time_channels,
    res_groups,
    res_act_fn,
    res_dropout,
):
    """Test the forward method of the ResidualBlock module."""
    # Create dummy input tensor
    input_shape = (1, in_channels) + (32,) * dimensions
    input_tensor = torch.randn(input_shape)

    t_embedding = torch.randn((1, time_channels))

    res = ResidualBlock(
        dimensions,
        in_channels,
        out_channels,
        True,
        time_channels,
        res_groups,
        res_act_fn,
        res_dropout,
    )

    # Call the forward method
    output_tensor = res.forward(input_tensor, t_embedding)
    assert res.act1.__class__.__name__.lower() == res_act_fn
    assert res.act2.__class__.__name__.lower() == res_act_fn
    assert res.norm1.norm.__class__.__name__.lower() == "groupnorm"
    assert res.norm2.norm.__class__.__name__.lower() == "groupnorm"
    assert res.dropout.p == res_dropout
    # Check the output tensor shape
    assert output_tensor.shape == (1, out_channels) + (32,) * dimensions


@pytest.mark.parametrize(
    "dimensions, in_channels, out_channels, time_channels, res_groups, res_act_fn, res_dropout",
    [
        (1, 32, 32, 32, 16, "silu", 0.1),
        (2, 32, 32, 32, 16, "silu", 0.1),
        (3, 32, 32, 32, 16, "silu", 0.1),
        (1, 48, 32, 32, 16, "silu", 0.1),
        (2, 48, 32, 32, 16, "silu", 0.1),
        (3, 48, 32, 32, 16, "silu", 0.1),
        (1, 8, 16, 16, 4, "relu", 0.1),
        (2, 8, 16, 16, 4, "relu", 0.1),
        (3, 8, 16, 16, 4, "relu", 0.1),
        (1, 32, 64, 16, 16, "mish", 0.1),
        (2, 32, 64, 16, 16, "mish", 0.1),
        (3, 32, 64, 16, 16, "mish", 0.1),
    ],
)
def test_forward_residual_block_no_timeemb_and_alt_keys(
    dimensions,
    in_channels,
    out_channels,
    time_channels,
    res_groups,
    res_act_fn,
    res_dropout,
):
    """Test the forward method of the ResidualBlock module."""
    # Create dummy input tensor
    input_shape = (1, in_channels) + (32,) * dimensions
    input_tensor = torch.randn(input_shape)

    res = ResidualBlock(
        dimensions,
        in_channels,
        out_channels,
        time_embedding=False,
        time_channels=time_channels,
        num_groups=res_groups,
        act_fn=res_act_fn,
        dropout_p=res_dropout,
    )

    # Call the forward method
    output_tensor = res.forward(input_tensor)
    assert res.act1.__class__.__name__.lower() == res_act_fn
    assert res.act2.__class__.__name__.lower() == res_act_fn
    assert res.norm1.norm.__class__.__name__.lower() == "groupnorm"
    assert res.norm2.norm.__class__.__name__.lower() == "groupnorm"
    assert res.dropout.p == res_dropout
    # Check the output tensor shape
    assert output_tensor.shape == (1, out_channels) + (32,) * dimensions


@pytest.mark.parametrize(
    "dimensions,in_channels,out_channels,res_groups,res_act_fn,res_stride,res_dropout",
    [
        (1, 32, 32, 16, "silu", 2, 0.1),
        (2, 32, 32, 16, "silu", 2, 0.1),
        (3, 32, 32, 16, "silu", 2, 0.1),
        (1, 48, 32, 16, "silu", 2, 0.1),
        (2, 48, 32, 16, "silu", 2, 0.1),
        (3, 48, 32, 16, "silu", 2, 0.1),
        (1, 8, 16, 4, "relu", 2, 0.1),
        (2, 8, 16, 4, "relu", 2, 0.1),
        (3, 8, 16, 4, "relu", 2, 0.1),
        (1, 32, 64, 16, "mish", 4, 0.1),
        (2, 32, 64, 16, "mish", 4, 0.1),
        (3, 32, 64, 16, "mish", 4, 0.1),
    ],
)
def test_forward_residual_block_with_stride(
    dimensions,
    in_channels,
    out_channels,
    res_groups,
    res_act_fn,
    res_stride,
    res_dropout,
):
    """Test the forward method of the ResidualBlock module with stride."""
    # Create dummy input tensor
    whd = 32
    input_shape = (1, in_channels) + (whd,) * dimensions
    input_tensor = torch.randn(input_shape)

    res = ResidualBlock(
        dimensions,
        in_channels,
        out_channels,
        num_groups=res_groups,
        act_fn=res_act_fn,
        dropout_p=res_dropout,
        stride=res_stride,
    )

    # Call the forward method
    output_tensor = res.forward(input_tensor)
    assert res.act1.__class__.__name__.lower() == res_act_fn
    assert res.act2.__class__.__name__.lower() == res_act_fn
    assert res.norm1.norm.__class__.__name__.lower() == "groupnorm"
    assert res.norm2.norm.__class__.__name__.lower() == "groupnorm"
    assert res.dropout.p == res_dropout
    # Check the output tensor shape
    assert output_tensor.shape == (1, out_channels) + (whd // res_stride,) * dimensions


@pytest.mark.parametrize(
    "dimensions,input_channels, res_groups",
    [
        (1, 16, 17),
        (2, 16, 17),
        (3, 16, 17),
    ],
)
def test_forward_groups_not_divisible(dimensions, input_channels, res_groups):
    """Test the forward method of the ResidualBlock with invalid groups."""
    # Call the forward method
    with pytest.raises(ValueError):
        ResidualBlock(dimensions, input_channels, 32, True, 32, res_groups=res_groups)


@pytest.mark.parametrize(
    "dimensions,in_channels,out_channels,res_groups,res_act_fn,res_dropout",
    [
        (1, 32, 32, 32, "silu", 0.1),
        (2, 32, 32, 32, "silu", 0.1),
        (3, 32, 32, 32, "silu", 0.1),
        (1, 48, 32, 16, "silu", 0.1),
        (2, 48, 32, 16, "silu", 0.1),
        (3, 48, 32, 16, "silu", 0.1),
        (1, 8, 16, 8, "relu", 0.1),
        (2, 8, 16, 8, "relu", 0.1),
        (3, 8, 16, 8, "relu", 0.1),
        (1, 32, 64, 16, "mish", 0.1),
        (2, 32, 64, 16, "mish", 0.1),
        (3, 32, 64, 16, "mish", 0.1),
    ],
)
def test_forward_residual_bottleneck(
    dimensions,
    in_channels,
    out_channels,
    res_groups,
    res_act_fn,
    res_dropout,
):
    """Test the forward method of the ResidualBottleneck module."""
    # Create dummy input tensor
    whd = 64
    input_shape = (1, in_channels) + (whd,) * dimensions
    input_tensor = torch.randn(input_shape)

    res = ResidualBottleneck(
        dimensions,
        in_channels,
        out_channels,
        num_groups=res_groups,
        act_fn=res_act_fn,
        dropout_p=res_dropout,
    )

    # Call the forward method
    output_tensor = res.forward(input_tensor)
    assert res.act1.__class__.__name__.lower() == res_act_fn
    assert res.act2.__class__.__name__.lower() == res_act_fn
    assert res.act3.__class__.__name__.lower() == res_act_fn
    assert res.norm1.norm.__class__.__name__.lower() == "groupnorm"
    assert res.norm2.norm.__class__.__name__.lower() == "groupnorm"
    assert res.norm3.norm.__class__.__name__.lower() == "groupnorm"
    assert res.dropout.p == res_dropout
    # Check the output tensor shape
    assert output_tensor.shape == (1, out_channels) + (whd,) * dimensions


def test_ResNet():
    """Test initialization of a ResNet."""
    dimensions = 2
    img_wh = 224
    inc = 3
    model = ResNet(
        dimensions=dimensions,
        in_channels=inc,
        out_channels=1000,
        block_type="ResidualBlock",
        num_layers=[3, 4, 6, 3],
        channel_mults=[1, 2, 2, 2],
    )
    try:
        from torchinfo import summary

        summary(
            model,
            (1, inc) + (img_wh,) * dimensions,
            col_names=["input_size", "output_size", "kernel_size", "num_params"],
            depth=4,
        )
    except ImportError:
        print(model)
    print()


@pytest.mark.parametrize(
    "cls,dimensions,out_channels",
    [
        (
            ResNet18,
            2,
            1000,
        ),
        (
            ResNet34,
            2,
            1000,
        ),
        (
            ResNet50,
            2,
            1000,
        ),
        (
            ResNet101,
            2,
            1000,
        ),
        (
            ResNet152,
            2,
            1000,
        ),
    ],
)
def test_ResNetNN(cls, dimensions, out_channels):
    """Test initialization of a ResNetNN."""
    dimensions = 2
    img_wh = 224
    inc = 3
    model = cls(
        dimensions=dimensions,
        in_channels=inc,
        out_channels=1000,
    )
    try:
        from torchinfo import summary

        summary(
            model,
            (1, inc) + (img_wh,) * dimensions,
            col_names=["input_size", "output_size", "kernel_size", "num_params"],
            depth=4,
        )
    except ImportError:
        print(model)
    print()
