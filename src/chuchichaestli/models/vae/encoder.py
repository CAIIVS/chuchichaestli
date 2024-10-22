"""Encoder for variational autoencoder implementation.

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

from torch import nn

from chuchichaestli.models.activations import ACTIVATION_FUNCTIONS
from chuchichaestli.models.downsampling import Downsample
from chuchichaestli.models.blocks import BLOCK_MAP
from chuchichaestli.models.maps import DIM_TO_CONV_MAP


class SDVAEEncoderOutBlock(nn.Module):
    """SD-VAE-like output block for encoder."""

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        groups: int,
        act: str,
        **kwargs,
    ):
        """Initialize the output block.

        Args:
            dimensions (int): Number of dimensions.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            groups (int): Number of groups for normalization.
            act (str): Activation function.
            kwargs: Additional arguments.
        """
        super().__init__()
        self.conv1 = DIM_TO_CONV_MAP[dimensions](
            in_channels, in_channels, kernel_size=3, padding=1
        )
        self.conv2 = DIM_TO_CONV_MAP[dimensions](
            in_channels, out_channels, kernel_size=1, padding="same"
        )
        self.norm = nn.GroupNorm(num_channels=in_channels, num_groups=groups, eps=1e-6)
        self.act = ACTIVATION_FUNCTIONS[act]()

    def forward(self, x):
        """Forward pass."""
        x = self.norm(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class EncoderOutBlock(nn.Module):
    """Output block for encoder."""

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        groups: int,
        act: str,
        out_kernel_size: int,
    ):
        """Initialize the output block."""
        super().__init__()
        self.conv_norm_out = nn.GroupNorm(
            num_channels=in_channels, num_groups=groups, eps=1e-6
        )
        self.conv_act = ACTIVATION_FUNCTIONS[act]()

        self.conv_out = DIM_TO_CONV_MAP[dimensions](
            in_channels,
            out_channels,
            kernel_size=out_kernel_size,
            padding=1,
        )

    def forward(self, x):
        """Forward pass."""
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x


ENCODER_OUT_BLOCK_MAP = {
    "EncoderOutBlock": EncoderOutBlock,
    "SDVAEEncoderOutBlock": SDVAEEncoderOutBlock,
}


class Encoder(nn.Module):
    """Encoder implementation."""

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        n_channels: int,
        out_channels: int,
        down_block_types: tuple[str],
        block_out_channel_mults: tuple[int],
        mid_block_type: str,
        encoder_out_block_type: str,
        num_layers_per_block: int,
        res_args: dict = {},
        attn_args: dict = {},
        in_out_args: dict = {},
        double_z: bool = True,
    ) -> None:
        """Encoder implementation.

        Args:
            dimensions (int): Number of dimensions.
            in_channels (int): Number of input channels.
            n_channels (int): Number of channels for first down block.
            out_channels: int, Number of output channels (latent space).
            down_block_types (tuple[str]): Type of down blocks to use.
            block_out_channel_mults (tuple[int]): Multiplier for output channels of each block.
            mid_block_type (str): Type of mid block to use.
            encoder_out_block_type (str): Type of output block to use.
            num_layers_per_block (int): Number of layers per block.
            res_args (dict): Arguments for residual blocks.
            attn_args (dict): Arguments for attention blocks.
            in_out_args (dict): Arguments for input and output convolutions.
            double_z (bool): Whether to double the latent space.

        """
        super().__init__()

        n_mults = len(block_out_channel_mults)

        self.conv_in = DIM_TO_CONV_MAP[dimensions](
            in_channels,
            n_channels,
            kernel_size=in_out_args.get("kernel_size", 3),
            stride=1,
            padding=1,
        )

        self.down_blocks = nn.ModuleList()
        outs = ins = n_channels
        for i in range(n_mults):
            outs = ins * block_out_channel_mults[i]
            for _ in range(num_layers_per_block):
                down_block = BLOCK_MAP[down_block_types[i]](
                    dimensions=dimensions,
                    in_channels=ins,
                    out_channels=outs,
                    time_embedding=False,
                    time_channels=None,
                    res_args=res_args,
                    attn_args=attn_args,
                )
                self.down_blocks.append(down_block)
                ins = outs

            if i < n_mults - 1:
                self.down_blocks.append(Downsample(dimensions, ins))

        self.mid_block = BLOCK_MAP[mid_block_type](
            dimensions=dimensions,
            channels=outs,
            time_embedding=False,
            time_channels=None,
            res_args=res_args,
            attn_args=attn_args,
        )

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.out_block = ENCODER_OUT_BLOCK_MAP[encoder_out_block_type](
            dimensions=dimensions,
            in_channels=outs,
            out_channels=conv_out_channels,
            groups=in_out_args.get("groups", 8),
            act=in_out_args.get("act", "silu"),
            out_kernel_size=in_out_args.get("out_kernel_size", 3),
        )

    def forward(self, x):
        """Forward pass."""
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        x = self.out_block(x)
        return x
