"""Decoder for variational autoencoder implementation.

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
from chuchichaestli.models.blocks import BLOCK_MAP
from chuchichaestli.models.maps import DIM_TO_CONV_MAP
from chuchichaestli.models.upsampling import Upsample


class DecoderInBlock(nn.Module):
    """Input block for decoder."""

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        in_kernel_size: int,
        **kwargs,
    ):
        """Initialize the input block.

        Args:
            dimensions (int): Number of dimensions.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            in_kernel_size (int): Kernel size for input convolution.
            kwargs: Additional arguments.
        """
        super().__init__()
        self.conv = DIM_TO_CONV_MAP[dimensions](
            in_channels,
            out_channels,
            kernel_size=in_kernel_size,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        """Forward pass."""
        return self.conv(x)


class SDVAEDecoderInBlock(nn.Module):
    """SD-VAE-like input block for decoder."""

    def __init__(self, dimensions: int, in_channels: int, out_channels: int, **kwargs):
        """Initialize the input block.

        Args:
            dimensions (int): Number of dimensions.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kwargs: Additional arguments.
        """
        super().__init__()
        self.conv1 = DIM_TO_CONV_MAP[dimensions](
            in_channels, out_channels, kernel_size=1, stride=1, padding="same"
        )
        self.conv2 = DIM_TO_CONV_MAP[dimensions](
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        """Forward pass."""
        x = self.conv1(x)
        x = self.conv2(x)
        return x


DECODER_IN_BLOCK_MAP = {
    "DecoderInBlock": DecoderInBlock,
    "SDVAEDecoderInBlock": SDVAEDecoderInBlock,
}


class Decoder(nn.Module):
    """Decoder implementation."""

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        n_channels: int,
        out_channels: int,
        up_block_types: tuple[str],
        block_out_channel_mults: tuple[int],
        mid_block_type: str,
        decoder_in_block_type: str,
        num_layers_per_block: int,
        res_args: dict = {},
        attn_args: dict = {},
        in_out_args: dict = {},
    ) -> None:
        """Decoder implementation.

        Args:
            dimensions (int): Number of dimensions.
            in_channels (int): Number of input channels.
            n_channels (int): Number of input channels.
            out_channels: int, Number of output channels.
            up_block_types (tuple[str]): Type of up blocks to use.
            block_out_channel_mults (tuple[int]): Multiplier for output channels of each block.
            mid_block_type (str): Type of mid block to use.
            decoder_in_block_type (str): Type of input block to use.
            num_layers_per_block (int): Number of layers per block.
            res_args (dict): Arguments for residual blocks.
            attn_args (dict): Arguments for attention blocks.
            double_z (bool): Whether to double the latent space.
            groups (int): Number of groups for normalization.
            act (str): Activation function to use for output.
            in_out_args (dict): Arguments for input and output convolutions.
        """
        super().__init__()

        n_mults = len(block_out_channel_mults)

        outs = ins = n_channels
        for i in range(n_mults):
            outs = ins * block_out_channel_mults[i]
            ins = outs

        self.block_in = DECODER_IN_BLOCK_MAP[decoder_in_block_type](
            dimensions=dimensions,
            in_channels=in_channels,
            out_channels=outs,
            **in_out_args,
        )

        self.up_blocks = nn.ModuleList([])

        self.mid_block = BLOCK_MAP[mid_block_type](
            dimensions=dimensions,
            channels=outs,
            time_embedding=False,
            time_channels=None,
            res_args=res_args,
            attn_args=attn_args,
        )

        for i in reversed(range(n_mults)):
            outs = ins
            for _ in range(num_layers_per_block):
                up_block = BLOCK_MAP[up_block_types[i]](
                    dimensions=dimensions,
                    in_channels=ins,
                    out_channels=outs,
                    time_embedding=False,
                    time_channels=None,
                    res_args=res_args,
                    attn_args=attn_args,
                )
                self.up_blocks.append(up_block)

            outs = ins // block_out_channel_mults[i]
            up_block = BLOCK_MAP[up_block_types[i]](
                dimensions=dimensions,
                in_channels=ins,
                out_channels=outs,
                time_embedding=False,
                time_channels=None,
                res_args=res_args,
                attn_args=attn_args,
            )
            self.up_blocks.append(up_block)
            ins = outs
            if i > 0:
                self.up_blocks.append(Upsample(dimensions, outs))

        self.conv_norm_out = nn.GroupNorm(
            num_channels=n_channels, num_groups=in_out_args.get("groups", 4), eps=1e-6
        )
        self.conv_act = ACTIVATION_FUNCTIONS[in_out_args["act"]]()
        self.conv_out = DIM_TO_CONV_MAP[dimensions](
            n_channels,
            out_channels,
            kernel_size=in_out_args.get("out_kernel_size", 3),
            padding=1,
        )

    def forward(self, z):
        """Forward pass."""
        z = self.block_in(z)
        z = self.mid_block(z)
        for block in self.up_blocks:
            z = block(z)
        z = self.conv_norm_out(z)
        z = self.conv_act(z)
        z = self.conv_out(z)
        return z
