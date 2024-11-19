"""Vector Quantized-Variational AutoEncoder implementation.

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
from torch import nn

from chuchichaestli.models.vae.decoder import Decoder
from chuchichaestli.models.vae.encoder import Encoder
from chuchichaestli.models.maps import DIM_TO_CONV_MAP
from chuchichaestli.models.vae.vector_quantizer import VectorQuantizer


class VQVAE(nn.Module):
    """Vector Quantised-Variational AutoEncoder implementation."""

    def __init__(
        self,
        dimensions: int = 2,
        in_channels: int = 3,
        n_channels: int = 32,
        latent_dim: int = 4,
        vq_n_embeddings: int = 256,
        vq_embedding_dim: int = 32,
        down_block_types: tuple[str] = ("EncoderDownBlock",),
        mid_block_type: str = "EncoderMidBlock",
        up_block_types: tuple[str] = ("EncoderUpBlock",),
        encoder_out_block_type: str = "EncoderOutBlock",
        decoder_in_block_type: str = "DecoderInBlock",
        block_out_channel_mults: tuple[int] = (2,),
        num_layers_per_block: int = 1,
        res_groups: int = 4,
        res_act_fn: str = "silu",
        res_dropout: float = 0.1,
        res_norm_type: str = "group",
        res_kernel_size: int = 3,
        attn_head_dim: int = 32,
        attn_n_heads: int = 1,
        attn_gate_inter_channels: int = 32,
        encoder_norm_num_groups: int = 4,
        encoder_act_fn: str = "silu",
        encoder_in_kernel_size: int = 3,
        encoder_out_kernel_size: int = 3,
        decoder_norm_num_groups: int = 4,
        decoder_act_fn: str = "silu",
        decoder_in_kernel_size: int = 3,
        decoder_out_kernel_size: int = 3,
        double_z: bool = True,
    ):
        """Initializes the VQVAE model.

        Args:
            dimensions (int): Number of dimensions for the model. Default is 2.
            in_channels (int): Number of input channels. Default is 3.
            n_channels (int): Number of channels in the model. Default is 32.
            latent_dim (int): Dimension of the latent space. Default is 4.
            vq_n_embeddings (int): Number of embeddings for the vector quantizer. Default is 256.
            vq_embedding_dim (int): Dimension of the embeddings for the vector quantizer. Default is 32.
            out_channels (int): Number of output channels. Default is 3.
            down_block_types (tuple[str]): Types of downsampling blocks. Default is ("DownBlock",).
            mid_block_type (str): Type of middle block. Default is "MidBlock".
            up_block_types (tuple[str]): Types of upsampling blocks. Default is ("UpBlock",).
            encoder_out_block_type (str): Type of output block for the encoder. Default is "EncoderOutBlock".
            decoder_in_block_type (str): Type of input block for the decoder. Default is "DecoderInBlock".
            block_out_channel_mults (tuple[int]): Multipliers for the number of output channels in each block. Default is (2,).
            num_layers_per_block (int): Number of layers per block. Default is 1.
            res_groups (int): Number of groups for the residual blocks. Default is 32.
            res_act_fn (str): Activation function for the residual blocks. Default is "silu".
            res_dropout (float): Dropout rate for the residual blocks. Default is 0.1.
            res_norm_type (str): Normalization type for the residual blocks. Default is "group".
            res_kernel_size (int): Kernel size for the residual blocks. Default is 3.
            attn_head_dim (int): Dimension of each attention head. Default is 32.
            attn_n_heads (int): Number of attention heads. Default is 1.
            attn_gate_inter_channels (int): Number of intermediate channels for the attention gate. Default is 32.
            encoder_norm_num_groups (int): Number of groups for normalization in the encoder. Default is 8.
            encoder_act_fn (str): Activation function for the encoder. Default is "silu".
            encoder_in_kernel_size (int): Kernel size for the input layer of the encoder. Default is 3.
            encoder_out_kernel_size (int): Kernel size for the output layer of the encoder. Default is 3.
            decoder_norm_num_groups (int): Number of groups for normalization in the decoder. Default is 8.
            decoder_act_fn (str): Activation function for the decoder. Default is "silu".
            decoder_in_kernel_size (int): Kernel size for the input layer of the decoder. Default is 3.
            decoder_out_kernel_size (int): Kernel size for the output layer of the decoder. Default is 3.
            double_z (bool): Whether to use double the latent space dimensions. Default is True.
        """
        super().__init__()

        if encoder_out_block_type == "DeepCompressionEncoderOutBlock":
            assert (
                dimensions == 2
            ), "DeepCompressionEncoderOutBlock only supports 2D data."

        res_args = {
            "res_groups": res_groups,
            "res_act_fn": res_act_fn,
            "res_dropout": res_dropout,
            "res_norm_type": res_norm_type,
            "res_kernel_size": res_kernel_size,
        }

        attn_args = {
            "n_heads": attn_n_heads,
            "head_dim": attn_head_dim,
            "inter_channels": attn_gate_inter_channels,
        }

        encoder_args = {
            "groups": encoder_norm_num_groups,
            "act": encoder_act_fn,
            "in_kernel_size": encoder_in_kernel_size,
            "out_kernel_size": encoder_out_kernel_size,
        }

        decoder_args = {
            "groups": decoder_norm_num_groups,
            "act": decoder_act_fn,
            "in_kernel_size": decoder_in_kernel_size,
            "out_kernel_size": decoder_out_kernel_size,
        }

        self.encoder = Encoder(
            dimensions=dimensions,
            in_channels=in_channels,
            n_channels=n_channels,
            out_channels=latent_dim,
            down_block_types=down_block_types,
            block_out_channel_mults=block_out_channel_mults,
            mid_block_type=mid_block_type,
            encoder_out_block_type=encoder_out_block_type,
            num_layers_per_block=num_layers_per_block,
            res_args=res_args,
            attn_args=attn_args,
            in_out_args=encoder_args,
            double_z=double_z,
        )
        self.softplus = nn.Softplus()
        self.decoder = Decoder(
            dimensions=dimensions,
            in_channels=latent_dim,
            n_channels=n_channels,
            out_channels=in_channels,
            up_block_types=up_block_types,
            block_out_channel_mults=block_out_channel_mults,
            mid_block_type=mid_block_type,
            decoder_in_block_type=decoder_in_block_type,
            num_layers_per_block=num_layers_per_block,
            res_args=res_args,
            attn_args=attn_args,
            in_out_args=decoder_args,
        )

        self.quant_conv = DIM_TO_CONV_MAP[dimensions](
            latent_dim, vq_embedding_dim, kernel_size=1
        )
        self.quantize = VectorQuantizer(
            n_embeddings=vq_n_embeddings, embedding_dim=vq_embedding_dim
        )
        self.post_quant_conv = DIM_TO_CONV_MAP[dimensions](
            vq_embedding_dim, latent_dim, kernel_size=1
        )
        self.levels = len(block_out_channel_mults)

    def encode(self, x: torch.Tensor):
        """Encode the input tensor."""
        z = self.encoder(x)
        z = self.quant_conv(z)
        return z

    def decode(
        self,
        z: torch.Tensor,
        force_no_quant: bool = False,
        load_from_codebook: bool = False,
        shape: tuple[int, ...] | None = None,
    ):
        """Decode the input tensor."""
        if not force_no_quant:
            z, loss, _ = self.quantize(z)
        elif load_from_codebook:
            z = self.quantize.get_codebook_entry(z, shape)
            loss = torch.zeros(z.shape[0]).to(z.device, dtype=z.dtype)
        else:
            loss = torch.zeros(z.shape[0]).to(z.device, dtype=z.dtype)

        z = self.post_quant_conv(z)
        x_hat = self.decoder(z)
        return x_hat, loss

    def forward(
        self,
        x: torch.Tensor,
        force_no_quant: bool = False,
        load_from_codebook: bool = False,
    ):
        """Forward pass."""
        z = self.encode(x)
        x_hat, loss = self.decode(
            z, force_no_quant=force_no_quant, load_from_codebook=load_from_codebook
        )
        return x_hat, loss
