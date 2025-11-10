# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""A flexible vector-quantized variational autoencoder implementation."""

import torch
from torch import nn
from chuchichaestli.models.activations import ActivationTypes
from chuchichaestli.models.autoencoder.autoencoder import Autoencoder
from chuchichaestli.models.blocks import (
    AutoencoderDownBlockTypes,
    AutoencoderMidBlockTypes,
    AutoencoderUpBlockTypes,
    EncoderOutBlockTypes,
    DecoderInBlockTypes,
)
from chuchichaestli.models.downsampling import DownsampleTypes
from chuchichaestli.models.maps import DIM_TO_CONV_MAP
from chuchichaestli.models.norm import NormTypes
from chuchichaestli.models.upsampling import UpsampleTypes
from collections.abc import Sequence


__all__ = ["VectorQuantizer", "VQVAE"]


class VectorQuantizer(nn.Module):
    """Vector Quantizer for VQVAE."""

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        """Initialize VectorQuantizer.

        Args:
            num_embeddings: Size of the codebook.
            embedding_dim: Size of the embedding vectors.
            beta: Commitment cost parameter for the loss.
        """
        super().__init__()
        self.beta = beta
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    @property
    def num_embeddings(self) -> int:
        """Size of the embedding dictionary."""
        return self.embedding.num_embeddings

    @property
    def embedding_dim(self) -> int:
        """Size of each embedding vector."""
        return self.embedding.embedding_dim

    def forward(
        self, z: torch.Tensor, codebook_usage: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Forward pass through the vector quantizer (codebook lookup).

        Args:
            z: Input latent codes of shape `(B, embedding_dim, *S)`.
            codebook_usage: If `True`, compute and return codebook usage parameters.

        Returns:
            - Quantized latent code
            - Quantization loss
            - Codebook usage parameters (perplexity, flattened latent, codebook indices)
        """
        z = z.moveaxis(1, -1).contiguous()
        z_flat = z.view(-1, self.embedding_dim)
        # quantize
        e = self.embedding.weight
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + e.pow(2).sum(dim=1)
            - 2 * (z_flat @ e.T)
        )  # rather compute square distances (avoiding sqrt computation of torch.cdist)
        nearest_emb_idcs = torch.argmin(distances, dim=1)
        z_q = self.embedding(nearest_emb_idcs).view(z.shape)
        # loss calculation
        commitment_loss = (z_q.detach() - z).pow(2).mean()
        codebook_loss = (z_q - z.detach()).pow(2).mean()
        loss = self.beta * commitment_loss + codebook_loss
        # preserve gradient flow
        z_q = z + (z_q - z).detach()
        z_q = z_q.moveaxis(-1, 1).contiguous()
        # calculate codebook usage
        perplexity: torch.Tensor | None = None
        if codebook_usage:
            counts = torch.bincount(
                nearest_emb_idcs, minlength=self.embedding.num_embeddings
            )
            avg_probs = counts.float() / counts.sum()
            mask = avg_probs > 0
            entropy = torch.sum(avg_probs[mask] * torch.log(avg_probs[mask]))
            perplexity = torch.exp(-entropy)
        return (
            z_q,
            loss,
            {
                "perplexity": perplexity,
                "latent_flat": z_flat,
                "indices": nearest_emb_idcs,
            },
        )

    def get_codebook_entry(
        self, indices: torch.LongTensor, shape: tuple[int, ...] | None = None
    ) -> torch.Tensor:
        """Get the codebook entry for the given indices.

        Args:
            indices: Indices from the codebook.
            shape: Full latent shape (with the embedding dimension in the last axis).
        """
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.moveaxis(-1, 1).contiguous()
        return z_q


class VQVAE(Autoencoder):
    """Flexible vector-quantized variational autoencoder implementation.

    The architecture consists of an encoder-decoder structure with a codebook bottleneck.
    The encoder chains several residual and downsampling blocks.
    Each downsampling block separates the encoder into spatially hierarchical levels.
    The encoder ends in bottleneck blocks (optionally including attention blocks
    and a convolutional layer). The latent code is quantized with the codebook.
    The decoder is built with residual convolutional and upsampling blocks, and
    expands from the (quantized) latent space to the image domain.
    """

    def __init__(
        self,
        dimensions: int = 2,
        in_channels: int = 1,
        n_channels: int = 64,
        latent_dim: int = 4,
        vq_dim: int = 64,
        vq_embeddings: int = 512,
        out_channels: int = 1,
        down_block_types: Sequence[AutoencoderDownBlockTypes] = (
            "AutoencoderDownBlock",
            "AutoencoderDownBlock",
            "AutoencoderDownBlock",
            "AutoencoderDownBlock",
        ),
        down_layers_per_block: int | Sequence[int] = 2,
        downsample_type: DownsampleTypes = "Downsample",
        encoder_mid_block_types: Sequence[AutoencoderMidBlockTypes] = (
            "AutoencoderMidBlock",
            "AttnAutoencoderMidBlock",
        ),
        encoder_out_block_type: EncoderOutBlockTypes = "EncoderOutBlock",
        decoder_in_block_type: DecoderInBlockTypes = "DecoderInBlock",
        decoder_mid_block_types: Sequence[AutoencoderMidBlockTypes] = (
            "AutoencoderMidBlock",
            "AttnAutoencoderMidBlock",
        ),
        up_block_types: AutoencoderUpBlockTypes = (
            "AutoencoderUpBlock",
            "AutoencoderUpBlock",
            "AutoencoderUpBlock",
            "AutoencoderUpBlock",
        ),
        up_layers_per_block: int | Sequence[int] = 3,
        upsample_type: UpsampleTypes = "UpsampleInterpolate",
        block_out_channel_mults: Sequence[int] = (1, 2, 2, 2),
        res_act_fn: ActivationTypes = "silu",
        res_dropout: float = 0.0,
        res_norm_type: NormTypes = "group",
        res_groups: int = 8,
        res_kernel_size: int = 3,
        attn_head_dim: int = 32,
        attn_n_heads: int = 1,
        attn_dropout_p: float = 0.0,
        attn_norm_type: NormTypes = "group",
        attn_groups: int = 32,
        attn_kernel_size: int = 1,
        attn_scales: Sequence[int] = (5,),
        encoder_act_fn: ActivationTypes = "silu",
        encoder_norm_type: NormTypes = "group",
        encoder_groups: int = 8,
        encoder_kernel_size: int = 3,
        decoder_act_fn: ActivationTypes = "silu",
        decoder_norm_type: NormTypes = "group",
        decoder_groups: int = 8,
        decoder_kernel_size: int = 3,
    ):
        """Initializes the VAE model with the given parameters.

        Args:
            dimensions: Number of dimensions for the model.
            in_channels: Number of input channels.
            n_channels: Number of channels in the hidden layer.
            vq_emb_dim: Number of channels in the
            latent_dim: Number of channels in the latent space.
            vq_dim: Size of the quantized embedding vectors.
            vq_embeddings: Size of the quantization codebook.
            out_channels: Number of output channels.
            down_block_types: Types of down block(s) to use for each level.
            down_layers_per_block: Number of blocks per level in the encoder
                (blocks are repeated if `>1`).
            downsample_type: Type of downsampling block
                (see `chuchichaestli.models.downsampling` for details).
            encoder_mid_block_types: Types of middle block(s) in the encoder.
            encoder_out_block_type: Type of output block in the encoder.
            decoder_in_block_type: Type of input block in the decoder
            decoder_mid_block_types: Types of middle block(s) in the decoder.
            up_block_types: Type of up block(s) to use for each level.
            up_layers_per_block: Number of blocks per level in the decoder
                (blocks are repeated if `>1`).
            upsample_type: Type of upsampling block
                (see `chuchichaestli.models.upsampling` for details).
            block_out_channel_mults: Multiplier for output channels of each level block.
            use_latent_proj: Whether to use a linear layer between encoder and latent space.
            use_latent_deproj: Whether to use a linear layer between latent space and decoder.
            res_act_fn: Activation function for the residual blocks
                (see `chuchichaestli.models.activations` for details).
            res_dropout: Dropout rate for the residual blocks.
            res_norm_type: Normalization type for the residual block
                (see `chuchichaestli.models.norm` for details).
            res_groups: Number of groups for the residual block normalization (if group norm).
            res_kernel_size: Kernel size for the residual blocks.
            attn_head_dim: Dimension of the attention heads.
            attn_n_heads: Number of attention heads.
            attn_dropout_p: Dropout probability of the scaled dot product attention.
            attn_norm_type: Normalization type for the convolutional attention block
                (see `chuchichaestli.models.norm` for details).
            attn_groups: Number of groups for the convolutional attention block normalization
                (if `attn_norm_type` is `"group"`).
            attn_kernel_size: Kernel size for the convolutional attention block.
            attn_scales: Scales for the multi-scale attention block.
            encoder_act_fn: Activation function for the output layers in the encoder
                (see `chuchichaestli.models.activations` for details).
            encoder_norm_type: Normalization type for the encoder's output block
                (see `chuchichaestli.models.norm` for details).
            encoder_groups: Number of groups for normalization in the output layer of the encoder.
            encoder_kernel_size: Kernel size for the output convolution in the encoder.
            decoder_act_fn: Activation function for the input/output layers in the decoder
                (see `chuchichaestli.models.activations` for details).
            decoder_norm_type: Normalization type for the decoder's output block
                (see `chuchichaestli.models.norm` for details).
            decoder_groups: Number of groups for normalization in the input/output layer of the decoder.
            decoder_kernel_size: Kernel size for the output convolution in the decoder.
            double_z: Whether to double the latent space.
        """
        super().__init__(
            dimensions=dimensions,
            in_channels=in_channels,
            n_channels=n_channels,
            latent_dim=latent_dim,
            out_channels=out_channels,
            down_block_types=down_block_types,
            down_layers_per_block=down_layers_per_block,
            downsample_type=downsample_type,
            encoder_mid_block_types=encoder_mid_block_types,
            encoder_out_block_type=encoder_out_block_type,
            decoder_in_block_type=decoder_in_block_type,
            decoder_mid_block_types=decoder_mid_block_types,
            up_block_types=up_block_types,
            up_layers_per_block=up_layers_per_block,
            upsample_type=upsample_type,
            block_out_channel_mults=block_out_channel_mults,
            use_latent_proj=False,
            use_latent_deproj=False,
            res_act_fn=res_act_fn,
            res_dropout=res_dropout,
            res_norm_type=res_norm_type,
            res_groups=res_groups,
            res_kernel_size=res_kernel_size,
            attn_head_dim=attn_head_dim,
            attn_n_heads=attn_n_heads,
            attn_dropout_p=attn_dropout_p,
            attn_norm_type=attn_norm_type,
            attn_groups=attn_groups,
            attn_kernel_size=attn_kernel_size,
            encoder_act_fn=encoder_act_fn,
            encoder_norm_type=encoder_norm_type,
            encoder_groups=encoder_groups,
            encoder_kernel_size=encoder_kernel_size,
            decoder_act_fn=decoder_act_fn,
            decoder_norm_type=decoder_norm_type,
            decoder_groups=decoder_groups,
            decoder_kernel_size=decoder_kernel_size,
            double_z=False,
        )
        self.vq_dim = vq_dim
        self.latent_proj = DIM_TO_CONV_MAP[dimensions](
            self.latent_dim,
            self.vq_dim,
            kernel_size=1,
            stride=1,
            padding="same",
        )
        self.quantize = VectorQuantizer(
            num_embeddings=vq_embeddings, embedding_dim=vq_dim
        )
        self.latent_deproj = DIM_TO_CONV_MAP[dimensions](
            self.vq_dim,
            self.latent_dim,
            kernel_size=1,
            stride=1,
            padding="same",
        )

    def compute_embedding_shape(
        self, input_shape: tuple[int, ...], no_batch_dim: bool = False
    ):
        """Compute the shape of the latent space."""
        batch_dim = input_shape[0] if not no_batch_dim else None
        spatial_dims = tuple(dim // self.f_comp for dim in input_shape[2:])
        if batch_dim is None:
            shape = (self.vq_dim, *spatial_dims)
        else:
            shape = (batch_dim, self.vq_dim, *spatial_dims)
        return shape

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input without the codebook."""
        z = super().encode(x)
        return z

    def encode(
        self,
        x: torch.Tensor | torch.LongTensor,
        codebook_usage: bool = False,
        force_no_quant: bool = False,
        load_from_codebook: bool = False,
        shape: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Encode the input.

        Args:
            x: Input tensor or codebook indices (if `load_from_codebook` and `force_no_quant`).
            codebook_usage: If `True`, compute and return codebook usage parameters.
            force_no_quant: If `True`, quantization of the latent code is skipped.
            load_from_codebook: If `True`, indices are loaded from the codebook instead.
            shape: Full latent shape for codebook entries
                (with the embedding dimension in the last axis).

        Returns:
            - Quantized latent code
            - Quantization loss
            - Codebook usage parameters (perplexity, flattened latent, codebook indices)
        """
        z = self._encode(x)
        if not force_no_quant:
            z_q, loss, usage = self.quantize(z, codebook_usage=codebook_usage)
        elif load_from_codebook:
            z_q = self.quantize.get_codebook_entry(x, shape)
            loss = torch.as_tensor(0.0, device=z.device, dtype=z.dtype)
            usage = {}
        else:
            z_q = z
            loss = torch.as_tensor(0.0, device=z.device, dtype=z.dtype)
            usage = {}
        return z_q, loss, usage

    def forward(
        self,
        x: torch.Tensor | torch.LongTensor,
        codebook_usage: bool = False,
        force_no_quant: bool = False,
        load_from_codebook: bool = False,
        shape: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Forward pass through the model.

        Args:
            x: Input tensor or codebook indices (if `load_from_codebook` and `force_no_quant`).
            codebook_usage: If `True`, compute and return codebook usage parameters.
            force_no_quant: If `True`, quantization of the latent code is skipped.
            load_from_codebook: If `True`, indices are loaded from the codebook instead.
            shape: Full latent shape for codebook entries
                (with the embedding dimension in the last axis).

        Returns:
            - Quantized latent code
            - Quantization loss
            - Codebook usage parameters (perplexity, flattened latent, codebook indices)
        """
        z_q, loss, usage = self.encode(
            x,
            codebook_usage=codebook_usage,
            force_no_quant=force_no_quant,
            load_from_codebook=load_from_codebook,
            shape=shape,
        )
        return self.decode(z_q), loss, usage
