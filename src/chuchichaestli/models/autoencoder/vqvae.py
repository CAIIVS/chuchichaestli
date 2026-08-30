# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""A flexible vector-quantized variational autoencoder implementation."""

import torch
from torch import nn
from chuchichaestli.models.autoencoder.autoencoder import Autoencoder
from chuchichaestli.models.autoencoder.traits import DecoderLike, EncoderLike


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
        encoder: EncoderLike,
        decoder: DecoderLike,
        vq_dim: int = 64,
        vq_embeddings: int = 512,
    ):
        """Assemble a vector-quantized autoencoder from its two components.

        Args:
            encoder: Encoding component, mapping the input to latent space.
            decoder: Decoding component, expanding the latent space to the output.
            vq_dim: Size of the quantized embedding vectors.
            vq_embeddings: Size of the quantization codebook.
        """
        latent_dim = getattr(encoder, "latent_channels", encoder.out_channels)
        super().__init__(
            encoder,
            decoder,
            latent_proj=self.projection(encoder, latent_dim, vq_dim),
            latent_deproj=self.projection(decoder, vq_dim, latent_dim),
        )
        self.vq_dim = vq_dim
        self.quantize = VectorQuantizer(
            num_embeddings=vq_embeddings, embedding_dim=vq_dim
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
