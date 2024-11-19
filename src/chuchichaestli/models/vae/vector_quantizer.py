"""Vector Quantizer for VQ-VAE.

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


class VectorQuantizer(nn.Module):
    """Vector Quantizer for VQ-VAE."""

    def __init__(self, n_embeddings: int, embedding_dim: int, beta: float = 0.25):
        """Initialize VectorQuantizer.

        Args:
            n_embeddings (int): Number of embeddings.
            embedding_dim (int): Dimension of the embeddings.
            beta (float): Beta parameter for the loss.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        self.beta = beta
        self.embedding.weight.data.uniform_(-1 / n_embeddings, 1 / n_embeddings)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """Forward pass through the VQVAE."""
        z = z.moveaxis(1, -1).contiguous()
        z_shape = z.shape
        z_flat = z.view(-1, self.embedding_dim)

        nearest_embs = torch.argmin(torch.cdist(z_flat, self.embedding.weight), dim=1)
        z_q = self.embedding(nearest_embs).view(z_shape)
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )
        # This preserves the gradient flow.
        z_q: torch.Tensor = z + (z_q - z).detach()
        z_q = z_q.moveaxis(-1, 1).contiguous()

        counts = torch.bincount(nearest_embs, minlength=self.embedding.num_embeddings)
        e_mean = counts.float() / nearest_embs.numel()
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return z_q, loss, (perplexity, nearest_embs, z_flat)

    def get_codebook_entry(
        self, indices: torch.LongTensor, shape: tuple[int, ...] | None = None
    ) -> torch.Tensor:
        """Get the codebook entry for the given indices."""
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.moveaxis(-1, 1).contiguous()
        return z_q
