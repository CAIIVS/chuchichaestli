# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""A flexible variational autoencoder implementation."""

import torch
from torch import nn
from torch.distributions import MultivariateNormal, kl
from chuchichaestli.models.autoencoder.autoencoder import Autoencoder
from chuchichaestli.models.autoencoder.decoder import Decoder
from chuchichaestli.models.autoencoder.encoder import Encoder
from chuchichaestli.models.autoencoder.protocols import DecoderLike, EncoderLike


__all__ = ["VAE", "VAEDecoder", "VAEEncoder"]


class VAEEncoder(Encoder):
    """Encoding component of a variational autoencoder.

    Doubles the latent channels so that a mean and a variance can be read from
    them; see `Encoder` for the architecture arguments.
    """

    def __init__(self, *args, double_z: bool = True, **kwargs):
        """Constructor.

        Args:
            args: Positional architecture arguments for `Encoder`.
            double_z: Whether to double the latent space; a VAE requires `True`.
            kwargs: Keyword architecture arguments for `Encoder`.

        Raises:
            ValueError: If the doubling is disabled.
        """
        if not double_z:
            raise ValueError(
                "VAEEncoder doubles the latent channels to carry mean and variance;"
                " use a plain Encoder for a single latent code."
            )
        super().__init__(*args, double_z=True, **kwargs)


class VAEDecoder(Decoder):
    """Decoding component of a variational autoencoder.

    Consumes a latent code that sampling has already collapsed to a single set
    of channels, so structurally it is a plain `Decoder`; see there for the
    architecture arguments. It exists as the component a likelihood decoder would
    occupy, emitting a mean and a variance per output channel the way
    `VAEEncoder` doubles its latent channels on the way in.
    """


class VAE(Autoencoder):
    """Flexible variational autoencoder implementation.

    The architecture consists of an encoder-decoder structure.
    The encoder chains several residual and downsampling blocks.
    Each downsampling block separates the encoder into spatially hierarchical levels.
    The encoder ends in bottleneck blocks (optionally including attention blocks
    and a convolutional layer) and projects the input into latent space.
    Latent mean and variance are sampled from a Gaussian and passed to the decoder.
    The decoder is built with residual convolutional and upsampling blocks, and
    expands from the latent space to the image domain.

    Attributes:
        encoder_cls: Encoder class that `build` instantiates.
        decoder_cls: Decoder class that `build` instantiates.
    """

    encoder_cls: type = VAEEncoder
    decoder_cls: type = VAEDecoder

    def __init__(
        self,
        encoder: EncoderLike,
        decoder: DecoderLike,
        latent_proj: nn.Module | bool = True,
        latent_deproj: nn.Module | bool = True,
    ):
        """Assemble a variational autoencoder from its two components.

        Args:
            encoder: Encoding component; must double its latent channels.
            decoder: Decoding component, expanding the latent space to the output.
            latent_proj: Projection between encoder and latent space (see `Autoencoder`).
            latent_deproj: Projection between latent space and decoder.

        Raises:
            ValueError: If the encoder does not double its latent channels.
        """
        if not getattr(encoder, "double_z", False):
            raise ValueError(
                "VAE reads a mean and a variance from the latent channels; pass an"
                " encoder that doubles them (e.g. VAEEncoder, or double_z=True)."
            )
        super().__init__(encoder, decoder, latent_proj, latent_deproj)
        self.softplus = nn.Softplus()

    def encode(self, x: torch.Tensor, eps: float = 1e-12) -> MultivariateNormal:
        """Encode the input.

        Args:
            x: Input tensor.
            eps: Small constant value for numerical stability.

        Returns:
            Multivariate normal posterior distribution
        """
        z = self.encoder(x)
        z = self.latent_proj(z) if self.latent_proj is not None else z
        mean, log_var = z.chunk(2, dim=1)
        scale = self.softplus(log_var) + eps
        scale_tril = torch.diag_embed(scale)
        return MultivariateNormal(mean, scale_tril=scale_tril)

    def forward(
        self, x: torch.Tensor, sample_posterior: bool = True, eps: float = 1e-12
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model."""
        posterior = self.encode(x, eps=eps)
        if sample_posterior:
            z = posterior.rsample()
        else:
            z = posterior.mode()
        return self.decode(z), posterior

    @staticmethod
    def kl_divergence(posterior: torch.distributions.MultivariateNormal):
        """Compute the KL divergence between posterior and a multivariate Gaussian."""
        device = posterior.mean.device
        dtype = posterior.mean.dtype
        zeros = torch.zeros_like(posterior.mean)
        eye = torch.eye(posterior.mean.shape[-1], device=device, dtype=dtype)
        return kl.kl_divergence(posterior, MultivariateNormal(zeros, eye))
