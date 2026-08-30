# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Autoencoder implementations and utilities."""

from chuchichaestli.models.autoencoder.autoencoder import Autoencoder
from chuchichaestli.models.autoencoder.decoder import Decoder
from chuchichaestli.models.autoencoder.encoder import Encoder
from chuchichaestli.models.autoencoder.protocols import DecoderLike, EncoderLike
from chuchichaestli.models.autoencoder.vae import VAE, VAEDecoder, VAEEncoder
from chuchichaestli.models.autoencoder.vqvae import VQVAE, VectorQuantizer
from chuchichaestli.models.autoencoder.dcae import DCAE, DCDecoder, DCEncoder

__all__ = [
    "Decoder",
    "DecoderLike",
    "Encoder",
    "EncoderLike",
    "Autoencoder",
    "VAE",
    "VAEDecoder",
    "VAEEncoder",
    "VQVAE",
    "VectorQuantizer",
    "DCAE",
    "DCDecoder",
    "DCEncoder",
]
