# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Modular, composable pipeline classes for inference and training.

This module provides base classes and mixins for building flexible pipelines
that can be combined through multiple inheritance to create specialized
training and inference workflows.

Example:
    Combining mixins to create a latent diffusion trainer:
    
    >>> from chuchichaestli.pipelines import AutoencoderMixin, DiffusionMixin, Trainer
    >>> 
    >>> class LatentDiffusionTrainer(AutoencoderMixin, DiffusionMixin, Trainer):
    ...     '''A trainer combining autoencoder, diffusion, and base training functionality.'''
    ...     pass
"""

from chuchichaestli.pipelines.autoencoder import (
    AutoencoderMixin,
    AutoencoderInferencePipeline,
)
from chuchichaestli.pipelines.cgan import (
    CGANMixin,
    CGANInferencePipeline,
    CGANTrainer,
)
from chuchichaestli.pipelines.decoder import (
    DecoderMixin,
    DecoderInferencePipeline,
)
from chuchichaestli.pipelines.diffusion import (
    DiffusionMixin,
    DiffusionInferencePipeline,
)
from chuchichaestli.pipelines.encoder import (
    EncoderMixin,
    EncoderInferencePipeline,
)
from chuchichaestli.pipelines.inference import InferencePipeline
from chuchichaestli.pipelines.train import Trainer


__all__ = [
    # Base classes
    "InferencePipeline",
    "Trainer",
    # Autoencoder components
    "AutoencoderMixin",
    "AutoencoderInferencePipeline",
    # Encoder components
    "EncoderMixin",
    "EncoderInferencePipeline",
    # Decoder components
    "DecoderMixin",
    "DecoderInferencePipeline",
    # Diffusion components
    "DiffusionMixin",
    "DiffusionInferencePipeline",
    # Conditional GAN components
    "CGANMixin",
    "CGANInferencePipeline",
    "CGANTrainer",
]
