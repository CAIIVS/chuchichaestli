# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decoder pipeline components."""

import torch
from torch import nn


__all__ = ["DecoderMixin", "DecoderInferencePipeline"]


class DecoderMixin:
    """Mixin for decoder functionality in pipelines.
    
    This mixin provides decoding capabilities that can be combined
    with other pipeline components through multiple inheritance.
    """

    def __init__(
        self,
        decoder: nn.Module | None = None,
        **kwargs,
    ) -> None:
        """Initialize the decoder mixin.
        
        Args:
            decoder: Decoder network.
            **kwargs: Additional arguments passed to parent classes.
        """
        super().__init__(**kwargs)
        self.decoder = decoder

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode latent representation to output.
        
        Args:
            z: Latent tensor.
            **kwargs: Additional arguments for decoder.
            
        Returns:
            Decoded output.
        """
        if self.decoder is None:
            raise ValueError("Decoder must be initialized.")
        return self.decoder(z, **kwargs)


class DecoderInferencePipeline(DecoderMixin):
    """Inference pipeline for decoders.
    
    Provides methods for decoding latent representations to outputs.
    """

    def __init__(
        self,
        decoder: nn.Module | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        """Initialize the decoder inference pipeline.
        
        Args:
            decoder: Decoder network.
            device: Device to run inference on.
            **kwargs: Additional arguments.
        """
        super().__init__(decoder=decoder, **kwargs)
        self.device = device
        
        if self.decoder is not None:
            self.decoder.to(self.device)
            self.decoder.eval()

    def predict(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode latent representation to output.
        
        Args:
            z: Latent tensor.
            **kwargs: Additional arguments.
            
        Returns:
            Decoded output.
        """
        return self.decode(z, **kwargs)

    @torch.no_grad()
    def __call__(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Call the pipeline for inference.
        
        Args:
            z: Latent tensor.
            **kwargs: Additional arguments.
            
        Returns:
            Decoded output.
        """
        return self.predict(z, **kwargs)
