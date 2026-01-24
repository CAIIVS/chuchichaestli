# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Encoder pipeline components."""

import torch
from torch import nn


__all__ = ["EncoderMixin", "EncoderInferencePipeline"]


class EncoderMixin:
    """Mixin for encoder functionality in pipelines.
    
    This mixin provides encoding capabilities that can be combined
    with other pipeline components through multiple inheritance.
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        **kwargs,
    ) -> None:
        """Initialize the encoder mixin.
        
        Args:
            encoder: Encoder network.
            **kwargs: Additional arguments passed to parent classes.
        """
        super().__init__(**kwargs)
        self.encoder = encoder

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode input to latent representation.
        
        Args:
            x: Input tensor.
            **kwargs: Additional arguments for encoder.
            
        Returns:
            Latent representation.
        """
        if self.encoder is None:
            raise ValueError("Encoder must be initialized.")
        return self.encoder(x, **kwargs)


class EncoderInferencePipeline(EncoderMixin):
    """Inference pipeline for encoders.
    
    Provides methods for encoding inputs to latent representations.
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        """Initialize the encoder inference pipeline.
        
        Args:
            encoder: Encoder network.
            device: Device to run inference on.
            **kwargs: Additional arguments.
        """
        super().__init__(encoder=encoder, **kwargs)
        self.device = device
        
        if self.encoder is not None:
            self.encoder.to(self.device)
            self.encoder.eval()

    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode input to latent representation.
        
        Args:
            x: Input tensor.
            **kwargs: Additional arguments.
            
        Returns:
            Latent representation.
        """
        return self.encode(x, **kwargs)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Call the pipeline for inference.
        
        Args:
            x: Input tensor.
            **kwargs: Additional arguments.
            
        Returns:
            Latent representation.
        """
        return self.predict(x, **kwargs)
