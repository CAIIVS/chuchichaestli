# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Autoencoder pipeline components."""

import torch
from torch import nn


__all__ = ["AutoencoderMixin", "AutoencoderInferencePipeline"]


class AutoencoderMixin:
    """Mixin for autoencoder functionality in pipelines.
    
    This mixin provides encoding and decoding capabilities that can be
    combined with other pipeline components through multiple inheritance.
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
        **kwargs,
    ) -> None:
        """Initialize the autoencoder mixin.
        
        Args:
            encoder: Encoder network.
            decoder: Decoder network.
            **kwargs: Additional arguments passed to parent classes.
        """
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

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

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode latent representation to output.
        
        Args:
            z: Latent tensor.
            **kwargs: Additional arguments for decoder.
            
        Returns:
            Reconstructed output.
        """
        if self.decoder is None:
            raise ValueError("Decoder must be initialized.")
        return self.decoder(z, **kwargs)

    def reconstruct(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode and decode input for reconstruction.
        
        Args:
            x: Input tensor.
            **kwargs: Additional arguments.
            
        Returns:
            Reconstructed output.
        """
        z = self.encode(x, **kwargs)
        return self.decode(z, **kwargs)


class AutoencoderInferencePipeline(AutoencoderMixin):
    """Inference pipeline for autoencoders.
    
    Provides methods for encoding, decoding, and reconstruction during inference.
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        """Initialize the autoencoder inference pipeline.
        
        Args:
            encoder: Encoder network.
            decoder: Decoder network.
            device: Device to run inference on.
            **kwargs: Additional arguments.
        """
        super().__init__(encoder=encoder, decoder=decoder, **kwargs)
        self.device = device
        
        if self.encoder is not None:
            self.encoder.to(self.device)
            self.encoder.eval()
        if self.decoder is not None:
            self.decoder.to(self.device)
            self.decoder.eval()

    def predict(self, x: torch.Tensor, mode: str = "reconstruct", **kwargs) -> torch.Tensor:
        """Generate predictions.
        
        Args:
            x: Input tensor.
            mode: Prediction mode ('encode', 'decode', or 'reconstruct').
            **kwargs: Additional arguments.
            
        Returns:
            Predictions based on the specified mode.
        """
        if mode == "encode":
            return self.encode(x, **kwargs)
        elif mode == "decode":
            return self.decode(x, **kwargs)
        elif mode == "reconstruct":
            return self.reconstruct(x, **kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'encode', 'decode', or 'reconstruct'.")

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, mode: str = "reconstruct", **kwargs) -> torch.Tensor:
        """Call the pipeline for inference.
        
        Args:
            x: Input tensor.
            mode: Prediction mode.
            **kwargs: Additional arguments.
            
        Returns:
            Predictions.
        """
        return self.predict(x, mode=mode, **kwargs)
