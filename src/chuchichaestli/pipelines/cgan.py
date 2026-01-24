# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Conditional GAN pipeline components."""

from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


__all__ = ["CGANMixin", "CGANInferencePipeline", "CGANTrainer"]


class CGANMixin:
    """Mixin for conditional GAN functionality in pipelines.
    
    This mixin provides generator and discriminator handling for conditional GANs
    that can be combined with other pipeline components through multiple inheritance.
    """

    def __init__(
        self,
        generator: nn.Module | None = None,
        discriminator: nn.Module | None = None,
        **kwargs,
    ) -> None:
        """Initialize the conditional GAN mixin.
        
        Args:
            generator: Generator network.
            discriminator: Discriminator network.
            **kwargs: Additional arguments passed to parent classes.
        """
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator

    def generate(
        self,
        noise: torch.Tensor,
        conditions: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples from noise and optional conditions.
        
        Args:
            noise: Input noise tensor.
            conditions: Optional conditioning information.
            **kwargs: Additional arguments for generator.
            
        Returns:
            Generated samples.
        """
        if self.generator is None:
            raise ValueError("Generator must be initialized.")
        
        if conditions is not None:
            return self.generator(noise, conditions, **kwargs)
        return self.generator(noise, **kwargs)

    def discriminate(
        self,
        x: torch.Tensor,
        conditions: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Discriminate real vs. fake samples.
        
        Args:
            x: Input samples.
            conditions: Optional conditioning information.
            **kwargs: Additional arguments for discriminator.
            
        Returns:
            Discriminator predictions.
        """
        if self.discriminator is None:
            raise ValueError("Discriminator must be initialized.")
        
        if conditions is not None:
            return self.discriminator(x, conditions, **kwargs)
        return self.discriminator(x, **kwargs)


class CGANInferencePipeline(CGANMixin):
    """Inference pipeline for conditional GANs.
    
    Provides methods for generating samples using trained conditional GANs.
    """

    def __init__(
        self,
        generator: nn.Module | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        """Initialize the conditional GAN inference pipeline.
        
        Args:
            generator: Generator network.
            device: Device to run inference on.
            **kwargs: Additional arguments.
        """
        super().__init__(generator=generator, **kwargs)
        self.device = device
        
        if self.generator is not None:
            self.generator.to(self.device)
            self.generator.eval()

    def predict(
        self,
        batch_size: int = 1,
        latent_dim: int = 100,
        conditions: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples.
        
        Args:
            batch_size: Number of samples to generate.
            latent_dim: Dimension of latent noise.
            conditions: Optional conditioning information.
            **kwargs: Additional arguments.
            
        Returns:
            Generated samples.
        """
        noise = torch.randn(batch_size, latent_dim, device=self.device)
        return self.generate(noise, conditions=conditions, **kwargs)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        latent_dim: int = 100,
        conditions: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Call the pipeline for inference.
        
        Args:
            batch_size: Number of samples to generate.
            latent_dim: Dimension of latent noise.
            conditions: Optional conditioning information.
            **kwargs: Additional arguments.
            
        Returns:
            Generated samples.
        """
        return self.predict(batch_size=batch_size, latent_dim=latent_dim, conditions=conditions, **kwargs)


class CGANTrainer(CGANMixin):
    """Training pipeline for conditional GANs.
    
    Provides methods for training conditional GANs with separate
    generator and discriminator optimizers.
    """

    def __init__(
        self,
        generator: nn.Module | None = None,
        discriminator: nn.Module | None = None,
        g_optimizer: Optimizer | None = None,
        d_optimizer: Optimizer | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        """Initialize the conditional GAN trainer.
        
        Args:
            generator: Generator network.
            discriminator: Discriminator network.
            g_optimizer: Optimizer for generator.
            d_optimizer: Optimizer for discriminator.
            device: Device to run training on.
            **kwargs: Additional arguments.
        """
        super().__init__(generator=generator, discriminator=discriminator, **kwargs)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.device = device
        
        if self.generator is not None:
            self.generator.to(self.device)
        if self.discriminator is not None:
            self.discriminator.to(self.device)

    def train_step(
        self,
        batch: Any,
        latent_dim: int = 100,
        **kwargs,
    ) -> dict[str, float]:
        """Execute a single training step for both generator and discriminator.
        
        Args:
            batch: A batch of real data and optional conditions.
            latent_dim: Dimension of latent noise.
            **kwargs: Additional arguments.
            
        Returns:
            Dictionary of training metrics.
        """
        # This is a simplified template - actual implementation would
        # include proper GAN training logic with loss functions
        raise NotImplementedError(
            "train_step must be implemented with specific GAN loss functions."
        )
