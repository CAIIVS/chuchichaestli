# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Diffusion model pipeline components."""

from typing import Any

import torch
from torch import nn


__all__ = ["DiffusionMixin", "DiffusionInferencePipeline"]


class DiffusionMixin:
    """Mixin for diffusion model functionality in pipelines.
    
    This mixin provides noise scheduling, denoising, and sampling capabilities
    that can be combined with other pipeline components through multiple inheritance.
    """

    def __init__(
        self,
        diffusion_model: nn.Module | None = None,
        noise_scheduler: Any | None = None,
        **kwargs,
    ) -> None:
        """Initialize the diffusion mixin.
        
        Args:
            diffusion_model: Diffusion denoising model (e.g., UNet).
            noise_scheduler: Noise scheduling object (e.g., DDPM, DDIM).
            **kwargs: Additional arguments passed to parent classes.
        """
        super().__init__(**kwargs)
        self.diffusion_model = diffusion_model
        self.noise_scheduler = noise_scheduler

    def add_noise(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Add noise to input according to timestep.
        
        Args:
            x: Clean input tensor.
            timesteps: Timesteps for noise scheduling.
            noise: Optional pre-generated noise.
            **kwargs: Additional arguments.
            
        Returns:
            Noisy tensor.
        """
        if self.noise_scheduler is None:
            raise ValueError("Noise scheduler must be initialized.")
        
        if noise is None:
            noise = torch.randn_like(x)
        
        return self.noise_scheduler.add_noise(x, noise, timesteps, **kwargs)

    def denoise_step(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Perform a single denoising step.
        
        Args:
            x_t: Noisy input at timestep t.
            timestep: Current timestep.
            **kwargs: Additional arguments for the model.
            
        Returns:
            Denoised tensor.
        """
        if self.diffusion_model is None:
            raise ValueError("Diffusion model must be initialized.")
        
        return self.diffusion_model(x_t, timestep, **kwargs)

    def sample(
        self,
        shape: tuple[int, ...],
        num_steps: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples from noise.
        
        Args:
            shape: Shape of samples to generate.
            num_steps: Number of denoising steps.
            **kwargs: Additional arguments.
            
        Returns:
            Generated samples.
        """
        if self.noise_scheduler is None or self.diffusion_model is None:
            raise ValueError("Both noise scheduler and diffusion model must be initialized.")
        
        # Start from pure noise
        device = next(self.diffusion_model.parameters()).device
        x_t = torch.randn(shape, device=device)
        
        # Determine number of steps
        if num_steps is None:
            num_steps = getattr(self.noise_scheduler, "num_time_steps", 1000)
        
        # Denoise iteratively
        for t in reversed(range(num_steps)):
            timestep = torch.tensor([t], device=device)
            noise_pred = self.denoise_step(x_t, timestep, **kwargs)
            
            # Update x_t using scheduler
            if hasattr(self.noise_scheduler, "step"):
                x_t = self.noise_scheduler.step(x_t, noise_pred, timestep)
            else:
                # Fallback: simple noise subtraction
                x_t = x_t - noise_pred
        
        return x_t


class DiffusionInferencePipeline(DiffusionMixin):
    """Inference pipeline for diffusion models.
    
    Provides methods for sampling and generation using diffusion models.
    """

    def __init__(
        self,
        diffusion_model: nn.Module | None = None,
        noise_scheduler: Any | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        """Initialize the diffusion inference pipeline.
        
        Args:
            diffusion_model: Diffusion denoising model.
            noise_scheduler: Noise scheduling object.
            device: Device to run inference on.
            **kwargs: Additional arguments.
        """
        super().__init__(
            diffusion_model=diffusion_model,
            noise_scheduler=noise_scheduler,
            **kwargs,
        )
        self.device = device
        
        if self.diffusion_model is not None:
            self.diffusion_model.to(self.device)
            self.diffusion_model.eval()

    def predict(
        self,
        batch_size: int = 1,
        shape: tuple[int, ...] | None = None,
        num_steps: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples using the diffusion model.
        
        Args:
            batch_size: Number of samples to generate.
            shape: Shape of each sample (if not provided, inferred from model).
            num_steps: Number of denoising steps.
            **kwargs: Additional arguments.
            
        Returns:
            Generated samples.
        """
        if shape is None:
            # Try to infer shape from model
            raise ValueError("Shape must be provided for generation.")
        
        full_shape = (batch_size,) + shape
        return self.sample(full_shape, num_steps=num_steps, **kwargs)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        shape: tuple[int, ...] | None = None,
        num_steps: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Call the pipeline for inference.
        
        Args:
            batch_size: Number of samples to generate.
            shape: Shape of each sample.
            num_steps: Number of denoising steps.
            **kwargs: Additional arguments.
            
        Returns:
            Generated samples.
        """
        return self.predict(batch_size=batch_size, shape=shape, num_steps=num_steps, **kwargs)
