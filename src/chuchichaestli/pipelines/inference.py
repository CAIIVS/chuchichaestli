# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Base inference pipeline class."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


__all__ = ["InferencePipeline"]


class InferencePipeline(ABC):
    """Base class for inference pipelines.
    
    This class provides a foundation for building modular inference pipelines
    that can be extended through mixins and multiple inheritance.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        """Initialize the inference pipeline.
        
        Args:
            model: The neural network model for inference.
            device: Device to run inference on ('cpu' or 'cuda').
            **kwargs: Additional arguments for subclasses.
        """
        self.model = model
        self.device = device
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """Generate predictions from the model.
        
        This method should be implemented by subclasses to define
        the specific prediction logic for the pipeline.
        
        Returns:
            Model predictions.
        """
        pass

    def load_model(self, path: str, **kwargs) -> None:
        """Load model weights from a checkpoint.
        
        Args:
            path: Path to the checkpoint file.
            **kwargs: Additional arguments for loading.
        """
        if self.model is None:
            raise ValueError("Model must be initialized before loading weights.")
        
        checkpoint = torch.load(path, map_location=self.device, **kwargs)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    def to(self, device: str) -> "InferencePipeline":
        """Move pipeline to specified device.
        
        Args:
            device: Target device ('cpu' or 'cuda').
            
        Returns:
            Self for method chaining.
        """
        self.device = device
        if self.model is not None:
            self.model.to(device)
        return self

    @torch.no_grad()
    def __call__(self, *args, **kwargs) -> Any:
        """Call the pipeline for inference.
        
        This is a convenience method that wraps predict() with
        torch.no_grad() context.
        
        Returns:
            Model predictions.
        """
        return self.predict(*args, **kwargs)
