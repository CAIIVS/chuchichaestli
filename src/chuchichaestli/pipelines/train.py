# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Base training pipeline class."""

from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


__all__ = ["Trainer"]


class Trainer(ABC):
    """Base class for training pipelines.
    
    This class provides a foundation for building modular training pipelines
    that can be extended through mixins and multiple inheritance.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        """Initialize the training pipeline.
        
        Args:
            model: The neural network model to train.
            optimizer: The optimizer for training.
            device: Device to run training on ('cpu' or 'cuda').
            **kwargs: Additional arguments for subclasses.
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.metrics = {}
        self.epoch = 0
        
        if self.model is not None:
            self.model.to(self.device)

    @abstractmethod
    def train_step(self, batch: Any, *args, **kwargs) -> dict[str, float]:
        """Execute a single training step.
        
        This method should be implemented by subclasses to define
        the specific training logic for one batch.
        
        Args:
            batch: A batch of training data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Dictionary of metrics from the training step.
        """
        pass

    def train_epoch(
        self, dataloader: DataLoader, *args, **kwargs
    ) -> dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Dictionary of average metrics for the epoch.
        """
        if self.model is None:
            raise ValueError("Model must be initialized before training.")
        
        self.model.train()
        epoch_metrics = {}
        
        for batch in dataloader:
            metrics = self.train_step(batch, *args, **kwargs)
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)
        
        # Average metrics
        avg_metrics = {
            key: sum(values) / len(values) 
            for key, values in epoch_metrics.items()
        }
        
        self.epoch += 1
        return avg_metrics

    def save_checkpoint(
        self, 
        path: str | Path, 
        include_optimizer: bool = True,
        **kwargs,
    ) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint.
            include_optimizer: Whether to include optimizer state.
            **kwargs: Additional items to save in checkpoint.
        """
        if self.model is None:
            raise ValueError("Model must be initialized before saving.")
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "epoch": self.epoch,
            "metrics": self.metrics,
            **kwargs,
        }
        
        if include_optimizer and self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        path: str | Path,
        load_optimizer: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            path: Path to the checkpoint file.
            load_optimizer: Whether to load optimizer state.
            **kwargs: Additional arguments for torch.load.
            
        Returns:
            The loaded checkpoint dictionary.
        """
        if self.model is None:
            raise ValueError("Model must be initialized before loading.")
        
        checkpoint = torch.load(path, map_location=self.device, **kwargs)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.metrics = checkpoint.get("metrics", {})
        
        if load_optimizer and self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        return checkpoint

    def to(self, device: str) -> "Trainer":
        """Move trainer to specified device.
        
        Args:
            device: Target device ('cpu' or 'cuda').
            
        Returns:
            Self for method chaining.
        """
        self.device = device
        if self.model is not None:
            self.model.to(device)
        return self
