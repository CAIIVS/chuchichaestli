# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Base evaluation metric class and utility functions implementation."""

import torch


__all__ = ["EvalMetric", "sanitize_ndim", "as_tri_channel", "as_batched_slices"]


def sanitize_ndim(x: torch.Tensor, check_2D: bool = True, check_3D: bool = False):
    """Standardize image dimensionality to (B, C, W, H)."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    if check_2D and check_3D and (x.ndim != 4 and x.ndim != 5):
        raise ValueError(
            f"Require input of shape {'(C, W, H) or (B, C, W, H)' if check_2D else ''}"
            f"{' or (B, C, W, H, D)' if check_3D else ''}."
        )
    elif check_3D and not check_2D and x.ndim != 5:
        raise ValueError("Require input of shape (B, C, W, H, D).")
    elif check_2D and not check_3D and x.ndim != 4:
        raise ValueError("Require input of shape (C, W, H) or (B, C, W, H).")
    return x


def as_tri_channel(x: torch.Tensor):
    """Morph input to resemble a three-channel image."""
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] < 3:
        x = x[:, 0:1, :, :].repeat(1, 3, 1, 1)
    if x.shape[1] > 3:
        raise ValueError(f"Input has more than three channels ({x.shape[1]})!")
    return x


def as_batched_slices(x: torch.Tensor, sample: int = 0) -> torch.Tensor:
    """Convert batches of volumetric 5D tensors into 4D slice-wise image tensors.

    Args:
        x: Volumetric 5D input tensor.
        sample: If `> 0`, the volume depth is sampled `sample` times from the centre.
    """
    if x.ndim == 5:
        B, C, W, H, D = x.shape
        if sample > 0:
            sample = min(sample, D)
            center = D // 2
            window = sample // 2
            start = center - window
            end = start + sample
            if sample % 2 == 0:
                start = center - window
                end = center + window
            x = x[..., start:end]
            D = sample
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(B * D, C, W, H)
    return x


class EvalMetric:
    """Base class for image evaluation metrics."""

    def __init__(
        self,
        min_value: float = 0,
        max_value: float = 1,
        n_observations: int = 0,
        n_images: int = 0,
        device: torch.device | None = None,
        **kwargs,
    ):
        """Constructor.

        Args:
            min_value: Minimum data value relative to metric computation.
            max_value: Maximum data value relative to metric computation.
            n_observations: Number of observations (pixels) seen by the internal state.
            n_images: Number of images seen by the internal state.
            device: Tensor allocation/computation device.
            kwargs: Additional keyword arguments.
        """
        self.device = torch.get_default_device() if device is None else device
        self.is_nan = torch.tensor(False, device=self.device)
        self.nan_count = torch.tensor(0, device=self.device)
        self.min_value = torch.tensor(min_value, device=self.device)
        self.max_value = torch.tensor(max_value, device=self.device)
        self.n_observations = torch.tensor(n_observations, device=self.device)
        self.n_images = torch.tensor(n_images, device=self.device)
        self.value = torch.tensor(0, device=self.device)
        self.aggregate = torch.tensor(0, device=self.device)

    def to(self, device: torch.device = None):
        """Perform tensor device conversion for all internal tensors.

        Args:
            device: Tensor allocation/computation device.
        """
        self.device = device
        self.is_nan = self.is_nan.to(device=self.device)
        self.nan_count = self.nan_count.to(device=self.device)
        self.min_value = self.min_value.to(device=self.device)
        self.max_value = self.max_value.to(device=self.device)
        self.n_observations = self.n_observations.to(device=self.device)
        self.n_images = self.n_images.to(device=self.device)
        self.value = self.value.to(device=self.device)
        self.aggregate = self.aggregate.to(device=self.device)

    @torch.inference_mode()
    def update(
        self, data: torch.Tensor, prediction: torch.Tensor, update_range: bool = True
    ):
        """Compute metric, aggregate, and update internal state.

        Args:
            data: Observed data aka target.
            prediction: Predicted data aka inferred target.
            update_range: If True, ranges are automatically updated based on
              new observation.
        """
        if self.device != prediction.device:
            self.to(prediction.device)
        self.is_nan = torch.isnan(data) | torch.isnan(prediction)
        self.nan_count += torch.sum(self.is_nan)
        self.n_images += torch.tensor(prediction.shape[0], device=self.device)
        self.n_observations += torch.tensor(
            prediction[~self.is_nan].numel(), device=self.device
        )
        if update_range:
            self.min_value = torch.minimum(
                prediction[~self.is_nan].min(), self.min_value
            )
            self.min_value = torch.minimum(data[~self.is_nan].min(), self.min_value)
            self.max_value = torch.maximum(
                prediction[~self.is_nan].max(), self.max_value
            )
            self.max_value = torch.maximum(data[~self.is_nan].max(), self.max_value)
        return self

    @property
    def data_range(self) -> torch.Tensor:
        """Data range of the metric."""
        return self.max_value - self.min_value

    @data_range.setter
    def data_range(self, data_range: float):
        """Setter for the data range."""
        self.max_value = self.min_value + data_range

    def reset(self, **kwargs):
        """Reset internal state of the metric (keeps range values)."""
        self.__init__(
            min_value=self.min_value.item(),
            max_value=self.max_value.item(),
            device=self.device,
            **kwargs,
        )
