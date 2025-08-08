# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Structural Similarity Index Measure evaluation metric and loss."""

import torch
from torch.nn import Module
from chuchichaestli.models.maps import DIM_TO_CONV_FN_MAP
from chuchichaestli.metrics.base import EvalMetric, sanitize_ndim
from typing import Literal
from collections.abc import Sequence, Callable


class SSIM(EvalMetric):
    """Structural similarity index measure."""

    def __init__(
        self,
        min_value: float = 0,
        max_value: float = 1,
        n_observations: int = 0,
        n_images: int = 0,
        device: torch.device = None,
        kernel_size: int | Sequence[int] = 11,
        kernel_sigma: float | Sequence[float] = 1.5,
        kernel_type: Literal["gaussian", "uniform"] = "gaussian",
        k1: float = 0.01,
        k2: float = 0.03,
        **kwargs,
    ):
        """Constructor.

        Args:
            min_value: Minimum data value relative to metric computation.
            max_value: Maximum data value relative to metric computation.
            n_observations: Number of observations (pixels) seen by the internal state.
            n_images: Number of images seen by the internal state.
            device: Tensor allocation/computation device.
            kernel_size: Size of the moving window aka kernel.
            kernel_sigma: Standard deviation for the Gaussian kernel.
            kernel_type: Type of kernel; one of `['gaussian', 'uniform']`.
            k1: Algorithm parameter, K1 (small constant).
            k2: Algorithm parameter, K1 (small constant).
            kwargs: Base class keyword arguments.
        """
        super().__init__(
            min_value=min_value,
            max_value=max_value,
            n_observations=n_observations,
            n_images=n_images,
            device=device,
        )
        self.kernel_type = kernel_type
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.k1 = k1
        self.k2 = k2

    @torch.inference_mode()
    def update(
        self, data: torch.Tensor, prediction: torch.Tensor, update_range: bool = True
    ):
        """Compute metric, aggregate, and update internal state.

        Args:
            data: Observed data.
            prediction: Predicted data.
            update_range: If True, ranges are automatically updated based on
              new observation.
        """
        super().update(data, prediction, update_range=update_range)
        ssim_full, _ = self._compute_ssim_and_cs(
            data,
            prediction,
            self.kernel_size,
            self.kernel_sigma,
            self.kernel_type,
            self.data_range,
            self.k1,
            self.k2,
        )
        batch_size = ssim_full.shape[0]
        aggregate_batch = ssim_full.view(batch_size, -1).nanmean(1, keepdim=True)
        self.aggregate = self.aggregate + torch.sum(aggregate_batch)
        return self

    @torch.inference_mode()
    def compute(self) -> float:
        """Return current metric state total."""
        if self.n_images == 0:
            return None
        ssim = self.aggregate / self.n_images
        self.value = ssim
        return ssim.item()

    def reset(self, **kwargs) -> float:
        """Reset the current metrics state."""
        self.__init__(
            min_value=self.min_value.item(),
            max_value=self.max_value.item(),
            device=self.device,
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            kernel_type=self.kernel_type,
            k1=self.k1,
            k2=self.k2,
            **kwargs,
        )

    @staticmethod
    def _compute_ssim_and_cs(
        data: torch.Tensor,
        prediction: torch.Tensor,
        kernel_size: int | Sequence[int],
        kernel_sigma: float | Sequence[float],
        kernel_type: Literal["gaussian", "uniform"] = "gaussian",
        data_range: float = 1.0,
        k1: float = 0.01,
        k2: float = 0.03,
        dtype: torch.dtype = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the SSIM and Contrast Sensitivity (CS) for a batch of images.

        See Wang, Zhou, et al. "Image quality assessment: from error visibility
        to structural similarity." IEEE transactions on image processing 13.4
        (2004): 600-612.

        Args:
            data: Target image batch of shape `(batch_size, channels, height, width)`.
            prediction: Predicted image batch of same shape as `data`.
            kernel_size: The size of the kernel for computation.
            kernel_sigma: The standard deviation of the kernel for computation.
            kernel_type: The type of kernel for computation; one of `['gaussian', 'uniform']`.
            data_range: The data range of the images.
            k1: the first stability constant.
            k2: the second stability constant.
            dtype: Data type of the images.

        Returns:
            ssim (torch.Tensor): the SSIM score for the batch of images.
            cs (torch.Tensor): the CS for the batch of images.
        """
        if data.shape != prediction.shape:
            raise ValueError(
                f"`prediction` and `data` should have same shapes, "
                f"got {prediction.shape} and {data.shape}."
            )
        # sanitize data and prediction
        data = sanitize_ndim(data, check_2D=True, check_3D=True)
        prediction = sanitize_ndim(prediction, check_2D=True, check_3D=True)
        if dtype is None:
            dtype = data.dtype
        data = data.to(dtype)
        prediction = prediction.to(dtype)

        # get kernel
        num_channels = data.size(1)
        spatial_dim = data.ndim - 2
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial_dim
        elif len(kernel_size) < spatial_dim:
            kernel_size = [11] * spatial_dim
        if isinstance(kernel_sigma, float):
            kernel_sigma = [kernel_sigma] * spatial_dim
        elif len(kernel_sigma) < spatial_dim:
            kernel_sigma = [1.5] * spatial_dim
        if kernel_type == "gaussian":
            kernel = SSIM._gaussian_kernel(num_channels, kernel_size, kernel_sigma)
        elif kernel_type == "uniform":
            kernel = torch.ones((num_channels, 1, *kernel_size)) / torch.prod(
                torch.tensor(kernel_size)
            )
        else:
            raise ValueError(f"Unknown kernel type {kernel_type}")
        kernel = kernel.to(dtype=dtype, device=prediction.device)
        # compute SSIM and CS
        conv_fn = DIM_TO_CONV_FN_MAP[spatial_dim]
        c1 = (k1 * data_range) ** 2  # stability constant for luminance
        c2 = (k2 * data_range) ** 2  # stability constant for contrast
        mu_x = conv_fn(prediction, kernel, groups=num_channels)
        mu_y = conv_fn(data, kernel, groups=num_channels)
        mu_xx = conv_fn(prediction * prediction, kernel, groups=num_channels)
        mu_yy = conv_fn(data * data, kernel, groups=num_channels)
        mu_xy = conv_fn(prediction * data, kernel, groups=num_channels)
        sigma_x = mu_xx - mu_x * mu_x
        sigma_y = mu_yy - mu_y * mu_y
        sigma_xy = mu_xy - mu_x * mu_y
        cs = (2 * sigma_xy + c2) / (sigma_x + sigma_y + c2)
        ssim = ((2 * mu_x * mu_y + c1) / (mu_x**2 + mu_y**2 + c1)) * cs
        return ssim, cs

    @staticmethod
    def _gaussian_kernel(
        num_channels: int, kernel_size: list[int], kernel_sigma: list[float]
    ) -> torch.Tensor:
        """Computes 2D gaussian kernel.

        Args:
            num_channels: Number of channels in the image
            kernel_size: Size of kernel
            kernel_sigma: Standard deviation for Gaussian kernel.
        """
        spatial_dim = len(kernel_size)
        if spatial_dim > 3:
            raise ValueError(
                f"SSIM only supports 2D or 3D kernels, got {spatial_dim}D kernel."
            )

        def gaussian_1d(kernel_size: int, sigma: float) -> torch.Tensor:
            """Computes 1D gaussian kernel.

            Args:
                kernel_size: Size of the gaussian kernel
                sigma: Standard deviation of the gaussian kernel
            """
            dist = torch.arange(
                start=(1 - kernel_size) / 2, end=(1 + kernel_size) / 2, step=1
            )
            gauss = torch.exp(-torch.pow(dist / sigma, 2) / 2)
            return (gauss / gauss.sum()).unsqueeze(dim=0)

        gaussian_kernel_x = gaussian_1d(kernel_size[0], kernel_sigma[0])
        gaussian_kernel_y = gaussian_1d(kernel_size[1], kernel_sigma[1])
        kernel = torch.matmul(gaussian_kernel_x.t(), gaussian_kernel_y)
        kernel_dimensions: tuple[int, ...] = (
            num_channels,
            1,
            kernel_size[0],
            kernel_size[1],
        )
        if spatial_dim == 3:
            gaussian_kernel_z = gaussian_1d(kernel_size[2], kernel_sigma[2])[None,]
            kernel = torch.mul(
                kernel.unsqueeze(-1).repeat(1, 1, kernel_size[2]),
                gaussian_kernel_z.expand(*kernel_size),
            )
            kernel_dimensions += (kernel_size[2],)
        return kernel.expand(kernel_dimensions)


class SSIMLoss(Module):
    """Structural similarity index measure loss function."""

    def __init__(
        self,
        data_range: float = 1.0,
        kernel_type: Literal["gaussian", "uniform"] = "gaussian",
        kernel_size: int | Sequence[int] = 11,
        kernel_sigma: float | Sequence[float] = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        reduction: Callable | None = torch.mean,
    ):
        """Constructor."""
        super().__init__()
        self.data_range = data_range
        self.kernel_type = kernel_type
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.k1 = k1
        self.k2 = k2
        self.reduction = reduction if reduction is not None else torch.nn.Identity()

    def forward(
        self, data: torch.Tensor, prediction: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute the SSIM loss.

        Args:
            data: Observed data.
            prediction: Predicted data.
            kwargs: Additional keyword arguments, e.g., `reduction`.

        Returns:
            Loss value.
        """
        reduction = kwargs.get("reduction", self.reduction)
        if reduction is None:
            reduction = torch.nn.Identity()
        ssim_full, _ = SSIM._compute_ssim_and_cs(
            data,
            prediction,
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            kernel_type=self.kernel_type,
            data_range=self.data_range,
            k1=self.k1,
            k2=self.k2,
        )
        batch_size = ssim_full.shape[0]
        ssim_batch = ssim_full.view(batch_size, -1).nanmean(1, keepdim=True)
        loss = 1 - ssim_batch
        return reduction(loss)


if __name__ == "__main__":
    metric = SSIM(0, 1)
    obs = torch.rand((5, 1, 64, 64))
    prd = torch.rand((5, 1, 64, 64))

    metric.update(obs, prd)
    val = metric.compute()
    metric.reset()
    print(f"SSIM (rand): {val}")

    metric.update(torch.ones_like(obs), torch.zeros_like(prd))
    val2 = metric.compute()
    metric.reset()
    print(f"SSIM (1-0): {val2}")

    metric.update(torch.ones_like(obs), torch.ones_like(prd))
    metric.update(torch.ones_like(obs), torch.ones_like(prd))
    metric.update(torch.ones_like(obs), torch.ones_like(prd))
    val3 = metric.compute()
    metric.reset()
    print(f"SSIM (1-1 x3): {val3}")

    metric.update(obs, prd * 2)
    val_edge = metric.compute()
    metric.reset()
    print(f"SSIM (>=max_val): {val_edge}")

    loss_fn = SSIMLoss()
    obs.requires_grad = True
    prd.requires_grad = True
    print(loss_fn(obs, prd))
