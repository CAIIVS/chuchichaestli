# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Functional image quality metric implementations."""

from pathlib import Path
import torch
from chuchichaestli.metrics.base import sanitize_ndim
from chuchichaestli.metrics.ssim import SSIM
from chuchichaestli.metrics.lpips import LPIPSLoss
from typing import Literal
from collections.abc import Sequence, Callable


__all__ = ["mse", "psnr", "ssim", "psnr_from_batches"]


def mse(data: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Compute mean-squared error (MSE) between observation and prediction.

    Args:
        data: Observed data.
        prediction: Predicted data.
    """
    data = sanitize_ndim(data, check_2D=True, check_3D=True)
    prediction = sanitize_ndim(prediction, check_2D=True, check_3D=True)
    is_nan = torch.isnan(data) | torch.isnan(prediction)
    return torch.mean(torch.pow(data[~is_nan] - prediction[~is_nan], 2))


def psnr(
    data: torch.Tensor, prediction: torch.Tensor, data_range: float | None = None
) -> torch.Tensor:
    """Compute peak-signal-to-noise ratio between observation and prediction.

    Args:
        data: Observed data.
        prediction: Predicted data.
        data_range: The data range of the images.
    """
    if data_range is None:
        data_range = 1.0
    mean_squared_error = mse(data, prediction)
    if mean_squared_error == 0:
        return torch.inf
    max2 = data_range**2
    return 10 * torch.log10(max2 / mean_squared_error)


def psnr_from_batches(
    psnr_values: Sequence[torch.Tensor],
    pixel_counts: Sequence[int] | None = None,
    data_range: float | None = None,
) -> float:
    """Compute overall PSNR from individual PSNR values (optionally weighted by pixel counts).

    Args:
        psnr_values: Pre-calculated PSNR values on individual batches.
        pixel_counts: Pixel count weights if image sizes vary across batches (optional).
        data_range: The data range of the image domain.
    """
    if pixel_counts is None:
        pixel_counts = [1 for val in psnr_values]
    if data_range is None:
        data_range = 1.0
    max2 = data_range**2
    total_squared_error = 0.0
    total_weight = 0
    for psnr, w in zip(psnr_values, pixel_counts):
        mse = max2 / (10 ** (psnr / 10))
        total_squared_error += w * mse
        total_weight += w
    mean_squared_error = total_squared_error / total_weight
    return 10 * torch.log10(max2 / mean_squared_error)


def ssim(
    data: torch.Tensor,
    prediction: torch.Tensor,
    data_range: float = 1.0,
    kernel_type: Literal["gaussian", "uniform"] = "gaussian",
    kernel_size: int | Sequence[int] = 11,
    kernel_sigma: float | Sequence[float] = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    reduction: Callable | None = torch.mean,
) -> float:
    """Compute peak-signal-to-noise ratio between observation and prediction.

    Args:
        data: Observed data.
        prediction: Predicted data.
        data_range: The data range of the images.
        kernel_size: The size of the kernel for computation.
        kernel_sigma: The standard deviation of the kernel for computation.
        kernel_type: The type of kernel for computation; one of `['gaussian', 'uniform']`.
        data_range: The data range of the images.
        k1: the first stability constant.
        k2: the second stability constant.
        reduction: Reduction function, e.g. `torch.mean` or `torch.sum`.
    """
    if data_range is None:
        data_range = (
            torch.maximum(data.max(), prediction.max()).item()
            - torch.minimum(data.min(), prediction.min()).item()
        )
    if reduction is None:
        reduction = torch.nn.Identity()
    ssim_full, _ = SSIM._compute_ssim_and_cs(
        data,
        prediction,
        kernel_size=kernel_size,
        kernel_sigma=kernel_sigma,
        kernel_type=kernel_type,
        data_range=data_range,
        k1=k1,
        k2=k2,
    )
    batch_size = ssim_full.shape[0]
    ssim_batch = ssim_full.view(batch_size, -1).nanmean(1, keepdim=True)
    return reduction(ssim_batch)


def lpips(
    data: torch.Tensor,
    prediction: torch.Tensor,
    model: Literal[
        "vgg16",
        "vgg",
        "alexnet",
        "squeezenet",
        "resnet18",
        "resnet",
        "convnext",
        "vit",
        "swinv2",
    ] = "vgg16",
    embedding: bool = True,
    embedding_weights: Path | str | None = None,
    reduction: Callable | None = torch.mean,
    spatial_average: bool = True,
) -> torch.Tensor:
    """Compute LPIPS (Learned Perceptual Image Patch Similarity) between two tensors.

    Args:
        data: Observed data.
        prediction: Predicted data.
        model: The type of network to use for LPIPS computation.
        embedding: If `True`, uses linear embedding layers for feature mapping.
        embedding_weights: Path to pretrained weights for the embedding layers.
        reduction: Reduction function, e.g. `torch.mean` or `torch.sum`.
        spatial_average: If `True`, averages across image dimensions.
    """
    lpips_fn = LPIPSLoss(
        model=model,
        embedding=embedding,
        pretrained_weights=embedding_weights,
        spatial_average=spatial_average,
        reduction=reduction,
    )
    return lpips_fn(data, prediction)


if __name__ == "__main__":
    obs = torch.rand((5, 1, 512, 512))
    prd = torch.rand((5, 1, 512, 512))
    obs.requires_grad = True
    prd.requires_grad = True
    val_mse = mse(obs, prd)
    val_psnr = psnr(obs, prd, 1)
    val_ssim = ssim(obs, prd, 1)
    val_lpips = lpips(obs, prd, model="vgg16", embedding=False)
    print(f"MSE (rand): {val_mse}")
    print(f"PSNR (rand): {val_psnr}")
    print(f"SSIM (rand): {val_ssim}")
    print(f"LPIPS (rand): {val_lpips}")

    val2_mse = mse(torch.ones_like(obs), torch.zeros_like(prd))
    val2_psnr = psnr(torch.ones_like(obs), torch.zeros_like(prd), 1)
    val2_ssim = ssim(torch.ones_like(obs), torch.zeros_like(prd), 1)
    val2_lpips = lpips(
        torch.ones_like(obs), torch.zeros_like(prd), model="vgg16", embedding=False
    )
    print(f"MSE (1-0): {val2_mse}")
    print(f"PSNR (1-0): {val2_psnr}")
    print(f"SSIM (1-0): {val2_ssim}")
    print(f"LPIPS (1-0): {val2_lpips}")

    val3_mse = mse(torch.ones_like(obs), torch.ones_like(prd) - 5e-8)
    val3_psnr = psnr(torch.ones_like(obs), torch.ones_like(prd) - 5e-8, 1)
    val3_ssim = ssim(torch.ones_like(obs), torch.ones_like(prd) - 5e-8, 1)
    val3_lpips = lpips(
        torch.ones_like(obs),
        torch.ones_like(prd) - 5e-8,
        model="vgg16",
        embedding=False,
    )
    print(f"MSE (1-1): {val3_mse}")
    print(f"PSNR (1-1): {val3_psnr}")
    print(f"SSIM (1-1): {val3_ssim}")
    print(f"LPIPS (1-1): {val3_lpips}")

    val3d_mse = mse(
        torch.ones((5, 3, 16, 16, 16)), torch.zeros((5, 3, 16, 16, 16)) - 5e-8
    )
    val3d_psnr = psnr(
        torch.ones((5, 3, 16, 16, 16)), torch.zeros((5, 3, 16, 16, 16)) - 5e-8, 1
    )
    val3d_ssim = ssim(
        torch.ones((5, 3, 16, 16, 16)), torch.zeros((5, 3, 16, 16, 16)) - 5e-8, 1
    )
    val3d_lpips = lpips(
        torch.ones((5, 3, 16, 16, 16)),
        torch.zeros((5, 3, 16, 16, 16)) - 5e-8,
        model="vgg16",
        embedding=False,
    )
    print(f"MSE (3D): {val3d_mse}")
    print(f"PSNR (3D): {val3d_psnr}")
    print(f"SSIM (3D): {val3d_ssim}")
    print(f"LPIPS (3D): {val3d_lpips}")

    total_psnr = psnr_from_batches([val_psnr, val3_psnr])
    print(f"PSNR (total:rand,1-1): {total_psnr}")
