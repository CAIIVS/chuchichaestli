# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""FID evaluation metric, including InceptionV3 for feature mapping."""

from functools import partial
import warnings
import torch
from torch.nn import Module
from torch.nn.functional import interpolate
from torchvision import models as tv
from chuchichaestli.metrics.base import (
    EvalMetric,
    sanitize_ndim,
    as_tri_channel,
    as_batched_slices,
)


__all__ = ["FIDInceptionV3", "FID"]


class FIDInceptionV3(Module):
    """InceptionV3 model for calculating FIDs."""

    def __init__(
        self,
        weights: tv.Inception_V3_Weights | None = None,
        use_default_transforms: bool = True,
        mode: str = "bilinear",
        antialias: bool = False,
    ):
        """Constructor.

        Args:
            weights: The pretrained weights for the model; for details and possible values
              see https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.inception_v3.html#torchvision.models.Inception_V3_Weights.
            use_default_transforms: If `True`, uses standard transforms for
              preprocessing (Inception_V3_Weights.IMAGENET1K_V1.transforms).
            mode: If `use_default_transforms=False`, a simple interpolation in
              given mode is performed.
            antialias: If `use_default_transforms=False` and `True`,
              antialiasing is used during interpolation.
        """
        super().__init__()
        self.weights = (
            tv.Inception_V3_Weights.IMAGENET1K_V1 if weights is None else weights
        )
        self.model = tv.inception_v3(weights=self.weights)
        self.model.fc = torch.nn.Identity()
        if use_default_transforms:
            if isinstance(self.weights, str):
                weights = tv.get_model_weights("inception_v3")[self.weights]
                self.transforms = weights.transforms()
            else:
                self.transforms = self.weights.transforms()
        else:
            self.transforms = partial(
                interpolate,
                size=(299, 299),
                mode=mode,
                align_corners=False,
                antialias=antialias,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for the FIDInceptionV3 model."""
        x = sanitize_ndim(x)
        x = as_tri_channel(x)
        x = self.transforms(x)
        x = self.model(x)
        return x


class FID(EvalMetric):
    """Frechet inception distance."""

    def __init__(
        self,
        model: Module | None = None,
        feature_dim: int | None = None,
        device: torch.device | None = None,
        n_images: int = 0,
        **kwargs,
    ):
        """Constructor.

        Args:
            model: Model from which to extract features.
            feature_dim: Feature dimension of the model output; if `None`, the
              feature dimension is determined automatically if possible).
            device: Tensor allocation/computation device.
            n_images: Number of images seen by the internal state.
            kwargs: Additional keyword arguments (passed to parent class).
        """
        super().__init__(device=device, n_images=n_images, **kwargs)
        if model is None:
            model = FIDInceptionV3()
        self.model = model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self._feature_dim = feature_dim
        self.n_images_fake = torch.tensor(n_images // 2, device=self.device)
        self.n_images_real = torch.tensor(n_images - n_images // 2, device=self.device)
        self.aggregate_fake = torch.zeros(self.feature_dim, device=self.device)
        self.aggregate_real = torch.zeros(self.feature_dim, device=self.device)
        self.aggregate_cov_fake = torch.zeros(
            (self.feature_dim, self.feature_dim), device=self.device
        )
        self.aggregate_cov_real = torch.zeros(
            (self.feature_dim, self.feature_dim), device=self.device
        )

    @property
    def feature_dim(self):
        """Feature dimension of the output."""
        if self._feature_dim is None:
            try:
                trial = torch.zeros((2, 3, 299, 299), device=self.device)
                out = self.model(trial)
                shape = out.shape
            except ValueError:
                shape = [2048]
            self._feature_dim = shape[-1]
        return self._feature_dim

    @torch.inference_mode()
    def update(
        self,
        data: torch.Tensor | None = None,
        prediction: torch.Tensor | None = None,
        **kwargs,
    ):
        """Compute metric on new input and update current state.

        Args:
            data: Observed (real) data.
            prediction: Predicted (fake) data.
            kwargs: Additional keyword arguments for parent class.
        """
        sample = kwargs.pop("sample", 0)
        # fake images
        if prediction is not None:
            if prediction.ndim == 5:
                prediction = as_batched_slices(prediction, sample=sample)
            super().update(prediction, prediction, **kwargs)
            features_fake = self.model(prediction)
            self.n_images_fake += prediction.shape[0]
            self.aggregate_fake += torch.sum(features_fake, dim=0)
            self.aggregate_cov_fake += torch.matmul(features_fake.T, features_fake)
        # real images
        if data is not None:
            if data.ndim == 5:
                data = as_batched_slices(data, sample=sample)
            super().update(data, data, **kwargs)
            features_real = self.model(data)
            self.n_images_real += data.shape[0]
            self.aggregate_real += torch.sum(features_real, dim=0)
            self.aggregate_cov_real += torch.matmul(features_real.T, features_real)
        return self

    @staticmethod
    def _calculate_frechet_distance(
        mu1: torch.Tensor,
        sigma1: torch.Tensor,
        mu2: torch.Tensor,
        sigma2: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the Frechet Distance between two multivariate Gaussian distributions."""
        delta_mu_sq = (mu1 - mu2).square().sum(dim=-1)
        trace_sum = sigma1.trace() + sigma2.trace()
        sigma_sq = torch.matmul(sigma1, sigma2)
        eigensum_sqrt = torch.linalg.eigvals(sigma_sq).sqrt().real.sum(dim=-1)
        return delta_mu_sq + trace_sum - 2 * eigensum_sqrt

    @torch.inference_mode()
    def compute(self) -> float:
        """Return current metric state total."""
        if (self.n_images_fake < 1) or (self.n_images_real < 1):
            warnings.warn(
                "Computing FID requires at least 1 real images and 1 fake images."
            )
            return torch.tensor(0.0)
        mean_fake = (self.aggregate_fake / self.n_images_fake).unsqueeze(0)
        mean_real = (self.aggregate_real / self.n_images_real).unsqueeze(0)
        n_cov_fake = self.aggregate_cov_fake - self.n_images_fake * torch.matmul(
            mean_fake.T, mean_fake
        )
        cov_fake = n_cov_fake / max(self.n_images_fake - 1, 1)
        n_cov_real = self.aggregate_cov_real - self.n_images_real * torch.matmul(
            mean_real.T, mean_real
        )
        cov_real = n_cov_real / max(self.n_images_real - 1, 1)
        return self._calculate_frechet_distance(
            mean_real.squeeze(), cov_real, mean_fake.squeeze(), cov_fake
        )

    def reset(self, **kwargs):
        """Reset the current metrics state."""
        self.__init__(
            min_value=self.min_value.item(),
            max_value=self.max_value.item(),
            device=self.device,
            model=self.model,
            feature_dim=self._feature_dim,
        )

    def to(self, device: torch.device = None):
        """Perform tensor device conversion for all internal tensors.

        Args:
            device: Tensor allocation/computation device.
        """
        super().to(device)
        self.model = self.model.to(device)
        self.n_images_fake = self.n_images_fake.to(device=device)
        self.n_images_real = self.n_images_real.to(device=device)
        self.aggregate_fake = self.aggregate_fake.to(device=device)
        self.aggregate_real = self.aggregate_real.to(device=device)


if __name__ == "__main__":
    weights = tv.Inception_V3_Weights
    fid_model = FIDInceptionV3(use_default_transforms=False)
    # print(fid_model)
    metric = FID()
    obs = torch.rand((5, 1, 512, 512))
    prd = torch.rand((5, 1, 512, 512))

    metric.update(obs, prd)
    metric.update(obs, prd)
    print(f"FID (randx2): {metric.compute()}")

    metric.update(obs, prd)
    metric.update(obs, prd)
    val = metric.compute()
    metric.reset()
    print(f"FID (randx4): {val}")

    metric.update(obs, prd * 2)
    metric.update(obs, prd * 2)
    print(f"FID (>=max_valx2): {metric.compute()}")
    metric.reset()

    obs3d = torch.rand((5, 1, 8, 8, 6))
    prd3d = torch.rand((5, 1, 8, 8, 6))
    metric.update(obs3d, prd3d)
    metric.update(obs3d, prd3d)
    metric.update(obs3d, prd3d)
    val = metric.compute()
    print(f"FID (3D): {val}")
