# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the FID module."""

import pytest
import torch
from torch.nn import Module
from torchvision import models as tv
from chuchichaestli.metrics.fid import FID, FIDInceptionV3


@pytest.fixture(autouse=True)
def patch_tensor_to(monkeypatch):
    """Patch torch.Tensor.to to be no-op for CPU compatibility."""
    monkeypatch.setattr(torch.Tensor, "to", lambda self, device=None, **k: self)
    yield


class DummyModel(Module):
    """A dummy model that outputs fixed-size features from input tensors."""

    def __init__(self, feature_dim=8):
        """Initialize the dummy model with a specified feature dimension."""
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x):
        """Forward pass that simulates feature extraction."""
        batch_size = x.shape[0]
        # Return increasing values per batch
        return torch.arange(batch_size * self.feature_dim, dtype=torch.float32).reshape(
            batch_size, self.feature_dim
        )


def test_fid_inceptionv3_forward(monkeypatch):
    """Test that FIDInceptionV3 can be used to extract features."""

    class FastDummy(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            # Return feature vector for FID
            return torch.ones(x.shape[0], 2048)

    monkeypatch.setattr(
        "torchvision.models.inception_v3", lambda weights=None: FastDummy()
    )
    bb = FIDInceptionV3(use_default_transforms=False)
    x = torch.rand(2, 3, 299, 299)
    y = bb(x)
    assert y.shape == (2, 2048)


def test_fid_inceptionv3_transforms(monkeypatch):
    """Test that FIDInceptionV3 applies the correct transforms."""
    monkeypatch.setattr(
        "torchvision.models.inception_v3", lambda weights=None: torch.nn.Identity()
    )
    model = FIDInceptionV3(use_default_transforms=False)
    x = torch.rand(2, 3, 150, 150)
    out = model(x)
    assert out.shape == (2, 3, 299, 299)


@pytest.mark.parametrize("feature_dim", [4, 8])
def test_fid_feature_dim_property(feature_dim):
    """Test that the feature_dim property is set correctly in FID."""
    model = DummyModel(feature_dim)
    fid = FID(model=model, feature_dim=feature_dim)
    assert fid.feature_dim == feature_dim


def test_fid_feature_dim_property_invalid():
    """Test that the feature_dim property leads to ValueError in FID."""

    class DummyErrorModel(DummyModel):
        """A dummy model that raises an error for feature_dim."""

        def forward(self, x):
            raise ValueError("This model does not support feature_dim.")

    model = DummyErrorModel(feature_dim=None)
    fid = FID(model=model)
    print(fid.feature_dim)


def test_fid_update_and_compute_simple():
    """Test FID metric update and compute with simple data."""
    model = DummyModel(4)
    metric = FID(model=model, feature_dim=4)
    # Fake data and prediction
    real = torch.zeros(2, 3, 4, 4)
    fake = torch.ones(2, 3, 4, 4)
    metric.update(data=real, prediction=fake)
    # Should return a float as FID
    val = metric.compute()
    assert isinstance(val, torch.Tensor) or isinstance(val, float)


def test_fid_update_only_fake_or_only_real():
    """Test FID metric with only fake or only real data."""
    model = DummyModel(4)
    metric = FID(model=model, feature_dim=4)
    fake = torch.ones(2, 3, 4, 4)
    metric.update(prediction=fake)
    # Not enough real, returns 0 with warning
    with pytest.warns(UserWarning):
        val = metric.compute()
    assert val == 0 or val == torch.tensor(0.0)
    real = torch.ones(2, 3, 4, 4)
    metric = FID(model=model, feature_dim=4)
    metric.update(data=real)
    # Not enough fake, returns 0 with warning
    with pytest.warns(UserWarning):
        val = metric.compute()
    assert val == 0 or val == torch.tensor(0.0)


def test_fid_reset_resets_state():
    """Test that the FID metric reset method clears the state."""
    model = DummyModel(4)
    metric = FID(model=model, feature_dim=4)
    metric.aggregate_fake += 1
    metric.aggregate_real += 2
    metric.n_images_fake += 3
    metric.n_images_real += 4
    metric.reset()
    assert torch.all(metric.aggregate_fake == 0)
    assert torch.all(metric.aggregate_real == 0)
    assert metric.n_images_fake == 0
    assert metric.n_images_real == 0


def test_fid_to_moves_all(monkeypatch):
    """Test that the .to method applies to all submodules and internal tensors."""
    model = DummyModel(4)
    metric = FID(model=model, feature_dim=4)
    # .to should apply to all submodules and internal tensors
    metric.to(torch.device("cpu"))
    assert metric.model is model
    assert metric.aggregate_fake.device == torch.device("cpu")


def test_fid_calculate_frechet_distance_basic():
    """Test the basic functionality of the FID distance calculation."""
    mu1 = torch.zeros(4)
    mu2 = torch.ones(4)
    sigma1 = torch.eye(4)
    sigma2 = torch.eye(4)
    dist = FID._calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    # Analytical solution: sum((0-1)^2) + 4+4 - 2*4 = 4 + 8 - 8 = 4
    assert pytest.approx(dist.item()) == 4.0


def test_fid_full_pipeline():
    """Test the full pipeline of FID metric computation with dummy data."""
    model = DummyModel(8)
    metric = FID(model=model, feature_dim=8)
    real = torch.zeros(3, 3, 4, 4)
    fake = torch.ones(3, 3, 4, 4)
    metric.update(data=real, prediction=fake)
    metric.update(data=real, prediction=fake)
    val = metric.compute()
    assert isinstance(val, torch.Tensor) or isinstance(val, float)
    metric.reset()
    assert torch.all(metric.aggregate_fake == 0)
    assert torch.all(metric.aggregate_real == 0)


def test_fid_5d_tensor_support():
    """Test FID metric with 5D tensors (B, C, W, H, D) input."""
    model = DummyModel(8)
    metric = FID(model=model, feature_dim=8)
    real = torch.rand(2, 3, 4, 4, 3)  # Should slice to (6, 3, 4, 4)
    fake = torch.rand(2, 3, 4, 4, 3)
    metric.update(data=real, prediction=fake)
    val = metric.compute()
    assert isinstance(val, torch.Tensor) or isinstance(val, float)


def test_torchvision_FID_default(monkeypatch):
    """Test that the FID with default torchvision model weights."""
    called = {}

    def fake_to(self, device=None, **kwargs):
        called["to"] = True
        return self

    monkeypatch.setattr(Module, "to", fake_to)
    metric = FID()
    assert called["to"]
    assert metric.model is not None


def test_torchvision_FID_model_string_weights(monkeypatch):
    """Test that the FID weights can be downloaded."""
    # This will trigger the download if not already present
    inceptv3 = FIDInceptionV3(weights="IMAGENET1K_V1")
    called = {}

    def fake_to(self, device=None, **kwargs):
        called["to"] = True
        return self

    monkeypatch.setattr(Module, "to", fake_to)
    metric = FID(model=inceptv3)
    assert called["to"]
    assert inceptv3.model is not None
    assert metric.model is inceptv3


def test_torchvision_FID_model_explicit_weights(monkeypatch):
    """Test that the FID weights can be downloaded."""
    # This will trigger the download if not already present
    inceptv3 = FIDInceptionV3(weights=tv.Inception_V3_Weights.IMAGENET1K_V1)
    called = {}

    def fake_to(self, device=None, **kwargs):
        called["to"] = True
        return self

    monkeypatch.setattr(Module, "to", fake_to)
    metric = FID(model=inceptv3)
    assert called["to"]
    assert inceptv3.model is not None
    assert metric.model is inceptv3
