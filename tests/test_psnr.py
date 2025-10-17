# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the PSNR module."""

import pytest
import torch
from chuchichaestli.metrics.psnr import PSNR


@pytest.fixture(autouse=True)
def patch_tensor_to(monkeypatch):
    """Patch torch.Tensor.to to be a no-op for CPU compatibility."""
    monkeypatch.setattr(torch.Tensor, "to", lambda self, device=None, **k: self)
    yield


def test_PSNR_update_and_compute_basic():
    """Test basic update and compute functionality of PSNR metric."""
    metric = PSNR(min_value=0, max_value=1)
    data = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    pred = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    metric.update(data, pred)
    # MSE = ((0+1+0+1)/4) = 0.5, data_range=1
    expected_psnr = 10 * torch.log10(torch.tensor(1.0**2) / 0.5)
    val = metric.compute()
    assert pytest.approx(val) == expected_psnr.item()


def test_PSNR_update_and_compute_perfect():
    """Test PSNR metric with perfect predictions."""
    metric = PSNR(min_value=0, max_value=1)
    data = torch.ones(1, 1, 2, 2)
    pred = torch.ones(1, 1, 2, 2)
    metric.update(data, pred)
    # MSE = 0, PSNR should be inf
    val = metric.compute()
    assert val == float("inf") or torch.isinf(torch.tensor(val))


def test_PSNR_update_and_compute_extreme_values():
    """Test PSNR metric with extreme values."""
    metric = PSNR(min_value=0, max_value=1)
    data = torch.ones(1, 1, 2, 2)
    pred = torch.zeros(1, 1, 2, 2)
    metric.update(data, pred)
    # MSE = 1, PSNR = 10*log10(1/1) = 0
    val = metric.compute()
    assert pytest.approx(val) == 0.0


def test_PSNR_update_and_compute_small_difference():
    """Test PSNR metric with small differences."""
    metric = PSNR(min_value=0, max_value=1)
    data = torch.ones(1, 1, 2, 2)
    pred = torch.ones(1, 1, 2, 2) - 5e-8
    metric.update(data, pred)
    val = metric.compute()
    # Should be very high PSNR for tiny error
    assert val > 120  # Should be very large


def test_PSNR_update_and_compute_none_if_no_data():
    """Test PSNR metric returns None if no data has been updated."""
    metric = PSNR()
    assert metric.compute() is None


def test_PSNR_update_with_nan():
    """Test PSNR metric handling NaN values."""
    metric = PSNR()
    data = torch.tensor([[[[1.0, float("nan")]]]])
    pred = torch.tensor([[[[1.0, 0.0]]]])
    metric.update(data, pred)
    # Only one valid pixel: (1-1)^2=0, so MSE=0, PSNR=inf
    val = metric.compute()
    assert val == float("inf") or torch.isinf(torch.tensor(val))


def test_PSNR_update_device_switch(monkeypatch):
    """Test PSNR metric updates with device mismatch."""
    metric = PSNR(device=torch.device("cpu"))
    # Simulate device mismatch
    called = {}

    def fake_to(self, device=None, **kwargs):
        called["to"] = True
        return self

    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    pred = torch.ones(1, 1, 2, 2)
    data = torch.ones(1, 1, 2, 2)
    metric.device = torch.device("meta")  # fake device
    metric.update(data, pred)
    assert called.get("to", False)


def test_PSNR_update_5d_tensor_random():
    """Test PSNR metric with 5D tensors."""
    metric = PSNR(min_value=0, max_value=1)
    obs = torch.rand((2, 1, 4, 4, 3))
    prd = torch.rand((2, 1, 4, 4, 3))
    metric.update(obs, prd)
    val = metric.compute()
    assert isinstance(val, float)


def test_PSNR_update_5d_tensor_perfect():
    """Test PSNR metric with equal 5D tensors."""
    metric = PSNR(min_value=0, max_value=1)
    obs = torch.ones((2, 1, 4, 4, 3))
    prd = torch.ones((2, 1, 4, 4, 3))
    metric.update(obs, prd)
    val = metric.compute()
    assert val == float("inf") or torch.isinf(torch.tensor(val))


def test_PSNR_update_5d_tensor_extreme():
    """Test PSNR metric with 5D tensors at data range extreme."""
    metric = PSNR(min_value=0, max_value=1)
    obs = torch.ones((2, 1, 4, 4, 3))
    prd = torch.zeros((2, 1, 4, 4, 3))
    metric.update(obs, prd)
    val = metric.compute()
    assert pytest.approx(val) == 0.0


def test_PSNR_update_5d_tensor_with_nan():
    """Test PSNR metric with NaN values in 5D tensors."""
    metric = PSNR(min_value=0, max_value=1)
    obs = torch.ones((1, 1, 2, 2, 2))
    prd = torch.ones((1, 1, 2, 2, 2))
    obs[0, 0, 0, 0, 0] = float("nan")
    metric.update(obs, prd)
    val = metric.compute()
    assert val == float("inf") or torch.isinf(torch.tensor(val))
