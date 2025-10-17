# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the MSE module."""

import pytest
import torch
from chuchichaestli.metrics.mse import MSE


@pytest.fixture(autouse=True)
def patch_tensor_to(monkeypatch):
    """Patch torch.Tensor.to to be a no-op for CPU compatibility."""
    monkeypatch.setattr(torch.Tensor, "to", lambda self, device=None, **k: self)
    yield


def test_MSE_update_and_compute_basic():
    """Test basic update and compute functionality of MSE metric."""
    metric = MSE()
    data = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    pred = torch.tensor([[[[1.0, 1.0], [2.0, 0.0]]]])
    metric.update(data, pred)
    # MSE = mean((0, 1, 1, 16)) = (0+1+1+16)/4 = 4.5
    expected = (0 + 1 + 1 + 16) / 4
    val = metric.compute()
    assert pytest.approx(val) == expected


def test_MSE_update_and_compute_multiple_batches():
    """Test MSE metric with multiple batches."""
    metric = MSE()
    data1 = torch.ones(2, 1, 4, 4)
    pred1 = torch.zeros(2, 1, 4, 4)
    metric.update(data1, pred1)
    # After first batch, aggregate=32, n_obs=32
    assert metric.aggregate == 32
    assert metric.n_observations == 32
    # Add a second batch
    data2 = 2 * torch.ones(2, 1, 4, 4)
    pred2 = torch.ones(2, 1, 4, 4)
    metric.update(data2, pred2)
    # Now aggregate should be 32 + 32 = 64, n_obs=64
    assert metric.aggregate == 64
    assert metric.n_observations == 64
    assert pytest.approx(metric.compute()) == 1.0


def test_MSE_update_with_nan():
    """Test MSE metric handling NaN values."""
    metric = MSE()
    data = torch.tensor([[[[1.0, float("nan")]]]])
    pred = torch.tensor([[[[0.0, 2.0]]]])
    metric.update(data, pred)
    # Only one valid difference: (1-0)^2 = 1
    assert metric.aggregate == 1
    assert metric.n_observations == 1
    assert pytest.approx(metric.compute()) == 1.0


def test_MSE_update_5d_tensor():
    """Test MSE metric with 5D tensors."""
    metric = MSE()
    data = torch.ones(2, 1, 4, 4, 3)
    pred = torch.zeros(2, 1, 4, 4, 3)
    # Should flatten to 2*3=6 batches of 1x4x4
    metric.update(data, pred)
    assert metric.n_observations == 2 * 1 * 4 * 4 * 3
    assert pytest.approx(metric.compute()) == 1.0


def test_MSE_compute_none_if_no_data():
    """Test MSE compute returns None if no data has been updated."""
    metric = MSE()
    assert metric.compute() is None


def test_MSE_reset_clears_state():
    """Test that reset clears the state of the MSE metric."""
    metric = MSE()
    data = torch.ones(1, 1, 2, 2)
    pred = torch.zeros(1, 1, 2, 2)
    metric.update(data, pred)
    metric.compute()
    metric.reset()
    assert metric.aggregate == 0
    assert metric.n_observations == 0
    assert metric.value == 0


def test_MSE_update_device_switch(monkeypatch):
    """Test MSE metric updates correctly when switching devices."""
    metric = MSE(device=torch.device("cpu"))
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
