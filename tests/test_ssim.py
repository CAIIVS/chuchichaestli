# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the SSIM module."""

import pytest
import torch
from chuchichaestli.metrics.ssim import SSIM, SSIMLoss


@pytest.fixture(autouse=True)
def patch_tensor_to(monkeypatch):
    """Patch torch.Tensor.to to be a no-op for CPU compatibility."""
    monkeypatch.setattr(torch.Tensor, "to", lambda self, device=None, **k: self)
    yield


def test_ssim_init_defaults():
    """Test default initialization of SSIM metric."""
    metric = SSIM()
    assert metric.kernel_type == "gaussian"
    assert metric.kernel_size == 11
    assert metric.kernel_sigma == 1.5
    assert metric.k1 == 0.01
    assert metric.k2 == 0.03


def test_ssim_init_custom_kernel():
    """Test initialization of SSIM metric with custom kernel parameters."""
    metric = SSIM(
        kernel_size=[9, 7],
        kernel_sigma=[2.0, 1.2],
        kernel_type="uniform",
        k1=0.05,
        k2=0.06,
    )
    assert metric.kernel_type == "uniform"
    assert metric.kernel_size == [9, 7]
    assert metric.kernel_sigma == [2.0, 1.2]
    assert metric.k1 == 0.05
    assert metric.k2 == 0.06


def test_ssim_update_and_compute_basic_2D():
    """Test basic update and compute functionality of SSIM metric."""
    metric = SSIM()
    data = torch.ones(1, 1, 32, 32)
    pred = torch.ones(1, 1, 32, 32)
    metric.update(data, pred)
    val = metric.compute()
    assert pytest.approx(val, abs=1e-5) == 1.0


def test_ssim_update_and_compute_identical_2D():
    """Test SSIM metric with perfect match."""
    metric = SSIM()
    data = torch.zeros(1, 1, 16, 16)
    pred = torch.zeros(1, 1, 16, 16)
    metric.update(data, pred)
    val = metric.compute()
    assert pytest.approx(val, abs=1e-5) == 1.0


def test_ssim_update_and_compute_complete_mismatch_2D():
    """Test SSIM metric with complete mismatch."""
    metric = SSIM()
    data = torch.ones(1, 1, 16, 16)
    pred = torch.zeros(1, 1, 16, 16)
    metric.update(data, pred)
    val = metric.compute()
    assert 0 <= val < 0.1


def test_ssim_update_and_compute_almost_identical_2D():
    """Test SSIM metric with almost identical images."""
    metric = SSIM()
    data = torch.ones(1, 1, 32, 32)
    pred = torch.ones(1, 1, 32, 32) - 1e-6
    metric.update(data, pred)
    val = metric.compute()
    assert val > 0.99


def test_ssim_update_and_compute_multi_channel():
    """Test SSIM metric with multi-channel images."""
    metric = SSIM()
    data = torch.ones(1, 3, 32, 32)
    pred = torch.ones(1, 3, 32, 32)
    metric.update(data, pred)
    val = metric.compute()
    assert pytest.approx(val, abs=1e-5) == 1.0


def test_ssim_update_and_compute_batch_2D():
    """Test SSIM metric with batch of images."""
    metric = SSIM()
    data = torch.ones(4, 1, 32, 32)
    pred = torch.ones(4, 1, 32, 32)
    metric.update(data, pred)
    val = metric.compute()
    assert pytest.approx(val, abs=1e-5) == 1.0


def test_ssim_update_and_compute_explicit_kernel_2D():
    """Test SSIM metric with explicit kernel size."""
    metric = SSIM(kernel_size=7)
    data = torch.ones(1, 1, 32, 32)
    pred = torch.ones(1, 1, 32, 32)
    metric.update(data, pred)
    val = metric.compute()
    assert val == pytest.approx(1.0, abs=1e-5)


def test_ssim_update_and_compute_3D():
    """Test SSIM metric with 3D data."""
    metric = SSIM(kernel_size=5, kernel_sigma=1.0)
    data = torch.ones(1, 1, 8, 8, 8)
    pred = torch.ones(1, 1, 8, 8, 8)
    metric.update(data, pred)
    val = metric.compute()
    assert pytest.approx(val, abs=1e-5) == 1.0


def test_ssim_update_and_compute_mismatch_3D():
    """Test SSIM metric with complete mismatch in 3D data."""
    metric = SSIM(kernel_size=3, kernel_sigma=1.0)
    data = torch.ones(1, 1, 6, 6, 6)
    pred = torch.zeros(1, 1, 6, 6, 6)
    metric.update(data, pred)
    val = metric.compute()
    assert 0 <= val < 0.1


def test_ssim_update_and_compute_almost_identical_3D():
    """Test SSIM metric with almost identical 3D data."""
    metric = SSIM(kernel_size=3, kernel_sigma=1.0)
    data = torch.ones(1, 1, 8, 8, 8)
    pred = torch.ones(1, 1, 8, 8, 8) - 1e-5
    metric.update(data, pred)
    val = metric.compute()
    assert val > 0.99


def test_ssim_update_and_compute_multi_channel_3D():
    """Test SSIM metric with multi-channel 3D data."""
    metric = SSIM(kernel_size=3, kernel_sigma=1.0)
    data = torch.ones(1, 2, 8, 8, 8)
    pred = torch.ones(1, 2, 8, 8, 8)
    metric.update(data, pred)
    val = metric.compute()
    assert pytest.approx(val, abs=1e-5) == 1.0


def test_ssim_update_and_compute_none_if_no_data():
    """Test SSIM metric compute returns None if no data has been updated."""
    metric = SSIM()
    assert metric.compute() is None


def test_ssim_reset_clears_state():
    """Test that reset clears the state of the SSIM metric."""
    metric = SSIM()
    data = torch.ones(1, 1, 16, 16)
    pred = torch.zeros(1, 1, 16, 16)
    metric.update(data, pred)
    metric.compute()
    metric.reset()
    assert metric.aggregate == 0
    assert metric.n_images == 0


def test_ssim_update_device_switch(monkeypatch):
    """Test that SSIM metric updates correctly when switching devices."""
    metric = SSIM(device=torch.device("cpu"))
    called = {}

    def fake_to(self, device=None, **kwargs):
        called["to"] = True
        return self

    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    data = torch.ones(1, 1, 32, 32)
    pred = torch.ones(1, 1, 32, 32)
    metric.device = torch.device("meta")  # fake device to force .to call
    metric.update(data, pred)
    assert called.get("to", False)


def test_ssim_update_with_nan():
    """Test SSIM metric handling of NaN values."""
    metric = SSIM()
    data = torch.ones(1, 1, 32, 32)
    data[0, 0, 0, 0] = float("nan")
    pred = torch.ones(1, 1, 32, 32)
    metric.update(data, pred)
    val = metric.compute()
    assert 0 <= val <= 1


def test_ssim_compute_ssim_and_cs_perfect():
    """Test SSIM computation with perfect match."""
    data = torch.ones(1, 1, 11, 11)
    pred = torch.ones(1, 1, 11, 11)
    ssim, cs = SSIM._compute_ssim_and_cs(data, pred, [11, 11], [1.5, 1.5])
    assert torch.allclose(ssim, torch.ones_like(ssim), atol=1e-5)
    assert cs is not None


def test_ssim_compute_ssim_and_cs_shape_mismatch():
    """Test SSIM computation with shape mismatch."""
    data = torch.ones(1, 1, 11, 11)
    pred = torch.ones(1, 1, 10, 10)
    with pytest.raises(ValueError):
        SSIM._compute_ssim_and_cs(data, pred, [11, 11], [1.5, 1.5])


def test_ssim_compute_ssim_and_cs_explicit_dtype():
    """Test SSIM computation with explicit type input."""
    data = torch.ones(1, 1, 11, 11)
    pred = torch.ones(1, 1, 11, 11)
    ssim, cs = SSIM._compute_ssim_and_cs(
        data, pred, [11, 11], [1.5, 1.5], dtype=torch.float32
    )
    assert ssim.shape == cs.shape
    assert ssim.dtype == torch.float32


def test_ssim_compute_ssim_and_cs_wrong_kernel_size():
    """Test SSIM computation with explicit type input."""
    data = torch.ones(1, 1, 11, 11, 11)
    pred = torch.ones(1, 1, 11, 11, 11)
    ssim, cs = SSIM._compute_ssim_and_cs(
        data, pred, [11, 11], [1.5, 1.5], dtype=torch.float32
    )
    assert ssim.shape == cs.shape
    assert ssim.dtype == torch.float32


def test_ssim_gaussian_kernel_basic():
    """Test Gaussian 2D kernel generation."""
    kernel = SSIM._gaussian_kernel(1, [5, 5], [1.0, 1.0])
    assert isinstance(kernel, torch.Tensor)
    assert kernel.shape == (1, 1, 5, 5)
    assert torch.isclose(kernel.sum(), torch.tensor(1.0), atol=1e-5)


def test_ssim_gaussian_kernel_3D():
    """Test Gaussian 3D kernel generation."""
    kernel = SSIM._gaussian_kernel(1, [5, 5, 5], [1.0, 1.0, 1.0])
    assert isinstance(kernel, torch.Tensor)
    assert kernel.shape == (1, 1, 5, 5, 5)
    assert torch.isclose(kernel.sum(), torch.tensor(1.0), atol=1e-5)


def test_ssim_gaussian_kernel_invalid():
    """Test Gaussian invalid kernel generation."""
    with pytest.raises(ValueError):
        SSIM._gaussian_kernel(1, [5, 5, 5, 5], [1.0, 1.0, 1.0, 1.0])


def test_ssim_compute_ssim_and_cs_uniform_kernel():
    """Test Uniform kernel generation."""
    data = torch.ones(1, 1, 11, 11)
    pred = torch.ones(1, 1, 11, 11)
    ssim, cs = SSIM._compute_ssim_and_cs(
        data, pred, [11, 11], [1.5, 1.5], kernel_type="uniform"
    )
    assert ssim.shape == cs.shape


def test_ssim_compute_ssim_and_cs_invalid_kernel_type():
    """Test SSIM computation with an invalid kernel type."""
    data = torch.ones(1, 1, 11, 11)
    pred = torch.ones(1, 1, 11, 11)
    with pytest.raises(ValueError):
        ssim, cs = SSIM._compute_ssim_and_cs(
            data, pred, [11, 11], [1.5, 1.5], kernel_type="invalid"
        )


def test_ssimloss_basic_2D():
    """Test basic functionality of SSIMLoss."""
    loss_fn = SSIMLoss()
    data = torch.ones(2, 1, 32, 32)
    pred = torch.ones(2, 1, 32, 32)
    loss = loss_fn(data, pred)
    assert loss < 1e-5


def test_ssimloss_basic_2D_no_reduction():
    """Test non-reduction SSIMLoss."""
    loss_fn = SSIMLoss()
    data = torch.ones(2, 1, 32, 32)
    pred = torch.ones(2, 1, 32, 32)
    loss = loss_fn(data, pred, reduction=None)
    assert loss.shape == (2, 1)


def test_ssimloss_mismatch_2D():
    """Test SSIMLoss with complete mismatch."""
    loss_fn = SSIMLoss()
    data = torch.ones(1, 1, 16, 16)
    pred = torch.zeros(1, 1, 16, 16)
    loss = loss_fn(data, pred)
    assert 0.9 < loss <= 1.0


def test_ssimloss_basic_3D():
    """Test basic functionality of SSIMLoss with 3D data."""
    loss_fn = SSIMLoss(kernel_size=3, kernel_sigma=1.0)
    data = torch.ones(1, 1, 8, 8, 8)
    pred = torch.ones(1, 1, 8, 8, 8)
    loss = loss_fn(data, pred)
    assert loss < 1e-5


def test_ssimloss_mismatch_3D():
    """Test SSIMLoss with complete mismatch in 3D data."""
    loss_fn = SSIMLoss(kernel_size=3, kernel_sigma=1.0)
    data = torch.ones(1, 1, 6, 6, 6)
    pred = torch.zeros(1, 1, 6, 6, 6)
    loss = loss_fn(data, pred)
    assert 0.9 < loss <= 1.0
