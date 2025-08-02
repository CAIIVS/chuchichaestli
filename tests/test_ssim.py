"""Tests for the metrics.ssim module.

This file is part of Chuchichaestli.

Chuchichaestli is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Chuchichaestli is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Chuchichaestli.  If not, see <http://www.gnu.org/licenses/>.

Developed by the Intelligent Vision Systems Group at ZHAW.
"""

import pytest
import torch
from chuchichaestli.metrics.ssim import SSIM


@pytest.fixture(autouse=True)
def patch_tensor_to(monkeypatch):
    """Patch torch.Tensor.to to be a no-op for CPU compatibility."""
    monkeypatch.setattr(torch.Tensor, "to", lambda self, device=None, **k: self)
    yield


def test_ssim_init_defaults():
    """Test default initialization of SSIM metric."""
    metric = SSIM()
    assert metric.kernel_type == "gaussian"
    assert metric.kernel_size == [11, 11]
    assert metric.kernel_sigma == [1.5, 1.5]
    assert metric.k1 == 0.01
    assert metric.k2 == 0.03


def test_ssim_init_custom_kernel():
    """Test initialization of SSIM metric with custom kernel parameters."""
    metric = SSIM(
        window=[9, 7], sigma=[2.0, 1.2], kernel_type="uniform", k1=0.05, k2=0.06
    )
    assert metric.kernel_type == "uniform"
    assert metric.kernel_size == [9, 7]
    assert metric.kernel_sigma == [2.0, 1.2]
    assert metric.k1 == 0.05
    assert metric.k2 == 0.06


def test_ssim_update_and_compute_basic():
    """Test basic update and compute functionality of SSIM metric."""
    metric = SSIM()
    data = torch.ones(1, 1, 32, 32)
    pred = torch.ones(1, 1, 32, 32)
    metric.update(data, pred)
    val = metric.compute()
    assert pytest.approx(val, abs=1e-5) == 1.0


def test_ssim_update_and_compute_perfect_match():
    """Test SSIM metric with perfect match."""
    metric = SSIM()
    data = torch.zeros(1, 1, 16, 16)
    pred = torch.zeros(1, 1, 16, 16)
    metric.update(data, pred)
    val = metric.compute()
    assert pytest.approx(val, abs=1e-5) == 1.0


def test_ssim_update_and_compute_complete_mismatch():
    """Test SSIM metric with complete mismatch."""
    metric = SSIM()
    data = torch.ones(1, 1, 16, 16)
    pred = torch.zeros(1, 1, 16, 16)
    metric.update(data, pred)
    val = metric.compute()
    assert 0 <= val < 0.1


def test_ssim_update_and_compute_almost_identical():
    """Test SSIM metric with almost identical images."""
    metric = SSIM()
    data = torch.ones(1, 1, 32, 32)
    pred = torch.ones(1, 1, 32, 32) - 1e-6
    metric.update(data, pred)
    val = metric.compute()
    assert val > 0.99


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
    data[0, 0, 0, 0] = float('nan')
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
    ssim, cs = SSIM._compute_ssim_and_cs(data, pred, [11, 11], [1.5, 1.5], dtype=torch.float32)
    assert ssim.shape == cs.shape
    assert ssim.dtype == torch.float32


def test_ssim_gaussian_kernel_basic():
    """Test Gaussian kernel generation."""
    kernel = SSIM._gaussian_kernel(1, [5, 5], [1.0, 1.0])
    assert isinstance(kernel, torch.Tensor)
    assert kernel.shape == (1, 1, 5, 5)
    assert torch.isclose(kernel.sum(), torch.tensor(1.0), atol=1e-5)


def test_ssim_compute_ssim_and_cs_uniform_kernel():
    """Test Uniform kernel generation."""
    data = torch.ones(1, 1, 11, 11)
    pred = torch.ones(1, 1, 11, 11)
    ssim, cs = SSIM._compute_ssim_and_cs(data, pred, [11, 11], [1.5, 1.5], kernel_type="uniform")
    assert ssim.shape == cs.shape


def test_ssim_compute_ssim_and_cs_invalid_kernel_type():
    """Test SSIM computation with an invalid kernel type."""
    data = torch.ones(1, 1, 11, 11)
    pred = torch.ones(1, 1, 11, 11)
    with pytest.raises(ValueError):
        ssim, cs = SSIM._compute_ssim_and_cs(data, pred, [11, 11], [1.5, 1.5], kernel_type="invalid")
