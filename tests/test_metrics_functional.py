# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the metrics.functional module."""

import torch
import pytest
from chuchichaestli.metrics import functional


@pytest.fixture(autouse=True)
def patch_tensor_to(monkeypatch):
    """Patch the `to` method of torch.Tensor to return self."""
    monkeypatch.setattr(torch.Tensor, "to", lambda self, device=None, **k: self)
    yield


def test_mse_perfect_match():
    """Test that MSE is zero for identical tensors."""
    data = torch.ones(2, 1, 8, 8)
    pred = torch.ones(2, 1, 8, 8)
    val = functional.mse(data, pred)
    assert torch.isclose(val, torch.tensor(0.0))


def test_mse_complete_mismatch():
    """Test that MSE is one for completely different tensors."""
    data = torch.ones(2, 1, 8, 8)
    pred = torch.zeros(2, 1, 8, 8)
    val = functional.mse(data, pred)
    assert torch.isclose(val, torch.tensor(1.0))


def test_mse_almost_identical():
    """Test that MSE is small for almost identical tensors."""
    data = torch.ones(1, 1, 8, 8)
    pred = torch.ones(1, 1, 8, 8) - 1e-6
    val = functional.mse(data, pred)
    assert val < 1e-10


def test_mse_with_nan():
    """Test that MSE ignores NaN values."""
    data = torch.tensor([[[[1.0, float("nan")], [1.0, 1.0]]]])
    pred = torch.tensor([[[[1.0, 2.0], [2.0, 1.0]]]])
    val = functional.mse(data, pred)
    # Only 3 valid positions: (1-1)^2, (1-2)^2, (1-1)^2 -> 0 + 1 + 0 = 1/3
    assert torch.isclose(val, torch.tensor(1 / 3))


def test_mse_3d():
    """Test MSE for 3D tensors."""
    data = torch.ones(2, 1, 5, 5, 5)
    pred = torch.zeros(2, 1, 5, 5, 5)
    val = functional.mse(data, pred)
    assert torch.isclose(val, torch.tensor(1.0))


def test_psnr_perfect():
    """Test that PSNR is infinite for identical tensors."""
    data = torch.ones(1, 1, 8, 8)
    pred = torch.ones(1, 1, 8, 8)
    val = functional.psnr(data, pred, data_range=1.0)
    assert val == float("inf") or torch.isinf(val)


def test_psnr_complete_mismatch():
    """Test that PSNR is zero for completely different tensors."""
    data = torch.ones(1, 1, 8, 8)
    pred = torch.zeros(1, 1, 8, 8)
    val = functional.psnr(data, pred, data_range=None)
    assert torch.isclose(val, torch.tensor(0.0))


def test_psnr_almost_identical():
    """Test that PSNR is high for almost identical tensors."""
    data = torch.ones(1, 1, 8, 8)
    pred = torch.ones(1, 1, 8, 8) - 1e-8
    val = functional.psnr(data, pred, data_range=1.0)
    assert val > 80


def test_psnr_3d():
    """Test PSNR for 3D tensors."""
    data = torch.ones(2, 1, 5, 5, 5)
    pred = torch.zeros(2, 1, 5, 5, 5)
    val = functional.psnr(data, pred, data_range=1.0)
    assert torch.isclose(val, torch.tensor(0.0))


def test_psnr_from_batches_simple():
    """Test PSNR from batches with simple values."""
    psnrs = [torch.tensor(20.0), torch.tensor(40.0)]
    pixels = [1, 1]
    val = functional.psnr_from_batches(psnrs, pixel_counts=pixels, data_range=1.0)
    # Should be between the two
    assert 20 < val < 40


def test_psnr_from_batches_weighted():
    """Test PSNR from batches with weighted values."""
    psnrs = [torch.tensor(10.0), torch.tensor(40.0)]
    pixels = [2, 8]
    val = functional.psnr_from_batches(psnrs, pixel_counts=pixels, data_range=1.0)
    assert 10 < val < 40


def test_psnr_from_batches_default_weights():
    """Test PSNR from batches with default weights."""
    psnrs = [torch.tensor(10.0), torch.tensor(40.0)]
    val = functional.psnr_from_batches(psnrs)
    assert 10 < val < 40


def test_ssim_perfect_2d():
    """Test that SSIM is 1.0 for identical 2D tensors."""
    data = torch.ones(1, 1, 16, 16)
    pred = torch.ones(1, 1, 16, 16)
    val = functional.ssim(data, pred, data_range=1.0)
    assert abs(val - 1.0) < 1e-5


def test_ssim_mismatch_2d():
    """Test that SSIM is low for completely different 2D tensors."""
    data = torch.ones(1, 1, 16, 16)
    pred = torch.zeros(1, 1, 16, 16)
    val = functional.ssim(data, pred, data_range=None)
    assert 0 <= float(val) < 0.1


def test_ssim_no_reduction():
    """Test that SSIM can return per-sample values without reduction."""
    data = torch.ones(4, 1, 16, 16)
    pred = torch.zeros(4, 1, 16, 16)
    val = functional.ssim(data, pred, data_range=None, reduction=None)
    assert isinstance(val, torch.Tensor)
    assert len(val) == 4


def test_ssim_perfect_3d():
    """Test that SSIM is 1.0 for identical 3D tensors."""
    data = torch.ones(1, 1, 8, 8, 8)
    pred = torch.ones(1, 1, 8, 8, 8)
    val = functional.ssim(data, pred, data_range=1.0, kernel_size=3)
    assert abs(float(val) - 1.0) < 1e-5


def test_ssim_mismatch_3d():
    """Test that SSIM is low for completely different 3D tensors."""
    data = torch.ones(1, 1, 8, 8, 8)
    pred = torch.zeros(1, 1, 8, 8, 8)
    val = functional.ssim(data, pred, data_range=1.0, kernel_size=3)
    assert 0 <= float(val) < 0.1


def test_lpips_runs(monkeypatch):
    """Test that LPIPS runs without requiring a real model."""

    class DummyLPIPS:
        def __init__(self, *a, **k):
            pass

        def __call__(self, data, pred):
            return torch.tensor(0.5)

    monkeypatch.setattr(functional, "LPIPSLoss", DummyLPIPS)
    data = torch.ones(1, 3, 16, 16)
    pred = torch.zeros(1, 3, 16, 16)
    val = functional.lpips(data, pred, model="vgg16", embedding=False)
    assert torch.isclose(val, torch.tensor(0.5))


def test_lpips_3d(monkeypatch):
    """Test that LPIPS works for 3D tensors."""

    class DummyLPIPS:
        def __init__(self, *a, **k):
            pass

        def __call__(self, data, pred):
            return torch.tensor(0.42)

    monkeypatch.setattr(functional, "LPIPSLoss", DummyLPIPS)
    data = torch.ones(1, 3, 8, 8, 8)
    pred = torch.zeros(1, 3, 8, 8, 8)
    val = functional.lpips(data, pred, model="vgg16", embedding=False)
    assert torch.isclose(val, torch.tensor(0.42))
