# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the metrics.base module."""

import pytest
import torch
from chuchichaestli.metrics.base import (
    sanitize_ndim,
    as_tri_channel,
    as_batched_slices,
    EvalMetric,
)


class DummyTensor(torch.Tensor):
    """A dummy tensor class to simulate the behavior of .to("cuda")."""

    def to(self, device=None, **kwargs):
        """Simulate device transfer."""
        return self


def dummy_tensor(*args, **kwargs):
    """Create a dummy tensor that simulates .to("cuda") behavior."""
    t = torch.Tensor(*args, **kwargs)

    # Monkeypatch .to to be a no-op if called with device="cuda"
    def _to(device=None, **k):
        return t

    t.to = _to
    return t


@pytest.fixture(autouse=True)
def patch_tensor_to(monkeypatch):
    """Patch torch.Tensor.to method to simulate device transfer."""
    monkeypatch.setattr(torch.Tensor, "to", lambda self, device=None, **k: self)
    yield


@pytest.mark.parametrize(
    "x",
    [
        torch.rand((10, 10)),
        torch.rand((1, 10, 10)),
        torch.rand((1, 1, 10, 10)),
    ],
)
def test_sanitize_ndim(x):
    """Test `sanitize_ndim` function."""
    # Test 2/3/4D input tensor
    result = sanitize_ndim(x)
    assert result.shape == (1, 1, 10, 10)


def test_sanitize_ndim_invalid():
    """Test `sanitize_ndim` function: case ValueError."""
    # Test invalid dimensions
    with pytest.raises(ValueError):
        x = torch.rand((1, 1, 10, 10, 10))
        sanitize_ndim(x)


def test_sanitize_ndim_invalid_2_and_3D():
    """Test `sanitize_ndim` function: case check_3D=True."""
    with pytest.raises(ValueError):
        x = torch.rand((1, 1, 10, 10, 10, 1))
        sanitize_ndim(x, check_2D=True, check_3D=True)


def test_sanitize_ndim_invalid_3D():
    """Test `sanitize_ndim` function: case check_3D=True."""
    with pytest.raises(ValueError):
        x = torch.rand((1, 1, 10, 10))
        sanitize_ndim(x, check_2D=False, check_3D=True)


@pytest.mark.parametrize("channel", [1, 2, 3])
def test_as_tri_channel(channel):
    """Test `as_tri_channel` function."""
    # Test single channel
    x = torch.rand((1, channel, 10, 10))
    result = as_tri_channel(x)
    assert result.shape == (1, 3, 10, 10)


def test_as_tri_channel_error():
    """Test `as_tri_channel` function: case UserWarning."""
    # Test more than 3 channels
    x = torch.rand((1, 4, 10, 10))
    with pytest.raises(ValueError):
        as_tri_channel(x)


@pytest.mark.parametrize("sample", [0, 1, 3, 4, 8])
def test_as_batched_slices(sample):
    """Test `as_batched_slices` function."""
    # Test 5D input without sampling
    x = torch.rand((2, 1, 8, 8, 4))
    x_sliced = as_batched_slices(x, sample=sample)
    if sample == 0:
        target = 2 * 4
    else:
        target = 2 * min(sample, 4)
    assert x_sliced.shape == (target, 1, 8, 8)


def test_as_batched_slices_non_5D():
    """Test `as_batched_slices` function case: non 5D."""
    x = torch.rand((2, 3, 10, 10))
    x_slices = as_batched_slices(x)
    assert x_slices.shape == (2, 3, 10, 10)


def test_EvalMetric_init_defaults():
    """Test `EvalMetric` initialization."""
    m = EvalMetric()
    assert m.device == torch.get_default_device()
    assert m.min_value == 0
    assert m.max_value == 1
    assert m.n_observations == 0
    assert m.n_images == 0
    assert m.value == 0
    assert m.aggregate == 0
    assert not m.is_nan
    assert m.nan_count == 0


def test_EvalMetric_init_custom():
    """Test `EvalMetric` initialization with custom parameters."""
    d = torch.device("cpu")
    m = EvalMetric(min_value=-1, max_value=2, n_observations=5, n_images=2, device=d)
    assert m.device == d
    assert m.min_value == -1
    assert m.max_value == 2
    assert m.n_observations == 5
    assert m.n_images == 2


def test_EvalMetric_to_moves_all(monkeypatch):
    """Test `EvalMetric.to` method."""
    m = EvalMetric()
    # .to should be no-op on cpu, but we check all fields get updated
    new_dev = torch.device("cuda")
    m.to(new_dev)
    assert m.device == new_dev


def test_EvalMetric_data_range_setter():
    """Test `EvalMetric.data_range` setter."""
    m = EvalMetric(min_value=2, max_value=4)
    assert m.data_range == 2
    m.data_range = 5
    assert m.max_value == m.min_value + 5


def test_EvalMetric_update():
    """Test `EvalMetric.update` method."""
    m = EvalMetric()
    data = 10 * torch.ones(2, 3, 8, 8)
    pred = torch.zeros(2, 3, 8, 8)
    m.update(data, pred)
    assert m.min_value == 0
    assert m.max_value == 10
    assert m.data_range == 10
    assert m.n_images == 2
    assert m.n_observations == 2 * 3 * 8 * 8


def test_EvalMetric_update_no_range_update():
    """Test `EvalMetric.update` method with `update_range=False`."""
    m = EvalMetric()
    data = 10 * torch.ones(2, 3, 8, 8)
    pred = torch.zeros(2, 3, 8, 8)
    m.update(data, pred, update_range=False)
    # min/max should be unchanged from init
    assert m.min_value == 0
    assert m.max_value == 1


def test_EvalMetric_update_with_nan():
    """Test `EvalMetric.update` method with NaN values."""
    m = EvalMetric()
    data = torch.tensor([[[[1.0, float("nan")]]]])
    pred = torch.tensor([[[[1.0, 2.0]]]])
    m.update(data, pred)
    # One nan in data, so one nan counted
    assert m.nan_count == 1
    # Only one valid
    assert m.n_observations == 1


def test_EvalMetric_reset():
    """Test `EvalMetric.reset` method."""
    metric = EvalMetric()
    data = torch.rand((2, 3, 8, 8))
    prediction = torch.rand((2, 3, 8, 8))
    metric.update(data, prediction)
    metric.reset()
    assert metric.n_images.item() == 0
    assert metric.n_observations.item() == 0
    assert metric.nan_count.item() == 0


def test_EvalMetric_update_to_device_switch(monkeypatch):
    """Test `EvalMetric.update` method with device switch."""
    m = EvalMetric(device=torch.device("cpu"))
    # Simulate prediction device different from m.device
    pred = torch.ones(1, 1, 8, 8)
    data = torch.ones(1, 1, 8, 8)
    # monkeypatch .to to simulate device move
    called = {}

    def fake_to(self, device=None, **kwargs):
        called["to"] = True
        return self

    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    m.device = torch.device("meta")  # Fake device to force .to call
    m.update(data, pred)
    assert called.get("to", False)
