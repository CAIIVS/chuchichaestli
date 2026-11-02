# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the metrics.base module."""

import pytest
import torch
from chuchichaestli.metrics.base import EvalMetric


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
