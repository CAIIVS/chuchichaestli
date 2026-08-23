# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the module tracing helpers."""

import torch
from torch import nn
from chuchichaestli.utils.modules import info_forward_pass, layer_info


class _EmbeddingModel(nn.Module):
    """Model whose first layer needs integer inputs."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 4)
        self.lin = nn.Linear(4, 2)

    def forward(self, x):
        """Embed the indices and project them."""
        return self.lin(self.emb(x))


class _KwargModel(nn.Module):
    """Model with a required forward keyword argument."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 2)

    def forward(self, x, *, scale):
        """Project the input and scale it."""
        return self.lin(x) * scale


def test_layer_info_honors_input_dtype():
    """`input_dtype` is used for the forward pass, not only shape inference."""
    info = layer_info(_EmbeddingModel(), input_dtype=torch.long, use_cache=False)
    assert any(i.class_name == "Embedding" for i in info)


def test_layer_info_forwards_kwargs():
    """Additional keyword arguments reach the model's forward pass."""
    info = layer_info(_KwargModel(), scale=2.0, use_cache=False)
    assert any(i.class_name == "Linear" for i in info)


class _TrainOnlyModel(nn.Module):
    """Model with a submodule that only runs while training."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.norm = nn.BatchNorm2d(4)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        """Convolve, normalize, and drop only while training."""
        x = self.norm(self.conv(x))
        if self.training:
            x = self.drop(x)
        return x


def test_trace_restores_buffers_of_a_training_model():
    """A synthetic pass leaves running statistics as it found them."""
    model = _TrainOnlyModel()
    model.train()
    running_mean = model.norm.running_mean.clone()
    info_forward_pass(model, input_shape=(1, 1, 16, 16), use_cache=False)
    assert torch.equal(running_mean, model.norm.running_mean)
    assert model.norm.num_batches_tracked.item() == 0
    assert model.training


def test_trace_keeps_training_only_branches():
    """Tracing stays in the model's mode, so train-only layers are visited."""
    model = _TrainOnlyModel()
    model.train()
    info = info_forward_pass(model, input_shape=(1, 1, 16, 16), use_cache=False)
    assert any(i.class_name == "Dropout" for i in info)


class _WideningModel(nn.Module):
    """Model whose output shape depends on a forward keyword argument."""

    def __init__(self):
        super().__init__()
        self.narrow = nn.Linear(4, 4)
        self.wide = nn.Linear(4, 8)

    def forward(self, x, *, wide=False):
        """Project through the wide or the narrow branch."""
        return self.wide(x) if wide else self.narrow(x)


def _root_output(info):
    """Output size recorded for the traced root module."""
    return [str(i.output_size) for i in info if i.class_name == "_WideningModel"]


def test_cache_key_separates_shape_affecting_kwargs():
    """A call differing only in forward kwargs is not served from the cache."""
    model = _WideningModel()
    narrow = info_forward_pass(model, input_shape=(1, 4), use_cache=True, wide=False)
    wide = info_forward_pass(model, input_shape=(1, 4), use_cache=True, wide=True)
    assert _root_output(narrow) == ["[1, 4]"]
    assert _root_output(wide) == ["[1, 8]"]


def test_cache_key_separates_input_dtypes():
    """Concrete inputs of equal shape but different dtype get separate entries."""
    model = _EmbeddingModel()
    long_info = info_forward_pass(
        model, input_data=torch.zeros(1, 3, dtype=torch.long), use_cache=True
    )
    int_info = info_forward_pass(
        model, input_data=torch.zeros(1, 3, dtype=torch.int32), use_cache=True
    )
    assert long_info is not int_info
