# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the module tracing helpers."""

import torch
from torch import nn
from chuchichaestli.utils.modules import layer_info


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
