"""Tests for the fetch module.

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
import torch.nn as nn
from torch import Tensor

from chuchichaestli.injectables.batchfetch import ExtractD, ExtractS
from chuchichaestli.injectables.diffusion import CondBase

from chuchichaestli.diffusion.ddpm.ddpm import DDPM
from chuchichaestli.models.unet import UNet

@pytest.fixture
def cond_base_model():
    model = UNet(in_channels=2)
    scheduler = DDPM(num_timesteps=5)
    train_loss = nn.MSELoss()
    valid_loss = nn.MSELoss()
    return CondBase(model, scheduler, train_loss, valid_loss)


def test_train_step_cond_base_with_dict_input(cond_base_model):
    batch_size = 2
    channels = 1
    height, width = 64, 64

    batch = {
        "cond": torch.randn(batch_size, channels, height, width),
        "target": torch.randn(batch_size, channels, height, width),
    }
    cond_base_model.train_fetch = ExtractD(["cond", "target"])
    result = cond_base_model.training_step(batch, 0)

    assert isinstance(result, dict)
    assert isinstance(result["loss"], Tensor)
    assert result["inputs"].shape == (batch_size, channels, height, width)
    assert result["output"].shape == (batch_size, channels, height, width)
    assert result["target"].shape == (batch_size, channels, height, width)


def test_train_step_cond_base_with_sequence_input(cond_base_model):
    batch_size = 2
    channels = 1
    height, width = 64, 64

    batch = (
        torch.randn(batch_size, channels, height, width),
        torch.randn(batch_size, channels, height, width),
    )
    cond_base_model.train_fetch = ExtractS([0, 1])
    result = cond_base_model.training_step(batch, 0)

    assert isinstance(result, dict)
    assert isinstance(result["loss"], Tensor)
    assert result["inputs"].shape == (batch_size, channels, height, width)
    assert result["output"].shape == (batch_size, channels, height, width)
    assert result["target"].shape == (batch_size, channels, height, width)


def test_valid_step_cond_base_with_dict_input(cond_base_model):
    batch_size = 2
    channels = 1
    height, width = 64, 64

    batch = {
        "cond": torch.randn(batch_size, channels, height, width),
        "target": torch.randn(batch_size, channels, height, width),
    }
    cond_base_model.valid_fetch = ExtractD(["cond", "target"])
    result = cond_base_model.validation_step(batch, 0)

    assert isinstance(result, dict)
    assert isinstance(result["loss"], Tensor)
    assert torch.equal(result["target"], batch["target"])
    assert result["inputs"].shape == (batch_size, channels, height, width)
    assert result["output"].shape == (batch_size, channels, height, width)
    assert result["target"].shape == (batch_size, channels, height, width)


def test_valid_step_cond_base_with_sequence_input(cond_base_model):
    batch_size = 2
    channels = 1
    height, width = 64, 64

    batch = (
        torch.randn(batch_size, channels, height, width),
        torch.randn(batch_size, channels, height, width),
    )
    result = cond_base_model.validation_step(batch, 0)

    assert isinstance(result, dict)
    assert isinstance(result["loss"], Tensor)
    assert torch.equal(result["target"], batch[1])
    assert result["inputs"].shape == (batch_size, channels, height, width)
    assert result["output"].shape == (batch_size, channels, height, width)
    assert result["target"].shape == (batch_size, channels, height, width)
