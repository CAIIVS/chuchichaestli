# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Activation functions for neural networks."""

from functools import partial
from torch import nn
from collections.abc import Callable


ACTIVATION_FUNCTIONS: dict[str, Callable] = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "prelu": nn.PReLU,
    "leakyrelu": nn.LeakyReLU,
    "softplus": nn.Softplus,
    "leakyrelu,0.1": partial(nn.LeakyReLU, negative_slope=0.1),
    "leakyrelu,0.2": partial(nn.LeakyReLU, negative_slope=0.2),
}
