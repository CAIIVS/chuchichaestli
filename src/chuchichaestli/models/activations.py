# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Activation functions for neural networks."""

from functools import partial
from torch import nn
from typing import Literal
from collections.abc import Callable


__all__ = ["ACTIVATION_FUNCTIONS"]


ActivationTypes = Literal[
    "swish",
    "hswish",
    "silu",
    "mish",
    "gelu",
    "relu",
    "relu6",
    "leaky_relu",
    "tanh",
    "sigmoid",
    "identity",
    "prelu",
    "leakyrelu",
    "softplus",
    "leakyrelu,0.1",
    "leakyrelu,0.2",
]


ACTIVATION_FUNCTIONS: dict[str, Callable] = {
    "swish": nn.SiLU,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "identity": nn.Identity,
    "prelu": nn.PReLU,
    "leakyrelu": nn.LeakyReLU,
    "softplus": nn.Softplus,
    "leakyrelu,0.1": partial(nn.LeakyReLU, negative_slope=0.1),
    "leakyrelu,0.2": partial(nn.LeakyReLU, negative_slope=0.2),
}
