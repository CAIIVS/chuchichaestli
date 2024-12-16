"""Activation functions for neural networks.

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
    "leakyrelu,0.1": partial(nn.LeakyReLU, negative_slope=0.1),
    "leakyrelu,0.2": partial(nn.LeakyReLU, negative_slope=0.2),
}
