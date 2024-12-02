"""Attention mechanism implementations.

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

from chuchichaestli.models.attention.self_attention import SelfAttention
from chuchichaestli.models.attention.conv_attention import ConvAttention
from chuchichaestli.models.attention.attention_gate import AttentionGate

ATTENTION_MAP = {
    "self_attention": SelfAttention,
    "conv_attention": ConvAttention,
    "attention_gate": AttentionGate,
}

__all__ = ["ATTENTION_MAP"]
