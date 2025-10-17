# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Attention mechanism implementations."""

from chuchichaestli.models.attention.self_attention import SelfAttention
from chuchichaestli.models.attention.conv_attention import ConvAttention
from chuchichaestli.models.attention.attention_gate import AttentionGate

ATTENTION_MAP = {
    "self_attention": SelfAttention,
    "conv_attention": ConvAttention,
    "attention_gate": AttentionGate,
}

__all__ = ["ATTENTION_MAP"]
