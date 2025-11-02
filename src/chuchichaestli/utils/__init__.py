# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Various utilities for chuchichaestli."""

from chuchichaestli.utils.functools import (
    partialclass,
    prod,
    nested_list_size,
    map_nested,
)
from chuchichaestli.utils.formatting import metric_suffix
from chuchichaestli.utils.modules import (
    info_forward_pass,
    layer_info,
    clear_info_cache,
    get_chuchichaestli_block_type,
    get_layer_type,
)
from chuchichaestli.utils.visualization import MermaidDiagram, mermaid_diagram

__all__ = [
    "partialclass",
    "prod",
    "nested_list_size",
    "map_nested",
    "metric_suffix",
    "info_forward_pass",
    "layer_info",
    "clear_info_cache",
    "get_chuchichaestli_block_type",
    "get_layer_type",
    "MermaidDiagram",
    "mermaid_diagram",
]
