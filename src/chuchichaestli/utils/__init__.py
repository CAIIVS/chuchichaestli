# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Various utilities for chuchichaestli."""

from chuchichaestli.utils.functools import (
    partialclass,
    alias_kwargs,
    prod,
    nested_list_size,
    map_nested,
    per_position,
    per_position_args,
)
from chuchichaestli.utils.formatting import metric_suffix
from chuchichaestli.utils.modules import (
    info_forward_pass,
    layer_info,
    clear_info_cache,
    get_chuchichaestli_block_type,
    get_layer_type,
)
from chuchichaestli.utils.ir import build_ir
from chuchichaestli.utils.info import summary
from chuchichaestli.utils.visualization import (
    MermaidDiagram,
    mermaid_diagram,
    matplotlib_diagram,
)

__all__ = [
    "partialclass",
    "alias_kwargs",
    "prod",
    "nested_list_size",
    "map_nested",
    "per_position",
    "per_position_args",
    "metric_suffix",
    "info_forward_pass",
    "layer_info",
    "clear_info_cache",
    "get_chuchichaestli_block_type",
    "get_layer_type",
    "MermaidDiagram",
    "mermaid_diagram",
    "matplotlib_diagram",
    "build_ir",
    "summary",
]
