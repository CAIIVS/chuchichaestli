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
    broadcast,
    broadcast_kwargs,
)
from chuchichaestli.utils.units import metric_suffix, nbytes
from chuchichaestli.utils.tensors import (
    as_array,
    as_inexact,
    npy_to_torch_dtype,
    torch_to_npy_dtype,
    view_along_axis,
)
from chuchichaestli.utils.modules import (
    info_forward_pass,
    layer_info,
    clear_info_cache,
    get_chuchichaestli_block_type,
    get_layer_type,
)
from chuchichaestli.utils.arithmetic import Laurent, Lifting, Step, divide, factor, reverse
from chuchichaestli.utils.ir import build_ir
from chuchichaestli.utils.info import summary
from chuchichaestli.utils.visualization import (
    MermaidDiagram,
    mermaid_diagram,
    matplotlib_diagram,
)

__all__ = [
    "Laurent",
    "Lifting",
    "Step",
    "divide",
    "factor",
    "reverse",
    "partialclass",
    "alias_kwargs",
    "prod",
    "nested_list_size",
    "map_nested",
    "broadcast",
    "broadcast_kwargs",
    "metric_suffix",
    "nbytes",
    "as_array",
    "as_inexact",
    "npy_to_torch_dtype",
    "torch_to_npy_dtype",
    "view_along_axis",
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
