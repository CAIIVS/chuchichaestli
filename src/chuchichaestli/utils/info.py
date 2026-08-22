# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""A torchinfo-style summary table built from the semantic IR."""

from __future__ import annotations
from collections.abc import Callable, Sequence
import torch
from torch import nn
from chuchichaestli.utils.ir import IRGraph, IRNode, GraphLevel, build_ir, normalize_level

__all__ = ["summary"]


def _fmt_shape(size: Sequence[int] | None) -> str:
    return f"[{', '.join(str(s) for s in size)}]" if size else "--"


def _subtree_mult_adds(node: IRNode) -> int:
    """Total mult-adds of a node's leaf descendants (or the node itself)."""
    if not node.children:
        return node.info.mult_adds if node.info is not None else 0
    return sum(_subtree_mult_adds(child) for child in node.children)


# Column key -> (header, cell extractor). "Layer (type)" is always the first column.
_COLUMNS: dict[str, tuple[str, Callable[[IRNode], str]]] = {
    "input": (
        "Input Shape",
        lambda n: _fmt_shape(n.info.input_size if n.info else None),
    ),
    "output": (
        "Output Shape",
        lambda n: _fmt_shape(n.info.output_size if n.info else None),
    ),
    "params": ("Param #", lambda n: f"{n.num_params:,}" if n.num_params else "--"),
    "kernel": (
        "Kernel",
        lambda n: str(n.info.kernel_size) if n.info and n.info.kernel_size else "--",
    ),
    "mult_adds": (
        "Mult-Adds",
        lambda n: f"{m:,}" if (m := _subtree_mult_adds(n)) else "--",
    ),
}


def summary(
    model: nn.Module | None = None,
    input_shape: Sequence[int] | torch.Size | None = None,
    input_dtype: torch.dtype = torch.float32,
    depth: int | str | GraphLevel | None = None,
    columns: Sequence[str] = ("input", "output", "params"),
    graph: IRGraph | None = None,
) -> str:
    """Return a torchinfo-style summary table built from the semantic IR.

    Walks the IR hierarchy (components, levels, blocks, layers), listing each
    node's requested metrics with a totals footer. Reuses `build_ir` for
    tracing, so it needs no extra dependency.

    Args:
        model: Model to summarize (ignored when `graph` is given).
        input_shape: Input shape for shape tracing; structure-only if None.
        input_dtype: Input dtype for tracing.
        depth: Deepest abstraction level to list, using the same convention as
            `matplotlib_diagram` (0=components ... 3=layers); None lists every
            node.
        columns: Metric columns to show, any of `"input"`, `"output"`,
            `"params"`, `"kernel"`, `"mult_adds"`.
        graph: Prebuilt IR graph to reuse instead of tracing `model`.
    """
    unknown = [c for c in columns if c not in _COLUMNS]
    if unknown:
        raise ValueError(f"Unknown summary column(s): {unknown}")
    if graph is None:
        if model is None:
            raise ValueError("Provide either `model` or `graph`.")
        graph = build_ir(model, input_shape, input_dtype)

    cutoff = normalize_level(depth) if depth is not None else None
    extractors = [_COLUMNS[c][1] for c in columns]
    rows: list[list[str]] = []

    def emit(node: IRNode, level: int) -> None:
        if cutoff is not None and level > cutoff:
            return
        rows.append(["  " * level + node.label, *(fn(node) for fn in extractors)])
        for child in node.children:
            emit(child, level + 1)

    emit(graph.root, 0)

    headers = ["Layer (type)", *(_COLUMNS[c][0] for c in columns)]
    widths = [max(len(r[i]) for r in (*rows, headers)) for i in range(len(headers))]
    rule = "=" * (sum(widths) + 2 * (len(widths) - 1))

    def line(cells: Sequence[str]) -> str:
        # Left-justify the tree-indented name column, right-justify the values.
        return "  ".join(
            c.ljust(w) if i == 0 else c.rjust(w)
            for i, (c, w) in enumerate(zip(cells, widths))
        )

    total = graph.root.num_params
    root_info = graph.root.info
    # Prefer the model's own LayerInfo (counts params a module holds directly,
    # which a leaf-sum would miss); fall back to summing leaves otherwise.
    if root_info is not None:
        trainable = root_info.trainable_params
    else:
        trainable = sum(
            n.info.trainable_params
            for n in graph.root.walk()
            if not n.children and n.info is not None
        )
    footer = [
        rule,
        f"Total params: {total:,}",
        f"Trainable params: {trainable:,}",
        f"Non-trainable params: {total - trainable:,}",
    ]
    if "mult_adds" in columns:
        footer.append(f"Total mult-adds: {_subtree_mult_adds(graph.root):,}")
    if root_info is not None and root_info.input_size:
        footer.append(f"Input shape: {_fmt_shape(root_info.input_size)}")
    if root_info is not None and root_info.output_size:
        footer.append(f"Output shape: {_fmt_shape(root_info.output_size)}")
    return "\n".join([rule, line(headers), rule, *(line(r) for r in rows), *footer])
