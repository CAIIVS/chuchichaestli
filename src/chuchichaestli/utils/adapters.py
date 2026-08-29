# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Architecture-aware adapters mapping models to the semantic IR."""

from __future__ import annotations
import re
from types import SimpleNamespace
from typing import Protocol, runtime_checkable
from torch import nn
from chuchichaestli.utils.modules import (
    LayerInfo,
    get_chuchichaestli_block_type,
    get_layer_type,
)
from chuchichaestli.utils.ir import (
    NodeRole,
    EdgeKind,
    Geometry,
    IRNode,
    IREdge,
    IRGraph,
)

__all__ = [
    "SemanticAdapter",
    "AdapterRegistry",
    "UNetAdapter",
    "AutoencoderAdapter",
    "SequentialAdapter",
    "GenericAdapter",
    "default_registry",
]

_CLASSES: SimpleNamespace | None = None


def _classes() -> SimpleNamespace:
    """Lazily import and cache the model classes used for isinstance dispatch."""
    global _CLASSES
    if _CLASSES is not None:
        return _CLASSES
    from chuchichaestli.models.blocks import (
        BaseConvBlock,
        DownBlock,
        MidBlock,
        UpBlock,
        AutoencoderDownBlock,
        AutoencoderMidBlock,
        AutoencoderUpBlock,
        ResidualBlock,
        ResidualBottleneck,
        LiteResidualBlock,
        GaussianNoiseBlock,
    )
    from chuchichaestli.models.downsampling import DOWNSAMPLE_BLOCKS
    from chuchichaestli.models.upsampling import UPSAMPLE_BLOCKS
    from chuchichaestli.models.autoencoder.autoencoder import Autoencoder
    from chuchichaestli.models.autoencoder.vae import VAE
    from chuchichaestli.models.autoencoder.vqvae import VQVAE

    residual = (ResidualBlock, ResidualBottleneck, LiteResidualBlock)
    containers = (nn.Sequential, nn.ModuleList, nn.ModuleDict)
    blocks = (
        BaseConvBlock,
        DownBlock,
        MidBlock,
        UpBlock,
        AutoencoderDownBlock,
        AutoencoderMidBlock,
        AutoencoderUpBlock,
    ) + residual
    _CLASSES = SimpleNamespace(
        Autoencoder=Autoencoder,
        VAE=VAE,
        VQVAE=VQVAE,
        GaussianNoiseBlock=GaussianNoiseBlock,
        RESIDUAL=residual,
        RECURSE=containers + blocks,
        DOWNSAMPLE=DOWNSAMPLE_BLOCKS,
        UPSAMPLE=UPSAMPLE_BLOCKS,
    )
    return _CLASSES


def _san(name: str) -> str:
    """Sanitize a path segment (no separators/whitespace)."""
    return re.sub(r"[^0-9A-Za-z._-]", "_", str(name))


def _type_label(module: nn.Module) -> str:
    """Human-readable type label for a module."""
    return get_chuchichaestli_block_type(module) or get_layer_type(module)


@runtime_checkable
class SemanticAdapter(Protocol):
    """Maps a model family to the semantic IR."""

    def matches(self, model: nn.Module) -> bool:
        """Whether this adapter can decompose `model`.

        Args:
            model: Root module to test.
        """
        ...

    def build(
        self,
        model: nn.Module,
        info_by_id: dict[int, LayerInfo],
        dataflow: dict[int, set[int]] | None = None,
    ) -> IRGraph:
        """Build the IR graph for `model`.

        Args:
            model: Root module to decompose.
            info_by_id: Map from id(module) to LayerInfo.
            dataflow: Optional traced leaf-module dataflow (see `trace_dataflow`);
                only adapters with `uses_dataflow` set consume it.
        """
        ...


class _Builder:
    """Shared IR-construction helpers for concrete adapters."""

    def __init__(self, model: nn.Module, info_by_id: dict[int, LayerInfo]) -> None:
        """Constructor.

        Args:
            model: Root module being decomposed.
            info_by_id: Map from id(module) to LayerInfo.
        """
        self.model = model
        self.info_by_id = info_by_id
        self.edges: list[IREdge] = []
        self._leaves: list[IRNode] = []
        self._used: set[str] = set()

    def _uid(self, candidate: str) -> str:
        uid, i = candidate, 2
        while uid in self._used:
            uid = f"{candidate}_{i}"
            i += 1
        self._used.add(uid)
        return uid

    def _info(self, module: nn.Module | None) -> LayerInfo | None:
        return self.info_by_id.get(id(module)) if module is not None else None

    def _geom(
        self,
        info: LayerInfo | None,
        module: nn.Module | None = None,
        level_index: int | None = None,
        is_down: bool = False,
        is_up: bool = False,
    ) -> Geometry:
        channels = spatial = None
        if info is not None and len(info.output_size) > 1:
            channels = info.output_size[1]
            if len(info.output_size) > 2:
                spatial = tuple(info.output_size[2:])
        if channels is None and module is not None:
            channels = getattr(module, "out_channels", None) or getattr(
                module, "channels", None
            )
        return Geometry(channels, spatial, level_index, is_down, is_up)

    def _root(self) -> IRNode:
        return IRNode(
            id=self._uid("model"),
            role=NodeRole.MODEL,
            type_label=_type_label(self.model) or self.model.__class__.__name__,
            label=self.model.__class__.__name__,
            module=self.model,
            info=self._info(self.model),
            depth=0,
        )

    def _child(
        self,
        parent: IRNode,
        key: str,
        role: NodeRole,
        label: str,
        module: nn.Module | None = None,
        geometry: Geometry | None = None,
    ) -> IRNode:
        nid = self._uid(f"{parent.id}/{_san(key)}")
        info = self._info(module)
        node = IRNode(
            id=nid,
            role=role,
            type_label=_type_label(module) if module is not None else label,
            label=label,
            module=module,
            info=info,
            depth=nid.count("/"),
            geometry=geometry or self._geom(info, module),
            num_params=info.num_params if info is not None else 0,
        )
        parent.add(node)
        return node

    def _level(self, component: IRNode, index: int) -> IRNode:
        return self._child(
            component,
            f"level{index}",
            NodeRole.LEVEL,
            f"Level {index}",
            geometry=Geometry(level_index=index),
        )

    def _block(
        self,
        level: IRNode,
        key: str,
        module: nn.Module,
        level_index: int | None = None,
        is_down: bool = False,
        is_up: bool = False,
    ) -> IRNode:
        info = self._info(module)
        node = self._child(
            level,
            key,
            NodeRole.BLOCK,
            module.__class__.__name__,
            module=module,
            geometry=self._geom(info, module, level_index, is_down, is_up),
        )
        self._emit_layers(node, module)
        self._maybe_residual(node, module)
        return node

    def _layer(self, block: IRNode, base_id: str, module: nn.Module) -> IRNode:
        nid = self._uid(base_id)
        info = self._info(module)
        node = IRNode(
            id=nid,
            role=NodeRole.LAYER,
            type_label=_type_label(module),
            label=module.__class__.__name__,
            module=module,
            info=info,
            depth=nid.count("/"),
            geometry=self._geom(info, module),
            num_params=info.num_params if info is not None else 0,
        )
        block.add(node)
        self._leaves.append(node)
        return node

    def _emit_layers(self, block: IRNode, module: nn.Module) -> None:
        recurse = _classes().RECURSE

        # Leaf layers are direct tree children of the block; the nested module
        # path is flattened into one dotted id segment (e.g. `res_block.conv1`)
        # so a layer's id depth equals its true tree depth (block depth + 1).
        def walk(m: nn.Module, name_path: tuple[str, ...]) -> None:
            kids = list(m.named_children())
            if kids and isinstance(m, recurse):
                for name, child in kids:
                    walk(child, (*name_path, name))
            else:
                suffix = ".".join(_san(p) for p in name_path)
                self._layer(block, f"{block.id}/{suffix}", m)

        kids = list(module.named_children())
        if kids and isinstance(module, recurse):
            for name, child in kids:
                walk(child, (name,))
        else:
            self._layer(block, f"{block.id}/{_san(module.__class__.__name__)}", module)

    def _maybe_residual(self, block: IRNode, module: nn.Module) -> None:
        residual = _classes().RESIDUAL
        if len(block.children) >= 2 and any(
            isinstance(sm, residual) for sm in module.modules()
        ):
            self.edges.append(
                IREdge(block.children[0].id, block.children[-1].id, EdgeKind.RESIDUAL)
            )

    def _chain_forward(self) -> None:
        for prev, nxt in zip(self._leaves, self._leaves[1:]):
            self.edges.append(IREdge(prev.id, nxt.id, EdgeKind.FORWARD))

    def _dataflow_forward(self, dataflow: dict[int, set[int]]) -> None:
        """Emit FORWARD edges from a traced module-id dataflow map.

        Maps each producer/consumer leaf module back to its IR node and adds
        one edge per real tensor dependency, so branched models keep their
        parallel paths and merges instead of a fabricated serial chain.
        """
        by_module: dict[int, str] = {}
        for leaf in self._leaves:
            if leaf.module is not None:
                by_module.setdefault(id(leaf.module), leaf.id)
        emitted: list[tuple[str, str]] = []
        for src_mid, dst_mids in dataflow.items():
            src = by_module.get(src_mid)
            if src is None:
                continue
            for dst_mid in dst_mids:
                dst = by_module.get(dst_mid)
                if dst is not None and dst != src:
                    emitted.append((src, dst))
        for src, dst in sorted(set(emitted)):
            self.edges.append(IREdge(src, dst, EdgeKind.FORWARD))

    def _graph(
        self, root: IRNode, dataflow: dict[int, set[int]] | None = None
    ) -> IRGraph:
        if dataflow:
            self._dataflow_forward(dataflow)
        else:
            self._chain_forward()
        return IRGraph(root=root, edges=self.edges)


class UNetAdapter:
    """Adapter for `UNet` models with encoder/bottleneck/decoder and skips."""

    def matches(self, model: nn.Module) -> bool:
        """Match any module exposing the U-Net attribute contract.

        Args:
            model: Root module to test.
        """
        return (
            hasattr(model, "down_blocks")
            and hasattr(model, "up_blocks")
            and hasattr(model, "mid_block")
            and not isinstance(model, _classes().Autoencoder)
        )

    def build(
        self,
        model: nn.Module,
        info_by_id: dict[int, LayerInfo],
        dataflow: dict[int, set[int]] | None = None,
    ) -> IRGraph:
        """Decompose a U-Net into the IR, replaying its skip pairing.

        Args:
            model: U-Net model.
            info_by_id: Map from id(module) to LayerInfo.
            dataflow: Unused; the U-Net forward flow is built explicitly.
        """
        del dataflow
        c = _classes()
        b = _Builder(model, info_by_id)
        root = b._root()
        nbpl = getattr(model, "num_blocks_per_level", 1)
        skip_to_all = getattr(model, "skip_connection_to_all_blocks", False)
        # the model records its own pairing; fall back to counting within a level
        is_skip_source = getattr(model, "skip_sources", None)
        up_roles = getattr(model, "up_roles", None)

        if getattr(model, "time_emb", None) is not None:
            cond = b._child(root, "conditioning", NodeRole.COMPONENT, "Conditioning")
            lvl = b._level(cond, 0)
            b._block(lvl, "time_emb", model.time_emb)

        enc = b._child(root, "encoder", NodeRole.COMPONENT, "Encoder")
        li = 0
        level = b._level(enc, li)
        b._block(level, "conv_in", model.conv_in, level_index=li)
        skip_sources: list[str] = []
        in_level = 0
        for i, m in enumerate(model.down_blocks):
            if isinstance(m, c.DOWNSAMPLE):
                b._block(level, f"downsample{li}", m, level_index=li, is_down=True)
                li += 1
                in_level = 0
                level = b._level(enc, li)
                continue
            if isinstance(m, c.GaussianNoiseBlock):
                b._block(level, f"noise{i}", m, level_index=li)
                continue
            block = b._block(level, f"block{i}", m, level_index=li)
            in_level += 1
            terminal = is_skip_source[i] if is_skip_source else in_level == nbpl
            if terminal:
                skip_sources.append(block.id)

        bottleneck = b._child(root, "bottleneck", NodeRole.COMPONENT, "Bottleneck")
        b._block(b._level(bottleneck, 0), "mid", model.mid_block)

        dec = b._child(root, "decoder", NodeRole.COMPONENT, "Decoder")
        dli = 0
        level = b._level(dec, dli)
        in_level = 0
        held: str | None = None
        for i, m in enumerate(model.up_blocks):
            if isinstance(m, (*c.UPSAMPLE, c.GaussianNoiseBlock)):
                is_up = isinstance(m, c.UPSAMPLE)
                b._block(level, f"sample{i}", m, level_index=dli, is_up=is_up)
                if is_up:
                    dli += 1
                    in_level = 0
                    level = b._level(dec, dli)
                continue
            block = b._block(level, f"block{i}", m, level_index=dli)
            first = up_roles[i] == "first" if up_roles else in_level == 0
            in_level += 1
            src = None
            if first and skip_sources:
                held = src = skip_sources.pop()
            elif skip_to_all:
                src = held
            action = getattr(m, "skip_connection_action", None)
            if src is not None and action is not None:
                b.edges.append(IREdge(src, block.id, EdgeKind.SKIP, str(action)))
        b._block(level, "out_block", model.out_block)

        return b._graph(root)


class AutoencoderAdapter:
    """Adapter for `Autoencoder` and its `VAE`/`VQVAE`/`DCAE` subclasses."""

    def matches(self, model: nn.Module) -> bool:
        """Match any `Autoencoder` subclass.

        Args:
            model: Root module to test.
        """
        return isinstance(model, _classes().Autoencoder)

    def build(
        self,
        model: nn.Module,
        info_by_id: dict[int, LayerInfo],
        dataflow: dict[int, set[int]] | None = None,
    ) -> IRGraph:
        """Decompose an autoencoder into encoder/latent/decoder (no skips).

        Args:
            model: Autoencoder model.
            info_by_id: Map from id(module) to LayerInfo.
            dataflow: Unused; the autoencoder forward flow is built explicitly.
        """
        del dataflow
        c = _classes()
        b = _Builder(model, info_by_id)
        root = b._root()

        enc = b._child(
            root, "encoder", NodeRole.COMPONENT, "Encoder", module=model.encoder
        )
        self._sequential_component(b, c, enc, model.encoder, down=True)

        kind = (
            "reparameterization"
            if isinstance(model, c.VAE)
            else "codebook"
            if isinstance(model, c.VQVAE)
            else "projection"
        )
        latent = b._child(root, "latent", NodeRole.COMPONENT, "Latent")
        latent.meta["latent_kind"] = kind
        llv = b._level(latent, 0)
        if getattr(model, "latent_proj", None) is not None:
            b._block(llv, "latent_proj", model.latent_proj)
        if isinstance(model, c.VAE):
            # Synthetic op (no module): thread it into the forward flow between
            # the latent projection and deprojection instead of leaving it
            # disconnected (`_chain_forward` links consecutive `_leaves`).
            reparam = b._child(
                llv, "reparameterize", NodeRole.BLOCK, "Reparameterize\n(μ, σ)"
            )
            b._leaves.append(reparam)
        if isinstance(model, c.VQVAE) and getattr(model, "quantize", None) is not None:
            b._block(llv, "quantize", model.quantize)
        if getattr(model, "latent_deproj", None) is not None:
            b._block(llv, "latent_deproj", model.latent_deproj)

        dec = b._child(
            root, "decoder", NodeRole.COMPONENT, "Decoder", module=model.decoder
        )
        self._sequential_component(b, c, dec, model.decoder, down=False)

        return b._graph(root)

    def _sequential_component(
        self,
        b: _Builder,
        c: SimpleNamespace,
        component: IRNode,
        sub: nn.Module,
        down: bool,
    ) -> None:
        """Populate an encoder/decoder component from its stage list."""
        stages = sub.down_blocks if down else sub.up_blocks
        boundary = c.DOWNSAMPLE if down else c.UPSAMPLE
        idx = 0
        level = b._level(component, idx)
        if down and hasattr(sub, "conv_in"):
            b._block(level, "conv_in", sub.conv_in, level_index=idx)
        if not down and hasattr(sub, "in_block"):
            b._block(level, "in_block", sub.in_block, level_index=idx)
        if not down:
            for j, m in enumerate(getattr(sub, "mid_blocks", [])):
                b._block(level, f"mid{j}", m, level_index=idx)
        bi = 0
        for m in stages:
            if isinstance(m, boundary):
                b._block(
                    level,
                    f"sample{idx}",
                    m,
                    level_index=idx,
                    is_down=down,
                    is_up=not down,
                )
                idx += 1
                level = b._level(component, idx)
                continue
            for stage_block in m:
                b._block(level, f"block{bi}", stage_block, level_index=idx)
                bi += 1
        if down:
            for j, m in enumerate(getattr(sub, "mid_blocks", [])):
                b._block(level, f"mid{j}", m, level_index=idx)
        cap = sub.out_block if hasattr(sub, "out_block") else None
        if cap is not None:
            b._block(level, "out_block", cap, level_index=idx)


class SequentialAdapter:
    """Adapter for `nn.Sequential`-based models such as discriminators."""

    def matches(self, model: nn.Module) -> bool:
        """Match sequential models.

        Args:
            model: Root module to test.
        """
        return isinstance(model, nn.Sequential)

    def build(
        self,
        model: nn.Module,
        info_by_id: dict[int, LayerInfo],
        dataflow: dict[int, set[int]] | None = None,
    ) -> IRGraph:
        """Decompose a sequential model into a single-level block chain.

        Args:
            model: Sequential model.
            info_by_id: Map from id(module) to LayerInfo.
            dataflow: Unused; a sequential model's flow is its block order.
        """
        del dataflow
        b = _Builder(model, info_by_id)
        root = b._root()
        component = b._child(root, "network", NodeRole.COMPONENT, "Network")
        level = b._level(component, 0)
        for i, (name, child) in enumerate(model.named_children()):
            b._block(level, name or f"block{i}", child, level_index=0)
        return b._graph(root)


class GenericAdapter:
    """Fallback adapter using LayerInfo depth banding for any model."""

    uses_dataflow = True

    def matches(self, model: nn.Module) -> bool:
        """Match everything.

        Args:
            model: Root module (unused).
        """
        del model
        return True

    def build(
        self,
        model: nn.Module,
        info_by_id: dict[int, LayerInfo],
        dataflow: dict[int, set[int]] | None = None,
    ) -> IRGraph:
        """Decompose an arbitrary module by its `named_children` hierarchy.

        Roles are assigned by module-tree depth (leaves are layers); LayerInfo
        supplies shapes and parameter counts. The real module tree is used
        rather than `LayerInfo.children` (which links transitive descendants
        and would duplicate nodes).

        Forward edges come from a traced tensor `dataflow` when available, so
        branched models keep their real parallel paths and merges; without a
        trace the builder falls back to a best-effort sequential chain.

        Args:
            model: Any model.
            info_by_id: Map from id(module) to LayerInfo.
            dataflow: Traced leaf-module dataflow (see `trace_dataflow`), or None.
        """
        residual = _classes().RESIDUAL
        b = _Builder(model, info_by_id)
        roles = [NodeRole.MODEL, NodeRole.COMPONENT, NodeRole.LEVEL, NodeRole.BLOCK]

        def rec(
            module: nn.Module, parent: IRNode | None, name: str, depth: int
        ) -> IRNode:
            if parent is None:
                node = b._root()
            else:
                leaf = not any(module.children())
                role = NodeRole.LAYER if leaf else roles[min(depth, len(roles) - 1)]
                node = b._child(
                    parent, name, role, module.__class__.__name__, module=module
                )
                if leaf:
                    b._leaves.append(node)
            for child_name, child in module.named_children():
                rec(child, node, child_name, depth + 1)
            if isinstance(module, residual) and len(node.children) >= 2:
                b.edges.append(
                    IREdge(node.children[0].id, node.children[-1].id, EdgeKind.RESIDUAL)
                )
            return node

        root = rec(model, None, "", 0)
        return b._graph(root, dataflow=dataflow)


class AdapterRegistry:
    """Ordered registry resolving the first matching adapter."""

    def __init__(self, adapters: list[SemanticAdapter] | None = None) -> None:
        """Constructor.

        Args:
            adapters: Ordered adapters (first match wins); defaults are used if None.
        """
        self.adapters: list[SemanticAdapter] = adapters or []

    def register(self, adapter: SemanticAdapter, priority: int = -1) -> None:
        """Insert an adapter.

        Args:
            adapter: Adapter to register.
            priority: Insertion index; appended before the fallback if -1.
        """
        if priority < 0:
            self.adapters.insert(max(len(self.adapters) - 1, 0), adapter)
        else:
            self.adapters.insert(priority, adapter)

    def resolve(self, model: nn.Module) -> SemanticAdapter:
        """Return the first adapter matching `model`.

        Args:
            model: Model to resolve an adapter for.
        """
        for adapter in self.adapters:
            if adapter.matches(model):
                return adapter
        raise RuntimeError("No adapter matched (GenericAdapter should always match).")


def default_registry() -> AdapterRegistry:
    """Build the default adapter registry."""
    return AdapterRegistry(
        [UNetAdapter(), AutoencoderAdapter(), SequentialAdapter(), GenericAdapter()]
    )
