# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
# Partly adapted from torchinfo
"""PyTorch module operations for chuchichaestli."""

from __future__ import annotations
from enum import Enum
import numpy as np
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle
from torch.jit import ScriptModule
from typing import Any, Literal
from collections.abc import Sequence, Iterable, Callable
from chuchichaestli.utils import nested_list_size, prod, map_nested

try:
    from torch.nn.parameter import is_lazy
except ImportError:

    def is_lazy(param: nn.Parameter) -> bool:  # type: ignore[misc]
        del param
        return False


__all__ = [
    "info_forward_pass",
    "layer_info",
    "clear_info_cache",
    "get_chuchichaestli_block_type",
    "get_layer_type",
]


IO_TENSORS_TYPES = Sequence[torch.Tensor] | dict[Any, torch.Tensor] | torch.Tensor
IO_SHAPE_TYPES = Sequence[int | Sequence[Any] | torch.Size]
MODULE_MODES = Literal["train", "eval", "same"]
_cached_info_forward_pass: dict[str, list[LayerInfo]] = {}


class DEFAULT_MODULE_LABELS(Enum):
    C3LI_UNET_ENCODER = "U-Net Encoder"
    C3LI_UNET_DECODER = "U-Net Decoder"
    C3LI_UNET_DOWNBLOCK = "U-Net Down-Block"
    C3LI_UNET_MIDBLOCK = "U-Net Mid-Block"
    C3LI_UNET_UPBLOCK = "U-Net Up-Block"
    C3LI_UNET_DOWNSAMP = "U-Net Downsampling Block"
    C3LI_UNET_UPSAMP = "U-Net Upsampling Block"
    C3LI_UNET_ATTNDOWNBLOCK = "U-Net Attention Down-Block"
    C3LI_UNET_ATTNMIDBLOCK = "U-Net Attention Mid-Block"
    C3LI_UNET_ATTNUPBLOCK = "U-Net Attention Up-Block"
    C3LI_UNET_GATEDATTNBLOCK = "U-Net Gated Attention Block"
    C3LI_GAN_CONVBLOCK = "PatchGAN Block"
    C3LI_GAN_ATTNBLOCK = "PatchGAN Attention Block"
    C3LI_VAE_DOWNBLOCK = "VAE Down-Block"
    C3LI_VAE_MIDBLOCK = "VAE Bottleneck"
    C3LI_VAE_UPBLOCK = "VAE Up-Block"
    C3LI_RESBLOCK = "Residual Block"
    C3LI_SELFATTN = "Self-Attention"
    C3LI_CONVATTN = "Conv-Attention"
    C3LI_NOISEBLOCK = "Noise Block"
    C3LI_TIME_EMB = "Time Embedding"
    C3LI_DISCRIMINATOR = "Discriminator"
    CONV = "Conv"
    UPSAMP = "Upsample"
    DOWNSAMP = "Downsample"
    LIN = "Linear"
    NORM = "Norm"
    ACT = "Activation"
    DROP = "Dropout"
    ATTN = "Attention"
    EMB = "Embedding"
    RECUR = "Recurrent"
    PAD = "Padding"
    SPARSE = "Sparse"
    SHUFFLE = "Shuffle"
    FLAT = "Flatten"
    IDENT = "Identity"
    LOSS = "Loss"
    DIST = "Distance"
    STRUCT = "Container"
    UNKN = "Unknown"


class InfoSettings(str, Enum):
    """Enum containing all available parameter for LayerInfo."""

    __slots__ = ()

    INPUT_SIZE = "input_size"
    OUTPUT_SIZE = "output_size"
    KERNEL_SIZE = "kernel_size"
    NUM_PARAMS = "num_params"
    GROUPS = "groups"
    MULT_ADDS = "mult_adds"
    TRAINABLE = "trainable"


class LayerInfo:
    """Graph node holding information about a PyTorch module.

    Information includes
        - layer names
        - input/output shapes
        - number of parameters
        - number of mult-add operations
        - kernel shape
        - whether layer is trainable
    """

    def __init__(
        self,
        name: str,
        module: nn.Module,
        depth: int,
        parent_info: LayerInfo | None = None,
    ) -> None:
        """Constructor.

        Args:
            name: Name of the module.
            module: Module root instance.
            depth: Maximum depth for graph recursion.
            parent_info: Reference to parent node (if not root).
        """
        # graph parameters
        self.layer_id = id(module)
        self.module = module
        self.class_name = (
            str(module.original_name)
            if isinstance(module, ScriptModule)
            else module.__class__.__name__
        )
        self.inner_layers: dict[str, dict[InfoSettings, Any]] = {}
        self.depth = depth
        self.depth_index: int | None = None  # set at the very end
        self.children: list[LayerInfo] = []  # set at the very end
        self.parent_info = parent_info
        if name.isdigit():
            name = f"{module.__class__.__name__.lower()}.{name}"
        self.name = name
        self.is_leaf_layer = not any(self.module.children())
        self.contains_lazy_param = False
        self._visited = False

        # info dimensions
        self.is_recursive = False
        self.input_size: list[int] = []
        self.output_size: list[int] = []
        self.kernel_size = self.get_kernel_size(module)
        self.groups = self.get_groups(module)
        self.trainable_params = 0
        self.num_params = 0
        self.param_bytes = 0
        self.output_bytes = 0
        self.mult_adds = 0

    def __repr__(self) -> str:
        """LayerInfo representation."""
        return f"<{self.class_name}: {self.depth}>"

    def __call__(self, indent: int = 24) -> str:
        """Instance call gives summary of the layer."""
        return "".join(
            [
                f"{self.class_name}\n",
                f"  Name:        {self.name:>{indent}}\n",
                f"  LayerID:     {self.layer_id:>{indent}}\n",
                f"  Depth:       {self.depth:>{indent}}\n",
                f"  Input size:  {str(self.input_size):>{indent}}\n",
                f"  Output size: {str(self.output_size):>{indent}}\n",
                f"  # Params:    {self.num_params:>{indent},}\n",
                f"  # Local:     {self.local_params():>{indent},}\n",
                f"  # Mult-Adds: {self.mult_adds:>{indent},}\n",
                f"  Kernel size: {str(self.kernel_size):>{indent}}\n"
                if self.kernel_size
                else "",
                f"  Groups:      {str(self.groups):>{indent}}\n" if self.groups else "",
                f"  # Children:  {len(self.children):>{indent}}\n",
                f"  Trainable:   {self.trainable:>{indent}}\n",
            ]
        )

    @property
    def trainable(self) -> str:
        """Checks if the module is trainable.

        Returns:
          `"True"` if all the parameters are trainable (`requires_grad=True`).
          `"False"` if none of the parameters are trainable.
          `"Partial"` if some weights are trainable, but not all.
          `"None"` if module has no parameters, like Dropout.
        """
        if self.num_params == 0:
            return "None"
        if self.trainable_params == 0:
            return "False"
        if self.num_params == self.trainable_params:
            return "True"
        if self.num_params > self.trainable_params:
            return "Partial"
        raise RuntimeError("Unreachable trainable calculation.")

    @staticmethod
    def calculate_size(
        inputs: IO_TENSORS_TYPES | None,
        batch_dim: int | None = None,
    ) -> tuple[list[int], int]:
        """Set `input_size` or `output_size` using the model's inputs.

        Args:
            inputs: Input tensor(s).
            batch_dim: Batch dimension in the input.

        Returns:
            - Corrected shape of `inputs`.
            - Size of a single element in bytes.
        """
        if inputs is None:
            size, elem_bytes = [], 0
        # pack_padded_seq and pad_packed_seq store feature into data attribute
        elif (
            isinstance(inputs, (list, tuple))
            and inputs
            and hasattr(inputs[0], "data")
            and hasattr(inputs[0].data, "size")
        ):
            size = list(inputs[0].data.size())
            elem_bytes = inputs[0].data.element_size()
            if batch_dim is not None:
                size = size[:batch_dim] + [1] + size[batch_dim + 1 :]
        elif isinstance(inputs, dict):
            output = list(inputs.values())[-1]
            size, elem_bytes = nested_list_size(output)
            if batch_dim is not None:
                size = [size[:batch_dim] + [1] + size[batch_dim + 1 :]]
        elif isinstance(inputs, torch.Tensor):
            size = list(inputs.size())
            elem_bytes = inputs.element_size()
        elif isinstance(inputs, np.ndarray):  # type: ignore[unreachable]
            inputs_ = torch.from_numpy(inputs)  # type: ignore[unreachable]
            size, elem_bytes = list(inputs_.size()), inputs_.element_size()
        elif isinstance(inputs, (list, tuple)):
            size, elem_bytes = nested_list_size(inputs)
            if batch_dim is not None and batch_dim < len(size):
                size[batch_dim] = 1
        else:
            raise TypeError(
                "Model contains a layer with an unsupported input or output type: "
                f"{inputs}, type: {type(inputs)}"
            )
        return size, elem_bytes

    @staticmethod
    def get_param_count(
        module: nn.Module, name: str, param: torch.Tensor
    ) -> tuple[int, str]:
        """Get count of number of params, accounting for mask.

        Masked models save parameters with the suffix "_orig" added.
        They have a buffer ending with "_mask" which has only 0s and 1s.
        If a mask exists, the sum of 1s in mask is number of params.

        Args:
            module: Module from which to get info.
            name: Module parameter name.
            param: Module parameters.

        Returns:
            - Parameter count.
            - Parameter name (without suffix).
        """
        if name.endswith("_orig"):
            without_suffix = name[:-5]
            pruned_weights = _rgetattr(module, f"{without_suffix}_mask")
            if pruned_weights is not None:
                parameter_count = int(torch.sum(pruned_weights))
                return parameter_count, without_suffix
        return param.nelement(), name

    @staticmethod
    def get_kernel_size(module: nn.Module) -> int | list[int] | None:
        """Kernel size of a module (if any).

        Args:
            module: Module from which to get info.
        """
        if hasattr(module, "kernel_size"):
            k = module.kernel_size
            kernel_size: int | list[int]
            if isinstance(k, Iterable):
                kernel_size = list(k)
            elif isinstance(k, int):
                kernel_size = int(k)
            else:
                raise TypeError(f"kernel_size has an unexpected type: {type(k)}")
            return kernel_size
        return None

    @staticmethod
    def get_groups(module: nn.Module) -> int | None:
        """Group size of a module (if any).

        Args:
            module: Module from which to get info.
        """
        if hasattr(module, "groups"):
            return int(module.groups)
        return None

    def get_layer_name(self, show_name: bool = False, show_depth: bool = False) -> str:
        """Layer name of the module.

        Args:
            show_name: If `True`, add label to layer name.
            show_depth: If `True`, add depth to layer name.
        """
        layer_name = self.class_name
        if show_name and self.name:
            layer_name += f" ({self.name})"
        if show_depth and self.depth > 0:
            layer_name += f": {self.depth}"
            if self.depth_index is not None:
                layer_name += f"-{self.depth_index}"
        return layer_name

    def get_traced_label(self) -> str:
        """Get the label name of the module including the full path in the graph."""
        labels = [self.name]
        parent_info = self.parent_info
        while parent_info:
            labels.append(parent_info.label)
            parent_info = parent_info.parent_info
        return ".".join(reversed(labels))
            

    def calculate_num_params(self):
        """Set `num_params`,`trainable`,`inner_layers`,`kernel_size` using the parameters."""
        self.num_params = 0
        self.param_bytes = 0
        self.trainable_params = 0
        self.inner_layers = {}

        final_name = ""
        for name, param in self.module.named_parameters():
            if is_lazy(param):
                self.contains_lazy_param = True
                continue
            cur_params, name = self.get_param_count(self.module, name, param)
            self.param_bytes += param.element_size() * cur_params

            self.num_params += cur_params
            if param.requires_grad:
                self.trainable_params += cur_params

            # kernel_size for inner layer parameters
            ksize = list(param.size())
            # to make [in_shape, out_shape, ksize, ksize]
            if name == "weight" and len(ksize) > 1:
                ksize[0], ksize[1] = ksize[1], ksize[0]

            # RNN modules have inner weights such as weight_ih_l0
            if self.parent_info is not None or "." not in name:
                self.inner_layers[name] = {
                    InfoSettings.KERNEL_SIZE: str(ksize),
                    InfoSettings.NUM_PARAMS: str(cur_params),
                }
                final_name = name
        if self.inner_layers:
            self.inner_layers[final_name][InfoSettings.NUM_PARAMS] = self.inner_layers[
                final_name
            ][InfoSettings.NUM_PARAMS]

    def calculate_mult_adds(self):
        """Set mult-add operations using the module's parameters and layer's output size.

        Note:
            Mult-adds numbers are for the full tensor, including the batch-dimension.
        """
        for name, param in self.module.named_parameters():
            cur_params, name = self.get_param_count(self.module, name, param)
            if name in ("weight", "bias"):
                # ignore C when calculating Mult-Adds in ConvNd
                if "Conv" in self.class_name:
                    self.mult_adds += int(
                        cur_params * prod(self.output_size[:1] + self.output_size[2:])
                    )
                elif "Linear" in self.class_name:
                    self.mult_adds += int(cur_params * prod(self.output_size[:-1]))
                else:
                    self.mult_adds += self.output_size[0] * cur_params
            # RNN modules have inner weights such as weight_ih_l0
            elif "weight" in name or "bias" in name:
                self.mult_adds += prod(self.output_size[:2]) * cur_params

    def check_recursive(self, layer_ids: set[int]):
        """If the current module has already been traversed, mark as recursive.

        Args:
           layer_ids: Set of traversed layer IDs.
        """
        if self.layer_id in layer_ids:
            self.is_recursive = True

    def local_params(self) -> int:
        """Local parameter count (of this layer), excluding children counts."""
        return self.num_params - sum(
            child.num_params if child.is_leaf_layer else child.local_params()
            for child in self.children
            if not child.is_recursive
        )

    def local_trainable_params(self) -> int:
        """Local trainable parameter count (of this layer), excluding chilren counts."""
        return self.trainable_params - sum(
            child.trainable_params
            if child.is_leaf_layer
            else child.local_trainable_params()
            for child in self.children
            if not child.is_recursive
        )


def _rgetattr(module: nn.Module, attr: str) -> torch.Tensor | None:
    """Get the tensor `attr` submodule from module.

    Args:
        module: Module from which to retrieve given `attr`.
        attr: Submodule tensor.
    """
    for iattr in attr.split("."):
        if not hasattr(module, iattr):
            return None
        module = getattr(module, iattr)
    assert isinstance(module, torch.Tensor)
    return module


def _get_children_layerinfo(info_list: list[LayerInfo], index: int) -> list[LayerInfo]:
    """Fetches all of the children's info of a given layer.

    Args:
        info_list: List of layer info instances (generated via `info_forward_pass`).
        index: Layer location index in `info_list`.
    """
    num_children = 0
    for layer in info_list[index + 1 :]:
        if layer.depth <= info_list[index].depth:
            break
        num_children += 1
    return info_list[index + 1 : index + 1 + num_children]


def _set_children_layerinfo(info_list: list[LayerInfo]):
    """Populate the children and depth index fields of all LayerInfo instances.

    Args:
        info_list: List of layer info instances (generated via `info_forward_pass`).
    """
    idx: dict[int, int] = {}
    for i, layerinfo in enumerate(info_list):
        idx[layerinfo.depth] = idx.get(layerinfo.depth, 0) + 1
        layerinfo.depth_index = idx[layerinfo.depth]
        layerinfo.children = _get_children_layerinfo(info_list, i)


def info_forward_pass(
    model: nn.Module,
    input_data: IO_TENSORS_TYPES | None = None,
    input_shape: IO_SHAPE_TYPES | None = None,
    size: int | None = None,
    input_dtype: torch.dtype = torch.float32,
    batch_dim: int | None = None,
    use_cache: bool = True,
    device: torch.device | None = None,
    mode: MODULE_MODES = "same",
    **kwargs,
) -> list[LayerInfo]:
    """Perform a forward pass on the model using info hooks.

    Args:
        model: Module from which to get info (recursively).
        input_data: Input for the forward pass (if `None`, `input_shape` has to be specified).
        input_shape: Alternative to `input_data` (only takes effect if `input_data == None`).
        size: Yet another alternative to `input_data` and `input_shape`
          (only takes effect if `input_data == None` and `input_shape == None`).
        input_dtype: Data type (only takes effect if `input_data == None`).
        batch_dim: Batch dimension (if specified, input is expanded at specified dimension).
        use_cache: If `True`, use global info cache.
        device: Device for computation; if `None`, the current device(s) is/are used.
        mode: Mode of the model in which to do a forward pass; one of ["train", "eval", "same"].
        kwargs: Additional keyword arguments to be passed to the model.
    """
    global _cached_info_forward_pass
    model_name = model.__class__.__name__
    if use_cache and model_name in _cached_info_forward_pass:
        return _cached_info_forward_pass[model_name]

    # construct test input
    if input_data is not None:
        input_shape = map_nested(
            input_data, action_fn=lambda data: data.size(), aggregate_fn=type
        )
        x = input_data if device is None else input_data.to(device)
        if isinstance(x, (torch.Tensor, np.ndarray)):
            x = [x]
    elif input_shape is not None:
        x = []
        if (
            isinstance(input_shape, tuple | list | torch.Size)
            and isinstance(input_shape[0], int)
        ) or (
            isinstance(input_shape, tuple | list | torch.Size)
            and isinstance(input_shape[0], int)
        ):
            input_shape = [input_shape]
        for s in input_shape:
            input_tensor = torch.rand(*s)
            if batch_dim is not None:
                input_tensor = input_tensor.unsqueeze(dim=batch_dim)
            input_tensor = input_tensor if device is None else input_tensor.to(device)
            input_tensor = input_tensor.type(input_dtype)
            x.append(input_tensor)
    elif size is not None:
        x = []
        input_shape = [_infer_input_shape(model, init_size=size, dtype=input_dtype)]
        for s in input_shape:
            input_tensor = torch.rand(*s)
            if batch_dim is not None:
                input_tensor = input_tensor.unsqueeze(dim=batch_dim)
            input_tensor = input_tensor if device is None else input_tensor.to(device)
            input_tensor = input_tensor.type(input_dtype)
            x.append(input_tensor)
    else:
        x = None

    # apply info graph hooks
    info_list, _, hooks = _apply_info_hooks(model_name, model, x, batch_dim)
    if x is None:
        _set_children_layerinfo(info_list)
        return info_list

    # forward pass
    original_model_mode = model.training
    try:
        if mode == "train":
            model.train()
        elif mode == "eval":
            model.eval()
        elif mode != "same":
            raise ValueError(
                f"Specified model mode ({list(MODULE_MODES)}) not recognized: {mode}"
            )
        with torch.no_grad():
            model = model if device is None else model.to(device)
            if isinstance(x, list | tuple):
                _ = model(*x, **kwargs)
            elif isinstance(x, dict):
                _ = model(**x, **kwargs)
            else:
                raise ValueError(f"Unknown input type: {type(x)}")
    except Exception as e:
        visited_layers = [layer for layer in info_list if layer._visited]
        raise RuntimeError(
            "Failed `info_forward_pass`. See above stack traces for more details. "
            f"Executed layers up to: {visited_layers}"
        ) from e
    finally:  # remove hooks again
        if hooks:
            for pre_hook, hook in hooks.values():
                pre_hook.remove()
                hook.remove()
        model.train(original_model_mode)

    # deal with skipped container submodules such as ModuleList, Sequential, etc.
    _add_missing_container_layers(info_list)
    _set_children_layerinfo(info_list)

    _cached_info_forward_pass[model_name] = info_list
    return info_list


def layer_info(
    model: nn.Module,
    input_size: int = 8,
    input_dtype: torch.dtype = torch.float32,
    use_cache: bool = True,
    **kwargs,
) -> list[LayerInfo]:
    """Perform a forward pass on the model using info hooks.

    Args:
        model: Module from which to get info (recursively).
        input_size: Minimum size of the (spatial) dimension of the inferred input tensor.
        input_dtype: Data type for model input.
        use_cache: If `True`, use global info cache.
        kwargs: Additional keyword arguments to be passed to the model.
    """
    input_shape = [_infer_input_shape(model, init_size=input_size, dtype=input_dtype)]
    device = next(model.parameters()).device
    return info_forward_pass(
        model, input_shape=input_shape, use_cache=use_cache, device=device, mode="same"
    )


def _create_info_pre_hook(
    global_layer_info: dict[int, LayerInfo],
    info_list: list[LayerInfo],
    layer_ids: set[int],
    label: str,
    curr_depth: int,
    parent_info: LayerInfo | None,
) -> Callable[[nn.Module, Any], None]:
    """Construct an info initialization hook function."""

    def pre_hook(module: nn.Module, inputs: Any) -> None:
        """Create a LayerInfo object to aggregate layer information."""
        del inputs
        info = LayerInfo(label, module, curr_depth, parent_info)
        info.calculate_num_params()
        info.check_recursive(layer_ids)
        info_list.append(info)
        layer_ids.add(info.layer_id)
        global_layer_info[info.layer_id] = info

    return pre_hook


def _create_info_hook(
    global_layer_info: dict[int, LayerInfo],
    batch_dim: int | None = None,
) -> Callable[[nn.Module, Any, Any], None]:
    """Construct an info hook function."""

    def hook(module: nn.Module, inputs: Any, outputs: Any) -> None:
        """Update LayerInfo after forward pass."""
        info = global_layer_info[id(module)]
        if info.contains_lazy_param:
            info.calculate_num_params()
        info.input_size, _ = info.calculate_size(inputs, batch_dim)
        info.output_size, elem_bytes = info.calculate_size(outputs, batch_dim)
        info.output_bytes = elem_bytes * prod(info.output_size)
        info.visited = True
        info.calculate_mult_adds()

    return hook


def _apply_info_hooks(
    label: str,
    module: nn.Module,
    input_data: IO_TENSORS_TYPES | None = None,
    batch_dim: int | None = None,
) -> tuple[
    list[LayerInfo],
    dict[int, LayerInfo],
    dict[int, tuple[RemovableHandle, RemovableHandle]],
]:
    """Apply info hooks recursively (using a forward pass if `input_data` is provided).

    Args:
        label: Name of the model.
        module: Module root instance, i.e. model.
        input_data: Input for the model.
        batch_dim: Batch dimension of the model input.
    """
    info_list: list[LayerInfo] = []
    layer_ids: set[int] = set()
    global_layer_info: dict[int, LayerInfo] = {}
    hooks: dict[int, tuple[RemovableHandle, RemovableHandle]] = {}
    stack: list[tuple[str, nn.Module, int, LayerInfo | None]] = [
        (label, module, 0, None)
    ]
    while stack:
        module_name, module, curr_depth, parent_info = stack.pop()
        module_id = id(module)

        # Fallback is used if the layer's pre-hook is never called, for example in
        # ModuleLists or Sequentials.
        global_layer_info[module_id] = LayerInfo(
            module_name, module, curr_depth, parent_info
        )
        pre_hook = _create_info_pre_hook(
            global_layer_info,
            info_list,
            layer_ids,
            module_name,
            curr_depth,
            parent_info,
        )
        if input_data is None or isinstance(module, ScriptModule):
            pre_hook(module, None)
        else:
            # Register the hook using the last layer that uses this module.
            if module_id in hooks:
                for hook in hooks[module_id]:
                    hook.remove()
            hooks[module_id] = (
                module.register_forward_pre_hook(pre_hook),
                module.register_forward_hook(
                    _create_info_hook(global_layer_info, batch_dim)
                ),
            )

        # Note: module.named_modules(remove_duplicate=False) doesn't work for
        # some unknown reason (infinite recursion)
        stack += [
            (name, mod, curr_depth + 1, global_layer_info[module_id])
            for name, mod in reversed(module._modules.items())
            if mod is not None
        ]
    return info_list, global_layer_info, hooks


def _add_missing_container_layers(info_list: list[LayerInfo]):
    """Finds container modules such as `ModuleList`, `Sequential`, etc.

    Args:
        info_list: List of layer info instances (generated via `info_forward_pass`).
    """
    layer_ids = {layer.layer_id for layer in info_list}
    current_hierarchy: dict[int, LayerInfo] = {}
    for idx, layer_info in enumerate(info_list):
        # to keep track index of current layer after inserting new layers
        rel_idx = 0

        # create full hierarchy of current layer
        hierarchy = {}
        parent = layer_info.parent_info
        while parent is not None and parent.depth > 0:
            hierarchy[parent.depth] = parent
            parent = parent.parent_info

        # show hierarchy if it is not there already
        for d in range(1, layer_info.depth):
            if (
                d not in current_hierarchy
                or current_hierarchy[d].module is not hierarchy[d].module
            ) and hierarchy[d] is not info_list[idx + rel_idx - 1]:
                hierarchy[d].calculate_num_params()
                hierarchy[d].check_recursive(layer_ids)
                info_list.insert(idx + rel_idx, hierarchy[d])
                layer_ids.add(hierarchy[d].layer_id)

                current_hierarchy[d] = hierarchy[d]
                rel_idx += 1

        current_hierarchy[layer_info.depth] = layer_info

        # remove deeper hierarchy
        d = layer_info.depth + 1
        while d in current_hierarchy:
            current_hierarchy.pop(d)
            d += 1


def _infer_input_shape(
    model: nn.Module,
    init_size: int = 8,
    max_size: int = 1024,
    dtype: torch.dtype = torch.float32,
) -> IO_SHAPE_TYPES:
    """Dynamically try to guess the input shape for a PyTorch model.

    Args:
        model: Model for which to guess the input shape.
        init_size: Initial spatial size to try with the model.
        max_size: Maximal spatial size to try with the model.
        dtype: Data type for trials.
    """
    device = next(model.parameters()).device
    first_layer = next(model.children())

    if isinstance(first_layer, nn.Conv2d):
        in_channels = first_layer.in_channels
        for size in range(init_size, max_size + 1, init_size):
            x = torch.randn(1, in_channels, size, size, device=device).type(dtype)
            try:
                with torch.no_grad():
                    model(x)
                return x.shape
            except RuntimeError:
                continue
    elif isinstance(first_layer, nn.Conv3d):
        in_channels = first_layer.in_channels
        for size in range(init_size, max_size + 1, init_size):
            x = torch.randn(1, in_channels, size, size, size // 2, device=device).type(
                dtype
            )
            try:
                with torch.no_grad():
                    model(x)
                return x.shape
            except RuntimeError:
                continue
    elif isinstance(first_layer, nn.Linear):
        in_features = first_layer.in_features
        x = torch.randn(1, in_features, device=device)
        return x.shape
    elif isinstance(first_layer, nn.Embedding):
        x = torch.randint(0, first_layer.num_embeddings, (1, 16), device=device).type(
            dtype
        )
        return x.shape
    elif isinstance(first_layer, nn.LSTM | nn.GRU | nn.RNN):
        input_size = first_layer.input_size
        if getattr(first_layer, "batch_first", False):
            x = torch.randn(1, 16, input_size, device=device).type(dtype)
        else:
            x = torch.randn(16, 1, input_size, device=device).type(dtype)
        return x.shape
    # Fallback
    return ((1, 1), torch.float32)


def clear_info_cache():
    """Clear the info forward pass cache."""
    global _cached_info_forward_pass
    _cached_info_forward_pass = {}


def get_chuchichaestli_block_type(
    module: nn.Module, labels: Enum = DEFAULT_MODULE_LABELS
) -> str:
    """Chuchichaestli model block type labelling.

    Args:
        module: PyTorch module to group.
        labels: Enum mapping types (CONV, UPSAMP, LIN, etc.) to names.
    """
    cls_name = module.__class__.__name__

    # U-Net Encoder Blocks
    if cls_name in ["DownBlock", "AttnDownBlock", "ConvAttnDownBlock"]:
        if "Attn" in cls_name:
            return labels.C3LI_UNET_ATTNDOWNBLOCK.value
        return labels.C3LI_UNET_DOWNBLOCK.value

    # U-Net Mid Blocks
    if cls_name in ["MidBlock", "AttnMidBlock", "ConvAttnMidBlock"]:
        if "Attn" in cls_name:
            return labels.C3LI_UNET_ATTNMIDBLOCK.value
        return labels.C3LI_UNET_MIDBLOCK.value

    # U-Net Decoder Blocks
    if cls_name in ["UpBlock", "AttnUpBlock", "ConvAttnUpBlock", "AttnGateUpBlock"]:
        if "AttnGate" in cls_name:
            return labels.C3LI_UNET_GATEDATTNBLOCK.value
        elif "Attn" in cls_name:
            return labels.C3LI_UNET_ATTNUPBLOCK.value
        return labels.C3LI_UNET_UPBLOCK.value

    # U-Net Downsampling Blocks
    if cls_name in ["Downsample", "DownsampleInterpolate"]:
        return labels.C3LI_UNET_DOWNSAMP.value

    # U-Net Upsampling Blocks
    if cls_name in ["Upsample", "UpsampleInterpolate"]:
        return labels.C3LI_UNET_UPSAMP.value

    # Adversarial Conv Blocks
    if cls_name in [
        "ConvBlock",
        "ConvDownBlock",
        "ConvDownsampleBlock",
        "NormConvBlock",
        "NormConvDownBlock",
        "NormConvDownsampleBlock",
        "ActConvBlock",
        "ActConvDownBlock",
        "ActConvDownsampleBlock",
        "NormActConvBlock",
        "NormActConvDownBlock",
        "NormActConvDownsampleBlock",
    ]:
        return labels.C3LI_GAN_CONVBLOCK.value

    # Adversarial Attention Blocks
    if cls_name in [
        "AttnConvBlock",
        "AttnConvDownBlock",
        "AttnConvDownsampleBlock",
        "NormAttnConvBlock",
        "NormAttnConvDownBlock",
        "NormAttnConvDownsampleBlock",
        "ActAttnConvBlock",
        "ActAttnConvDownBlock",
        "ActAttnConvDownsampleBlock",
        "NormActAttnConvBlock",
        "NormActAttnConvDownBlock",
        "NormActAttnConvDownsampleBlock",
    ]:
        return labels.C3LI_GAN_ATTNBLOCK.value

    # Residual Block
    if cls_name == "ResidualBlock":
        return labels.C3LI_RESBLOCK.value

    # Attention Mechanisms
    if cls_name == "SelfAttention":
        return labels.C3LI_SELFATTN.value
    if cls_name == "ConvAttention":
        return labels.C3LI_CONVATTN.value

    # Special Blocks
    if cls_name == "GaussianNoiseBlock":
        return labels.C3LI_NOISEBLOCK.value

    # Time Embeddings
    if cls_name in ["SinusoidalTimeEmbedding", "DeepSinusoidalTimeEmbedding"]:
        return labels.C3LI_TIME_EMB.value

    # U-Net encoder
    if "Encoder" in cls_name:
        return labels.C3LI_UNET_ENCODER.value

    # U-Net decoder
    if "Decoder" in cls_name:
        return labels.C3LI_UNET_DECODER.value

    # Discriminators
    if "Discriminator" in cls_name:
        return labels.C3LI_DISCRIMINATOR.value

    return None


def get_layer_type(module: nn.Module, labels: Enum = DEFAULT_MODULE_LABELS) -> str:
    """PyTorch layer type labelling with comprehensive coverage.

    Args:
        module: PyTorch module to group.
        labels: Enum mapping types (CONV, UPSAMP, LIN, ) to names.

    Return:
        Layer type string (given by `labels`)
    """
    cls_name = module.__class__.__name__

    if any(conv in cls_name for conv in ["Conv1d", "Conv2d", "Conv3d"]):
        if "Transpose" in cls_name:
            return labels.UPSAMP
        return labels.CONV.value

    if any(
        up in cls_name
        for up in ["ConvTranspose", "Upsample", "Upsampling", "Interpolate"]
    ):
        return labels.UPSAMP.value

    if any(linear in cls_name for linear in ["Linear", "Dense", "Bilinear"]):
        return labels.LIN.value

    if any(
        norm in cls_name
        for norm in [
            "BatchNorm",
            "BatchNorm1d",
            "BatchNorm2d",
            "BatchNorm3d",
            "LayerNorm",
            "GroupNorm",
            "InstanceNorm",
            "InstanceNorm1d",
            "InstanceNorm2d",
            "InstanceNorm3d",
            "LocalResponseNorm",
            "SyncBatchNorm",
            "LazyBatchNorm",
            "LazyInstanceNorm",
            "Norm",  # C3LI norm wrapper
        ]
    ):
        return labels.NORM.value

    if any(
        drop in cls_name
        for drop in [
            "Dropout",
            "Dropout1d",
            "Dropout2d",
            "Dropout3d",
            "AlphaDropout",
            "FeatureAlphaDropout",
        ]
    ):
        return labels.DROP.value

    if any(
        pool in cls_name
        for pool in [
            "MaxPool",
            "MaxPool1d",
            "MaxPool2d",
            "MaxPool3d",
            "AvgPool",
            "AvgPool1d",
            "AvgPool2d",
            "AvgPool3d",
            "AdaptiveMaxPool",
            "AdaptiveMaxPool1d",
            "AdaptiveMaxPool2d",
            "AdaptiveMaxPool3d",
            "AdaptiveAvgPool",
            "AdaptiveAvgPool1d",
            "AdaptiveAvgPool2d",
            "AdaptiveAvgPool3d",
            "FractionalMaxPool2d",
            "FractionalMaxPool3d",
            "LPPool1d",
            "LPPool2d",
        ]
    ):
        return labels.DOWNSAMP.value

    if any(
        unpool in cls_name
        for unpool in ["MaxUnpool", "MaxUnpool1d", "MaxUnpool2d", "MaxUnpool3d"]
    ):
        return labels.UPSAMP.value

    # Activation Functions
    if any(
        act in cls_name
        for act in [
            "ReLU",
            "ReLU6",
            "LeakyReLU",
            "PReLU",
            "RReLU",
            "ELU",
            "CELU",
            "SELU",
            "GELU",
            "Sigmoid",
            "LogSigmoid",
            "Tanh",
            "Tanhshrink",
            "Hardtanh",
            "Hardsigmoid",
            "Hardswish",
            "Softmax",
            "Softmax2d",
            "LogSoftmax",
            "Softmin",
            "Softplus",
            "Softshrink",
            "Softsign",
            "Mish",
            "SiLU",
            "GLU",
            "Hardshrink",
            "Threshold",
        ]
    ):
        # Special case: MultiheadAttention is attention, not just activation
        if "Attention" in cls_name:
            return labels.ATTN.value
        return labels.ACT.value

    # Attention Mechanisms
    if any(
        attn in cls_name
        for attn in [
            "Attention",
            "MultiheadAttention",
            "SelfAttention",
            "CrossAttention",
        ]
    ):
        return labels.ATTN.value

    # Embedding Layers
    if any(emb in cls_name for emb in ["Embedding", "EmbeddingBag", "LazyEmbedding"]):
        return labels.EMB.value

    # Recurrent Layers
    if any(
        rnn in cls_name
        for rnn in [
            "RNN",
            "LSTM",
            "GRU",
            "RNNCell",
            "LSTMCell",
            "GRUCell",
            "LazyLSTM",
            "LazyGRU",
        ]
    ):
        return labels.RECUR.value

    # Transformer Layers
    if any(
        trans in cls_name
        for trans in [
            "Transformer",
            "TransformerEncoder",
            "TransformerDecoder",
            "TransformerEncoderLayer",
            "TransformerDecoderLayer",
        ]
    ):
        return labels.ATTN.value

    # Padding Layers
    if any(
        pad in cls_name
        for pad in [
            "Pad",
            "ReflectionPad",
            "ReplicationPad",
            "ZeroPad",
            "ConstantPad",
            "CircularPad",
        ]
    ):
        return labels.PAD.value

    # Loss Functions (sometimes included in models)
    if "Loss" in cls_name:
        return labels.LOSS.value

    # Vision-specific Layers
    if any(
        vision in cls_name
        for vision in ["PixelShuffle", "PixelUnshuffle", "ChannelShuffle"]
    ):
        return labels.SHUFFLE.value

    # Sparse Layers
    if "Sparse" in cls_name:
        return labels.SPARSE.value

    # Distance Functions
    if any(
        dist in cls_name
        for dist in ["PairwiseDistance", "CosineSimilarity", "Distance"]
    ):
        return labels.DIST.value

    # Sequential, ModuleList, ModuleDict (containers)
    if any(
        container in cls_name
        for container in [
            "Sequential",
            "ModuleList",
            "ModuleDict",
            "ParameterList",
            "ParameterDict",
        ]
    ):
        return labels.STRUCT.value

    # Flatten and Unflatten
    if any(flat in cls_name for flat in ["Flatten", "Unflatten"]):
        return labels.FLAT.value

    # Identity (pass-through)
    if "Identity" in cls_name:
        return labels.IDENT.value

    # Lazy layers (weight initialization deferred)
    if "Lazy" in cls_name:
        # Try to determine base type
        if "Conv" in cls_name:
            return labels.CONV.value
        elif "Linear" in cls_name:
            return labels.LIN.value
        elif "BatchNorm" in cls_name or "InstanceNorm" in cls_name:
            return labels.NORM.value
        return labels.UNKN.value

    # Default case
    return labels.UNKN.value


if __name__ == "__main__":
    from pprint import pprint
    from chuchichaestli.models.unet import UNet

    model = UNet(
        dimensions=2,
        in_channels=3,
        n_channels=64,
        out_channels=3,
        down_block_types=("DownBlock",) * 4,
        up_block_types=("UpBlock", "UpBlock", "AttnUpBlock", "AttnUpBlock"),
        block_out_channel_mults=(1, 2, 2, 4),
        res_act_fn="prelu",
        res_dropout=0.4,
        attn_n_heads=2,
        skip_connection_action="concat",
    )

    # info_list = layer_info(model, 32)
    info_list = info_forward_pass(model, input_shape=(1, 3, 32, 32))
    # pprint(info_list)
    info = info_list[0]
    print(info(18))
    # print(info.children)
    print(info_list)
