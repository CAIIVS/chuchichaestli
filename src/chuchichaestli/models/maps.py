# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dim-to-Layer maps for different spatial dimensions."""

from torch import nn


def require_cls(
    name: str,
    registry: dict[str, type[nn.Module]],
    context: str = "",
) -> type:
    """Look up a class in a registry, raising if the name is not registered.

    Args:
        name: Name of the class to look up.
        registry: Classes accepted at this position.
        context: Description of the position, used in the error message.

    Raises:
        ValueError: If `name` is not one of the registered classes.
    """
    if name not in registry:
        what = f" {context}" if context else ""
        raise ValueError(f"Unsupported{what}: {name!r}. Use one of {sorted(registry)}.")
    return registry[name]


DIM_TO_CONV_MAP = {
    1: nn.Conv1d,
    2: nn.Conv2d,
    3: nn.Conv3d,
}

DIM_TO_CONV_FN_MAP = {
    1: nn.functional.conv1d,
    2: nn.functional.conv2d,
    3: nn.functional.conv3d,
}

DIM_TO_CONVT_MAP = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d,
}

DIM_TO_POOL_MAP = {
    1: {
        "MaxPool": nn.MaxPool1d,
        "AvgPool": nn.AvgPool1d,
        "AdaptiveMaxPool": nn.AdaptiveMaxPool1d,
        "AdaptiveAvgPool": nn.AdaptiveAvgPool1d,
    },
    2: {
        "MaxPool": nn.MaxPool2d,
        "AvgPool": nn.AvgPool2d,
        "AdaptiveMaxPool": nn.AdaptiveMaxPool2d,
        "AdaptiveAvgPool": nn.AdaptiveAvgPool2d,
    },
    3: {
        "MaxPool": nn.MaxPool3d,
        "AvgPool": nn.AvgPool3d,
        "AdaptiveMaxPool": nn.AdaptiveMaxPool3d,
        "AdaptiveAvgPool": nn.AdaptiveAvgPool3d,
    },
}

UPSAMPLE_MODE = {
    1: "linear",
    2: "bilinear",
    3: "trilinear",
}

DOWNSAMPLE_MODE = UPSAMPLE_MODE
