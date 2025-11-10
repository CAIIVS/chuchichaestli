# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dim-to-Layer maps for different spatial dimensions."""

from torch import nn


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
