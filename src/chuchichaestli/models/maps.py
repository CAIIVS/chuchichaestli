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

UPSAMPLE_MODE = {
    1: "linear",
    2: "bilinear",
    3: "trilinear",
}

DOWNSAMPLE_MODE = UPSAMPLE_MODE
