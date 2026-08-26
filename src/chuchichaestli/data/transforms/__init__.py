# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""N-dimensional data transforms built on torchvision transforms v2."""

from chuchichaestli.data.transforms.crop import RandomCropND, CenterCropND
from chuchichaestli.data.transforms.geometric import RandomFlipND, RandomRot90ND
from chuchichaestli.data.transforms.resize import ResizeND, PadND
from chuchichaestli.data.transforms.intensity import (
    Affine,
    HEPTransform,
    InvHEPTransform,
    LogTransform,
    InvLogTransform,
    LogP1Transform,
    InvLogP1Transform,
    MinMaxScale,
    InvMinMaxScale,
    Clamp,
    HounsfieldScale,
    InvHounsfieldScale,
    HounsfieldClamp,
    ZScaleInterval,
    ZScale,
)
from chuchichaestli.data.transforms.channel import ChannelExpand, ChannelCollapse
from chuchichaestli.data.transforms.basis import (
    BasisProjection,
    InvBasisProjection,
    BASIS_REGISTRY,
)
from chuchichaestli.data.transforms.compose import SequentialTransform

__all__ = [
    "RandomCropND",
    "CenterCropND",
    "RandomFlipND",
    "RandomRot90ND",
    "ResizeND",
    "PadND",
    "Affine",
    "HEPTransform",
    "InvHEPTransform",
    "LogTransform",
    "InvLogTransform",
    "LogP1Transform",
    "InvLogP1Transform",
    "MinMaxScale",
    "InvMinMaxScale",
    "Clamp",
    "HounsfieldScale",
    "InvHounsfieldScale",
    "HounsfieldClamp",
    "ZScaleInterval",
    "ZScale",
    "ChannelExpand",
    "ChannelCollapse",
    "BasisProjection",
    "InvBasisProjection",
    "BASIS_REGISTRY",
    "SequentialTransform",
]
