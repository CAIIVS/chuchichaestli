# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Data module of chuchichaestli."""

from chuchichaestli.data.base import (
    FileDataset,  # abstract
    CachingDataset,  # abstract
    IndexedSample,
    WithIndices,
    with_indices,
)
from chuchichaestli.data.zip import ZipDataset
from chuchichaestli.data.hdf5 import HDF5Dataset, ZipHDF5Dataset
from chuchichaestli.data.numpy import NumpyDataset, ZipNumpyDataset
from chuchichaestli.data.safetensors import SafetensorsDataset, ZipSafetensorsDataset
from chuchichaestli.data.image import ImageDataset, ZipImageDataset
from chuchichaestli.data.procedural import (
    ProceduralDataset,  # abstract
    HalfMoonsDataset,
    SpiralsDataset,
    CheckerboardDataset,
    RingsDataset,
    ConcentricSpheresDataset,
    GaussiansDataset,
    SwissRollDataset,
    generate_procedural_dataset,
)
from chuchichaestli.data.batching import (
    HierarchicalBatchSampler,
    HierarchicalFileBatchSampler,
)
from chuchichaestli.data.collate import (
    SequenceCollate,
    sequence_collate,
)
from chuchichaestli.data.transforms import (
    RandomCropND,
    CenterCropND,
    RandomFlipND,
    RandomRot90ND,
    ResizeND,
    PadND,
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
    ChannelExpand,
    ChannelCollapse,
    ComplexExpand,
    ComplexCollapse,
    BasisProjection,
    InvBasisProjection,
    SequentialTransform,
)

__all__ = [
    "FileDataset",
    "CachingDataset",
    "ZipDataset",
    "HDF5Dataset",
    "ZipHDF5Dataset",
    "NumpyDataset",
    "ZipNumpyDataset",
    "SafetensorsDataset",
    "ZipSafetensorsDataset",
    "ImageDataset",
    "ZipImageDataset",
    "ProceduralDataset",
    "HalfMoonsDataset",
    "SpiralsDataset",
    "CheckerboardDataset",
    "RingsDataset",
    "ConcentricSpheresDataset",
    "GaussiansDataset",
    "SwissRollDataset",
    "generate_procedural_dataset",
    "HierarchicalBatchSampler",
    "HierarchicalFileBatchSampler",
    "IndexedSample",
    "WithIndices",
    "with_indices",
    "SequenceCollate",
    "sequence_collate",
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
    "ComplexExpand",
    "ComplexCollapse",
    "BasisProjection",
    "InvBasisProjection",
    "SequentialTransform",
]
