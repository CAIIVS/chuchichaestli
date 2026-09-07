# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Wavelet transform functions and layers for 1, 2, and 3D inputs."""

from chuchichaestli.dwt.functional import (
    dwt,
    dwt_coeff_len,
    dwt_max_level,
    dwtn,
    dwtn_approx,
    idwt,
    idwtn,
    wavedec,
    wavedecn,
    waverec,
    waverecn,
)
from chuchichaestli.dwt.modes import ExtensionModeTypes
from chuchichaestli.dwt.wavelet import Wavelet, WaveletTypes, wavelist
from chuchichaestli.models.dwt.functional import (
    SubbandOrderTypes,
    dwt_nd,
    dwt_nd_approx,
    idwt_nd,
    subband_names,
    wavedec_nd,
    waverec_nd,
)
from chuchichaestli.models.dwt.layers import (
    WAVELET_LAYER_MAP,
    InverseWaveletTransform1D,
    InverseWaveletTransform2D,
    InverseWaveletTransform3D,
    InverseWaveletTransformND,
    LowpassWaveletTransform2D,
    LowpassWaveletTransform3D,
    LowpassWaveletTransformND,
    MultilevelWaveletTransformND,
    WaveletLayerTypes,
    WaveletTransform1D,
    WaveletTransform2D,
    WaveletTransform3D,
    WaveletTransformND,
)


__all__ = [
    "WAVELET_LAYER_MAP",
    "ExtensionModeTypes",
    "InverseWaveletTransform1D",
    "InverseWaveletTransform2D",
    "InverseWaveletTransform3D",
    "InverseWaveletTransformND",
    "LowpassWaveletTransform2D",
    "LowpassWaveletTransform3D",
    "LowpassWaveletTransformND",
    "MultilevelWaveletTransformND",
    "SubbandOrderTypes",
    "Wavelet",
    "WaveletLayerTypes",
    "WaveletTransform1D",
    "WaveletTransform2D",
    "WaveletTransform3D",
    "WaveletTransformND",
    "WaveletTypes",
    "dwt",
    "dwt_coeff_len",
    "dwt_max_level",
    "dwt_nd",
    "dwt_nd_approx",
    "dwtn",
    "dwtn_approx",
    "idwt",
    "idwt_nd",
    "idwtn",
    "subband_names",
    "wavedec",
    "wavedec_nd",
    "wavedecn",
    "waverec",
    "waverec_nd",
    "waverecn",
    "wavelist",
]
