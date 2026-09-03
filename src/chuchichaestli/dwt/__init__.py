# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Discrete wavelet transforms: filter banks, extension modes and the transforms themselves."""

from chuchichaestli.dwt.functional import (
    dwt,
    dwt_coeff_len,
    dwt_max_level,
    dwtn,
    dwtn_approx,
    idwt,
    idwtn,
    subband_keys,
    wavedec,
    wavedecn,
    waverec,
    waverecn,
)
from chuchichaestli.dwt.modes import (
    MODE_TO_CODE,
    ExtensionModeTypes,
    extension_indices,
    pad_signal,
)
from chuchichaestli.dwt.wavelet import (
    WAVELET_REGISTRY,
    Wavelet,
    WaveletTypes,
    wavelet,
    wavelist,
)


__all__ = [
    "MODE_TO_CODE",
    "WAVELET_REGISTRY",
    "ExtensionModeTypes",
    "Wavelet",
    "WaveletTypes",
    "dwt",
    "dwt_coeff_len",
    "dwt_max_level",
    "dwtn",
    "dwtn_approx",
    "extension_indices",
    "idwt",
    "idwtn",
    "pad_signal",
    "subband_keys",
    "wavedec",
    "wavedecn",
    "waverec",
    "waverecn",
    "wavelet",
    "wavelist",
]
