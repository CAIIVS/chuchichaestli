# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Wavelet transform layers for 1, 2, and 3D inputs."""

import torch
from torch import nn

from chuchichaestli.dwt.modes import ExtensionModeTypes
from chuchichaestli.dwt.wavelet import Wavelet, wavelet as as_wavelet
from chuchichaestli.models.dwt.functional import (
    SubbandOrderTypes,
    _stack,
    _unstack,
    dwt_nd,
    dwt_nd_approx,
    idwt_nd,
    subband_names,
    wavedec_nd,
)
from chuchichaestli.utils import partialclass
from collections.abc import Sequence
from typing import Literal


__all__ = [
    "WaveletTransformND",
    "InverseWaveletTransformND",
    "LowpassWaveletTransformND",
    "MultilevelWaveletTransformND",
    "WaveletTransform1D",
    "WaveletTransform2D",
    "WaveletTransform3D",
    "InverseWaveletTransform1D",
    "InverseWaveletTransform2D",
    "InverseWaveletTransform3D",
    "LowpassWaveletTransform2D",
    "LowpassWaveletTransform3D",
    "WAVELET_LAYER_MAP",
    "WaveletLayerTypes",
]


class _WaveletLayer(nn.Module):
    """Common configuration of the wavelet transform layers.

    The filters are constants derived from the wavelet for the dtype and device
    of each input, so the layer holds no parameters or buffers and follows the
    input rather than needing to be moved.
    """

    def __init__(
        self,
        dimensions: int,
        wavelet: str | Wavelet = "haar",
        mode: ExtensionModeTypes = "zero",
        subband_order: SubbandOrderTypes = "subband",
    ):
        """Constructor.

        Args:
            dimensions: Number of spatial dimensions.
            wavelet: Wavelet, by name or as a `Wavelet`.
            mode: Signal extension mode.
            subband_order: How the subbands are laid out in the channel axis.
        """
        super().__init__()
        self.dimensions = dimensions
        self.wavelet = as_wavelet(wavelet)
        self.mode = mode
        self.subband_order = subband_order
        self.factor = 2

    def extra_repr(self) -> str:
        """Report the wavelet and extension mode."""
        return (
            f"dimensions={self.dimensions}, wavelet={self.wavelet.name!r},"
            f" mode={self.mode!r}, subband_order={self.subband_order!r}"
        )


class WaveletTransformND(_WaveletLayer):
    """Single-level wavelet transform, stacking the subbands on the channel axis.

    Maps `(N, C, s1, ..., sn)` onto `(N, 2**n C, s1 / 2, ..., sn / 2)`.
    """

    changes_channels = True

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the wavelet transform."""
        return dwt_nd(
            x, self.dimensions, self.wavelet, self.mode, self.subband_order
        )


class InverseWaveletTransformND(_WaveletLayer):
    """Single-level inverse wavelet transform; the inverse of `WaveletTransformND`.

    Maps `(N, 2**n C, s1, ..., sn)` onto `(N, C, 2 s1, ..., 2 sn)`.
    """

    changes_channels = True

    def forward(
        self, x: torch.Tensor, output_size: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Forward pass through the inverse wavelet transform.

        Args:
            x: Stacked subbands.
            output_size: Length of each spatial axis in the reconstruction;
                twice the input is assumed if omitted.
        """
        return idwt_nd(
            x,
            self.dimensions,
            self.wavelet,
            self.mode,
            self.subband_order,
            output_size,
        )


class LowpassWaveletTransformND(_WaveletLayer):
    """Approximation band of a single-level wavelet transform.

    Maps `(N, C, s1, ..., sn)` onto `(N, C, s1 / 2, ..., sn / 2)`, skipping the
    detail bands rather than computing and discarding them.
    """

    changes_channels = False

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """Forward pass through the approximation branch."""
        return dwt_nd_approx(x, self.dimensions, self.wavelet, self.mode)


class MultilevelWaveletTransformND(_WaveletLayer):
    """Wavelet transform applied repeatedly to the approximation band.

    Level `l` holds every subband of that level at `1 / 2**l` of the input
    resolution, which is the multiscale input a wavelet encoder consumes.
    """

    changes_channels = True

    def __init__(
        self,
        dimensions: int,
        wavelet: str | Wavelet = "haar",
        mode: ExtensionModeTypes = "zero",
        subband_order: SubbandOrderTypes = "subband",
        levels: int = 3,
    ):
        """Constructor.

        Args:
            dimensions: Number of spatial dimensions.
            wavelet: Wavelet, by name or as a `Wavelet`.
            mode: Signal extension mode.
            subband_order: How the subbands are laid out in the channel axis.
            levels: Number of levels.

        Raises:
            ValueError: If `levels` is not positive.
        """
        super().__init__(dimensions, wavelet, mode, subband_order)
        if levels < 1:
            raise ValueError(f"A decomposition needs at least one level; got {levels}.")
        self.levels = levels
        self.factor = 2**levels

    def forward(self, x: torch.Tensor, *args) -> list[torch.Tensor]:
        """Forward pass returning one stacked tensor per level, finest first."""
        return wavedec_nd(
            x,
            self.dimensions,
            self.wavelet,
            self.mode,
            self.levels,
            self.subband_order,
        )

    def extra_repr(self) -> str:
        """Report the wavelet, extension mode and number of levels."""
        return f"{super().extra_repr()}, levels={self.levels}"


class _NamedWaveletTransform(WaveletTransformND):
    """Wavelet transform returning the subbands separately rather than stacked."""

    def forward(self, x: torch.Tensor, *args) -> tuple[torch.Tensor, ...]:
        """Forward pass returning one tensor per subband, in `subband_names` order."""
        stacked = super().forward(x)
        return _unstack(stacked, self.dimensions, self.subband_order)

    @property
    def subbands(self) -> tuple[str, ...]:
        """Names of the subbands `forward` returns, in order."""
        return subband_names(self.dimensions)


class _NamedInverseWaveletTransform(InverseWaveletTransformND):
    """Inverse wavelet transform taking the subbands separately rather than stacked."""

    def forward(
        self, *subbands: torch.Tensor, output_size: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Forward pass over one tensor per subband, in `subband_names` order.

        Args:
            subbands: The `2**dimensions` subbands.
            output_size: Length of each spatial axis in the reconstruction.

        Raises:
            ValueError: If the number of subbands does not fit the dimensions.
        """
        expected = 2**self.dimensions
        if len(subbands) != expected:
            raise ValueError(
                f"A {self.dimensions}-dimensional transform has {expected} subbands;"
                f" got {len(subbands)}."
            )
        return super().forward(_stack(subbands, self.subband_order), output_size)

    @property
    def subbands(self) -> tuple[str, ...]:
        """Names of the subbands `forward` expects, in order."""
        return subband_names(self.dimensions)


WaveletTransform1D = partialclass("WaveletTransform1D", _NamedWaveletTransform, 1)
WaveletTransform2D = partialclass("WaveletTransform2D", _NamedWaveletTransform, 2)
WaveletTransform3D = partialclass("WaveletTransform3D", _NamedWaveletTransform, 3)
InverseWaveletTransform1D = partialclass(
    "InverseWaveletTransform1D", _NamedInverseWaveletTransform, 1
)
InverseWaveletTransform2D = partialclass(
    "InverseWaveletTransform2D", _NamedInverseWaveletTransform, 2
)
InverseWaveletTransform3D = partialclass(
    "InverseWaveletTransform3D", _NamedInverseWaveletTransform, 3
)
LowpassWaveletTransform2D = partialclass(
    "LowpassWaveletTransform2D", LowpassWaveletTransformND, 2
)
LowpassWaveletTransform3D = partialclass(
    "LowpassWaveletTransform3D", LowpassWaveletTransformND, 3
)


WaveletLayerTypes = Literal[
    "WaveletTransformND",
    "InverseWaveletTransformND",
    "LowpassWaveletTransformND",
    "MultilevelWaveletTransformND",
]

WAVELET_LAYER_MAP: dict[str, type[nn.Module]] = {
    "WaveletTransformND": WaveletTransformND,
    "InverseWaveletTransformND": InverseWaveletTransformND,
    "LowpassWaveletTransformND": LowpassWaveletTransformND,
    "MultilevelWaveletTransformND": MultilevelWaveletTransformND,
}
