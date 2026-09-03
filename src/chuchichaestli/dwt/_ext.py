# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional compiled kernels for the wavelet transform, and the fallback to torch."""

import torch

from chuchichaestli.dwt.modes import (
    MODE_TO_CODE,
    ExtensionModeTypes,
    pad_signal_adjoint,
)


__all__ = ["USE_CUSTOM_KERNELS", "kernels_available", "dwt_axis", "idwt_axis"]


try:
    from chuchichaestli.dwt import _dwt_kernels
except ImportError:  # no cov
    _dwt_kernels = None

# Whether the extension was compiled and imported, which says nothing about
# whether an accelerator is present: it carries CPU kernels either way.
_KERNELS_AVAILABLE = _dwt_kernels is not None

# Global switch, so a benchmark or a test can force the pure-torch path.
USE_CUSTOM_KERNELS: bool = True


def kernels_available(device: torch.device | None = None) -> bool:
    """Whether the compiled kernels can serve a tensor on this device.

    Args:
        device: Device to check; any device if omitted.
    """
    if not (_KERNELS_AVAILABLE and USE_CUSTOM_KERNELS):
        return False
    if device is None or device.type == "cpu":
        return True
    return bool(_dwt_kernels.has_gpu())


class _AnalysisAxis(torch.autograd.Function):
    """Analysis along one axis, computed by the compiled kernel.

    The transform is linear, so the backward pass is its adjoint: undo the
    strided convolution, then scatter through the transpose of the boundary
    extension. Both are written with differentiable operations, so a second
    derivative works too.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        dec_lo: torch.Tensor,
        dec_hi: torch.Tensor,
        axis: int,
        mode: ExtensionModeTypes,
        pad_lo: int,
        pad_hi: int,
        out_length: int,
    ) -> torch.Tensor:
        """Run the kernel and remember what the adjoint needs."""
        if dec_lo.requires_grad or dec_hi.requires_grad:
            raise ValueError(
                "the wavelet filters are constants; a learnable filter bank would"
                " need gradients the compiled kernels do not compute"
            )
        ctx.save_for_backward(dec_lo, dec_hi)
        ctx.config = (axis, mode, pad_lo, pad_hi, out_length, x.shape[2 + axis])
        return _dwt_kernels.dwt_axis(
            x.contiguous(), dec_lo, dec_hi, axis, MODE_TO_CODE[mode], pad_lo, out_length
        )

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """Apply the adjoint of the analysis."""
        dec_lo, dec_hi = ctx.saved_tensors
        axis, mode, pad_lo, pad_hi, _, length = ctx.config
        from chuchichaestli.dwt.functional import _bank, DIM_TO_CONVT_FN_MAP

        dimensions = grad_out.ndim - 2
        groups = grad_out.shape[1] // 2
        weight = _bank(dec_lo.flip(0), dec_hi.flip(0), groups, axis, dimensions)
        stride = [1] * dimensions
        stride[axis] = 2
        padded = DIM_TO_CONVT_FN_MAP[dimensions](
            grad_out.contiguous(), weight, stride=stride, groups=groups
        )
        # the transposed convolution can fall short of the padded length when the
        # last taps reach past the final coefficient
        wanted = length + pad_lo + pad_hi
        short = wanted - padded.shape[2 + axis]
        if short > 0:
            pad = [0] * (2 * dimensions)
            pad[2 * (dimensions - 1 - axis) + 1] = short
            padded = torch.nn.functional.pad(padded, pad)
        elif short < 0:
            padded = padded.narrow(2 + axis, 0, wanted)
        grad = pad_signal_adjoint(padded, 2 + axis, pad_lo, pad_hi, mode, length)
        return grad, None, None, None, None, None, None, None


def dwt_axis(
    x: torch.Tensor,
    dec_lo: torch.Tensor,
    dec_hi: torch.Tensor,
    axis: int,
    mode: ExtensionModeTypes,
    pad_lo: int,
    pad_hi: int,
    out_length: int,
) -> torch.Tensor:
    """Analysis along one spatial axis, through the compiled kernel.

    Args:
        x: Bands so far, shaped `(batch, groups, spatial...)`.
        dec_lo: Decomposition low-pass filter.
        dec_hi: Decomposition high-pass filter.
        axis: Spatial axis to transform.
        mode: Signal extension mode.
        pad_lo: Number of samples the analysis prepends.
        pad_hi: Number of samples the analysis appends.
        out_length: Length of the transformed axis.
    """
    return _AnalysisAxis.apply(
        x, dec_lo, dec_hi, axis, mode, pad_lo, pad_hi, out_length
    )


def idwt_axis(
    coeffs: torch.Tensor,
    rec_lo: torch.Tensor,
    rec_hi: torch.Tensor,
    axis: int,
    mode: ExtensionModeTypes,
    trim: int,
    out_length: int,
) -> torch.Tensor:
    """Synthesis along one spatial axis, through the compiled kernel.

    Args:
        coeffs: Band pairs, shaped `(batch, 2 * groups, spatial...)`.
        rec_lo: Reconstruction low-pass filter.
        rec_hi: Reconstruction high-pass filter.
        axis: Spatial axis to reconstruct.
        mode: Signal extension mode the analysis used.
        trim: Number of samples the reconstruction leads by.
        out_length: Length the reconstructed axis is trimmed to.
    """
    return _dwt_kernels.idwt_axis(
        coeffs.contiguous(), rec_lo, rec_hi, axis, MODE_TO_CODE[mode], trim, out_length
    )
