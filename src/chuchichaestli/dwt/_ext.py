# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional compiled kernels for the wavelet transform, and the fallback to torch."""

import torch

from functools import lru_cache

from chuchichaestli.dwt.modes import (
    MODE_TO_CODE,
    ExtensionModeTypes,
    pad_signal_adjoint,
)
from chuchichaestli.dwt.wavelet import Wavelet


_HAAR = Wavelet.from_name("haar")


__all__ = [
    "kernel_dtypes",
    "USE_CUSTOM_KERNELS",
    "kernels_available",
    "kernels_built",
    "dwt_axis",
    "haar_nd",
    "idwt_axis",
    "idwt_kernel_applies",
    "fused_haar_applies",
    "haar_wavedec",
    "wavedec_axes",
    "fused_recursion_applies",
]


# Global switch, so a benchmark or a test can force the pure-torch path.
USE_CUSTOM_KERNELS: bool = True

_dwt_kernels = None
_looked_for_kernels = False


def kernels_built() -> bool:
    """Whether the compiled kernels can be had, compiling them if need be.

    A wheel ships the sources rather than a compiled object, so the first call
    here is where a lazy just-in-time build happens.
    """
    global _dwt_kernels, _looked_for_kernels
    if not _looked_for_kernels:
        _looked_for_kernels = True
        try:
            from chuchichaestli.dwt import _dwt_kernels as built
        except ImportError:  # no cov
            from chuchichaestli import _jit

            built = _jit.load("dwt")
        _dwt_kernels = built
    return _dwt_kernels is not None


@lru_cache(maxsize=1)
def kernel_dtypes() -> frozenset[torch.dtype]:
    """The tensor types the compiled kernels were instantiated for.

    Read from the extension rather than restated here, so the fallback cannot
    disagree with what the kernels actually dispatch over.
    """
    if not kernels_built():
        return frozenset()
    return frozenset(_dwt_kernels.supported_dtypes())


def kernels_available(
    device: torch.device | str | None = None, dtype: torch.dtype | None = None
) -> bool:
    """Whether the compiled kernels can serve a tensor of this device and dtype.

    Args:
        device: Device to check, by name or as a `torch.device`; any device if
            omitted.
        dtype: Dtype to check; any dtype the kernels accept if omitted.
    """
    if not (USE_CUSTOM_KERNELS and kernels_built()):
        return False
    if dtype is not None and dtype not in kernel_dtypes():
        return False
    if device is not None and not isinstance(device, torch.device):
        device = torch.device(device)
    if device is None or device.type == "cpu":
        return True
    return bool(_dwt_kernels.has_gpu())


def _require_constant_filters(*filters: torch.Tensor) -> None:
    """Refuse a filter bank that expects a gradient.

    Args:
        filters: Filters the transform was handed.

    Raises:
        ValueError: If any filter requires a gradient.
    """
    if any(filt.requires_grad for filt in filters):
        raise ValueError(
            "the wavelet filters are constants; a learnable filter bank would"
            " need gradients the compiled kernels do not compute"
        )


class _DwtAxis(torch.autograd.Function):
    """Wavelet decomposition of one axis, computed by the compiled kernel.

    The transform is linear, so the backward pass is its adjoint, written with
    differentiable operations so that a second derivative works too.
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
        ctx.save_for_backward(dec_lo, dec_hi)
        ctx.config = (axis, mode, pad_lo, pad_hi, out_length, x.shape[2 + axis])
        return _dwt_kernels.dwt_axis(
            x.contiguous(), dec_lo, dec_hi, axis, MODE_TO_CODE[mode], pad_lo, out_length
        )

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """Apply the adjoint of the decomposition."""
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
    """Decomposition along one spatial axis, through the compiled kernel.

    The autograd machinery costs more than the kernel on a small transform,
    so it is only entered when there is a gradient to record.
    
    Args:
        x: Bands so far, shaped `(batch, groups, spatial...)`.
        dec_lo: Decomposition low-pass filter.
        dec_hi: Decomposition high-pass filter.
        axis: Spatial axis to transform.
        mode: Signal extension mode.
        pad_lo: Number of samples the decomposition prepends.
        pad_hi: Number of samples the decomposition appends.
        out_length: Length of the transformed axis.
    """
    _require_constant_filters(dec_lo, dec_hi)
    if not (torch.is_grad_enabled() and x.requires_grad):
        return _dwt_kernels.dwt_axis(
            x.contiguous(), dec_lo, dec_hi, axis, MODE_TO_CODE[mode], pad_lo, out_length
        )
    return _DwtAxis.apply(
        x, dec_lo, dec_hi, axis, mode, pad_lo, pad_hi, out_length
    )


class _IdwtAxis(torch.autograd.Function):
    """Wavelet reconstruction of one axis, computed by the compiled kernel.

    The transform is linear, so the backward pass is its adjoint, written with
    differentiable operations so that a second derivative works too.
    """

    @staticmethod
    def forward(
        ctx,
        coeffs: torch.Tensor,
        rec_lo: torch.Tensor,
        rec_hi: torch.Tensor,
        axis: int,
        mode: ExtensionModeTypes,
        trim: int,
        out_length: int,
    ) -> torch.Tensor:
        """Run the kernel and remember what the adjoint needs."""
        ctx.save_for_backward(rec_lo, rec_hi)
        ctx.config = (axis, mode, trim, out_length, coeffs.shape[1] // 2,
                      coeffs.shape[2 + axis])
        return _dwt_kernels.idwt_axis(
            coeffs.contiguous(), rec_lo, rec_hi, axis, MODE_TO_CODE[mode], trim,
            out_length,
        )

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """Apply the adjoint of the reconstruction."""
        rec_lo, rec_hi = ctx.saved_tensors
        axis, mode, trim, out_length, groups, coeff_len = ctx.config
        from chuchichaestli.dwt.functional import _bank, DIM_TO_CONV_FN_MAP

        dimensions = grad_out.ndim - 2
        filter_len = rec_lo.numel()
        full = 2 * (coeff_len - 1) + filter_len
        pad = [0] * (2 * dimensions)
        back = 2 * (dimensions - 1 - axis)
        if mode == "periodization":
            even = 2 * coeff_len
            pad[back + 1] = even - grad_out.shape[2 + axis]
            spread = torch.nn.functional.pad(grad_out, pad)
            index = (torch.arange(full, device=grad_out.device) - trim) % even
            padded = spread.index_select(2 + axis, index)
        else:
            pad[back] = trim
            pad[back + 1] = full - trim - out_length
            padded = torch.nn.functional.pad(grad_out, pad)
        weight = _bank(rec_lo, rec_hi, groups, axis, dimensions)
        stride = [1] * dimensions
        stride[axis] = 2
        grad = DIM_TO_CONV_FN_MAP[dimensions](
            padded.contiguous(), weight, stride=stride, groups=groups
        )
        return grad, None, None, None, None, None, None


def idwt_axis(
    coeffs: torch.Tensor,
    rec_lo: torch.Tensor,
    rec_hi: torch.Tensor,
    axis: int,
    mode: ExtensionModeTypes,
    trim: int,
    out_length: int,
) -> torch.Tensor:
    """Reconstruction along one spatial axis, through the compiled kernel.

    The autograd machinery costs more than the kernel on a small transform,
    so it is only entered when there is a gradient to record.

    Args:
        coeffs: Band pairs, shaped `(batch, 2 * groups, spatial...)`.
        rec_lo: Reconstruction low-pass filter.
        rec_hi: Reconstruction high-pass filter.
        axis: Spatial axis to reconstruct.
        mode: Signal extension mode the decomposition used.
        trim: Number of samples the reconstruction leads by.
        out_length: Length the reconstructed axis is trimmed to.
    """
    _require_constant_filters(rec_lo, rec_hi)
    if not (torch.is_grad_enabled() and coeffs.requires_grad):
        return _dwt_kernels.idwt_axis(
            coeffs.contiguous(), rec_lo, rec_hi, axis, MODE_TO_CODE[mode], trim,
            out_length,
        )
    return _IdwtAxis.apply(coeffs, rec_lo, rec_hi, axis, mode, trim, out_length)


def idwt_kernel_applies(device: torch.device) -> bool:
    """Whether the compiled reconstruction beats the transposed convolution.

    On the host it does not: the kernel is a plain loop against a vendor
    library, and measures several times slower. On an accelerator it wins.

    Args:
        device: Device the reconstruction runs on.
    """
    return device.type != "cpu"


def fused_haar_applies(
    wave, mode: ExtensionModeTypes, shape: tuple[int, ...]
) -> bool:
    """Whether the fused Haar transform can serve this call.

    Haar has two taps, so on an even axis it consumes no boundary extension at
    all and every mode agrees.

    Args:
        wave: Wavelet the transform was asked for.
        mode: Signal extension mode.
        shape: Lengths of the axes being transformed.
    """
    return (
        wave.filter_bank == _HAAR.filter_bank
        and all(n % 2 == 0 for n in shape)
        and len(shape) <= 3
    )


class _FusedHaar(torch.autograd.Function):
    """Haar decomposition over every axis, computed by the compiled kernel."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, dimensions: int) -> torch.Tensor:
        """Run the fused kernel and remember what the adjoint needs."""
        ctx.dimensions = dimensions
        return _dwt_kernels.haar_nd(x.contiguous(), 2**-0.5)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """Apply the adjoint, which for an orthogonal transform is the inverse."""
        from chuchichaestli.dwt.functional import _HAAR_ADJOINT

        return _HAAR_ADJOINT(grad_out, ctx.dimensions), None


def haar_nd(x: torch.Tensor, dimensions: int) -> torch.Tensor:
    """Haar decomposition over every spatial axis, through the compiled kernel.

    The autograd machinery costs more than the kernel on a small transform, so
    it is only entered when there is a gradient to record.

    Args:
        x: Input tensor, shaped `(batch, groups, spatial...)`.
        dimensions: Number of spatial axes.
    """
    if not (torch.is_grad_enabled() and x.requires_grad):
        return _dwt_kernels.haar_nd(x.contiguous(), 2**-0.5)
    return _FusedHaar.apply(x, dimensions)


def haar_wavedec(x: torch.Tensor, dimensions: int, levels: int) -> list[torch.Tensor]:
    """Multi-level Haar decomposition, with the recursion run by the kernel.

    Only for the path that records no gradient; with one, each level goes
    through the single-level entry point so autograd sees every step.

    Args:
        x: Input tensor, shaped `(batch, groups, spatial...)`.
        dimensions: Number of spatial axes.
        levels: Number of levels.
    """
    return _dwt_kernels.haar_wavedec(x.contiguous(), levels, 2**-0.5)


def wavedec_axes(
    x: torch.Tensor,
    dec_lo: torch.Tensor,
    dec_hi: torch.Tensor,
    mode: ExtensionModeTypes,
    levels: int,
) -> list[torch.Tensor]:
    """Multi-level decomposition over every axis, with the recursion run by the kernel.

    Only for the path that records no gradient; with one, each level goes
    through the single-level entry point so autograd sees every step.

    Args:
        x: Input tensor, shaped `(batch, groups, spatial...)`.
        dec_lo: Decomposition low-pass filter.
        dec_hi: Decomposition high-pass filter.
        mode: Signal extension mode.
        levels: Number of levels.
    """
    _require_constant_filters(dec_lo, dec_hi)
    return _dwt_kernels.wavedec_axes(
        x.contiguous(), dec_lo, dec_hi, MODE_TO_CODE[mode], levels
    )


def fused_recursion_applies(
    mode: ExtensionModeTypes, shapes: list[tuple[int, ...]]
) -> bool:
    """Whether the kernel can run the level recursion for this decomposition.

    The critically sampled mode pads an odd axis up to even, which the kernel
    does not do, so that combination stays with the per-level path.

    Args:
        mode: Signal extension mode.
        shapes: Length of every transformed axis, at each level.
    """
    if mode != "periodization":
        return True
    return all(n % 2 == 0 for shape in shapes for n in shape)
