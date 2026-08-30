# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""A highly-customizable autoencoder implementation."""

import torch
from torch import nn
from chuchichaestli.models.activations import ActivationTypes
from chuchichaestli.models.autoencoder.decoder import Decoder
from chuchichaestli.models.autoencoder.encoder import Encoder
from chuchichaestli.models.autoencoder.traits import DecoderLike, EncoderLike
from chuchichaestli.models.maps import DIM_TO_CONV_MAP
from chuchichaestli.models.norm import NormTypes
from collections.abc import Sequence
from itertools import chain


__all__ = ["Autoencoder"]


ENCODER_RESERVED = {
    "dimensions": "pass `dimensions`",
    "in_channels": "pass `in_channels`",
    "out_channels": "pass `latent_dim`",
}

DECODER_RESERVED = {
    "dimensions": "pass `dimensions`",
    "in_channels": "derived from the encoder's latent width",
    "n_channels": "derived from the encoder's bottleneck width",
    "out_channels": "pass `out_channels`",
}


def _merge_shared_args(
    component_args: dict,
    res_args: dict,
    attn_args: dict,
    reserved: dict[str, str],
    name: str,
) -> dict:
    """Merge the shared block arguments into one component's own arguments.

    Args:
        component_args: Arguments for one component; its own block arguments win.
        res_args: Shared residual block arguments.
        attn_args: Shared attention block arguments.
        reserved: Keys the model sets itself, mapped to what to pass instead.
        name: Name of the checked dict, for the error message.

    Raises:
        ValueError: If a reserved key is present.
    """
    for key, hint in reserved.items():
        if key in component_args:
            raise ValueError(f'{name}["{key}"]: set by the model; {hint}.')
    return {
        **component_args,
        "res_args": {**res_args, **component_args.get("res_args", {})},
        "attn_args": {**attn_args, **component_args.get("attn_args", {})},
    }


def _reject_sequences(**kwargs) -> None:
    """Reject per-level sequences on arguments that both components share.

    Args:
        kwargs: Shared argument names and values.

    Raises:
        ValueError: If any value is a list or tuple.
    """
    for name, value in kwargs.items():
        if isinstance(value, (list, tuple)):
            group = "res_args" if name.startswith("res_") else "attn_args"
            raise ValueError(
                f"{name}: shared arguments take a single value; pass per-level values"
                f' in encoder_args["{group}"] or decoder_args["{group}"] instead.'
            )


class Autoencoder(nn.Module):
    """Flexible autoencoder implementation.

    The architecture consists of an encoder-decoder structure.
    The encoder chains several residual and downsampling blocks.
    Each downsampling block separates the encoder into spatially hierarchical levels.
    The encoder ends in bottleneck blocks (optionally including attention blocks
    and a convolutional layer) and projects the input into latent space.
    The decoder is built with residual convolutional and upsampling blocks, and
    expands from the latent space to the image domain.

    Both components are constructed externally and passed in, which lets a
    configuration framework instantiate them as separate groups. Use the `build`
    constructor to assemble a model from architecture arguments instead.

    Attributes:
        encoder_cls: Encoder class that `build` instantiates.
        decoder_cls: Decoder class that `build` instantiates.
    """

    encoder_cls: type = Encoder
    decoder_cls: type = Decoder

    def __init__(
        self,
        encoder: EncoderLike,
        decoder: DecoderLike,
        latent_proj: nn.Module | bool = True,
        latent_deproj: nn.Module | bool = True,
    ):
        """Assemble an autoencoder from an encoding and a decoding component.

        Args:
            encoder: Encoding component, mapping the input to latent space.
            decoder: Decoding component, expanding the latent space to the output.
            latent_proj: Projection between encoder and latent space; `True`
                builds a pointwise convolution, `False` omits it, and a module
                is used as given.
            latent_deproj: Projection between latent space and decoder, with the
                same options as `latent_proj`.
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        if latent_proj is True:
            latent_proj = self.projection(
                encoder, encoder.out_channels, encoder.out_channels
            )
        if latent_deproj is True:
            latent_deproj = self.projection(decoder, self.latent_dim, self.latent_dim)
        self.latent_proj = latent_proj if isinstance(latent_proj, nn.Module) else None
        self.latent_deproj = (
            latent_deproj if isinstance(latent_deproj, nn.Module) else None
        )
        self._check_components()

    @staticmethod
    def projection(
        component: EncoderLike | DecoderLike, in_channels: int, out_channels: int
    ) -> nn.Module:
        """Pointwise convolution matching the component it serves.

        Device and dtype are read from the component's first parameter or
        buffer, so a component placed before the model is assembled keeps the
        projection alongside it; a component holding neither leaves both to the
        torch defaults.

        Args:
            component: Encoding or decoding component the projection is attached to.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        ref = next(chain(component.parameters(), component.buffers()), None)
        return DIM_TO_CONV_MAP[component.dimensions](
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding="same",
            device=None if ref is None else ref.device,
            dtype=None if ref is None else ref.dtype,
        )

    def _check_components(self) -> None:
        """Check that the components and their projections agree on shape.

        Each check is skipped when either side does not expose the attribute it
        reads, so components that implement only part of the interface still work.

        Raises:
            ValueError: If the components disagree on dimensions or channel widths.
        """
        enc_dims = getattr(self.encoder, "dimensions", None)
        dec_dims = getattr(self.decoder, "dimensions", None)
        if None not in (enc_dims, dec_dims) and enc_dims != dec_dims:
            raise ValueError(
                f"encoder is {enc_dims}-dimensional but decoder is {dec_dims}-dimensional."
            )

        enc_out = getattr(self.encoder, "out_channels", None)
        proj_in = getattr(self.latent_proj, "in_channels", None)
        if None not in (enc_out, proj_in) and enc_out != proj_in:
            raise ValueError(
                f"latent_proj takes {proj_in} channels but the encoder emits {enc_out}."
            )

        dec_in = getattr(self.decoder, "in_channels", None)
        expected = getattr(self.latent_deproj, "out_channels", None)
        if expected is None and enc_out is not None:
            expected = self.latent_dim
        if None not in (dec_in, expected) and dec_in != expected:
            raise ValueError(
                f"decoder takes {dec_in} latent channels but receives {expected};"
                " pass an encoder with a matching `out_channels` (or a"
                " `latent_deproj` that projects onto the decoder's width)."
            )

    @classmethod
    def build(
        cls,
        dimensions: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_dim: int = 4,
        res_act_fn: ActivationTypes = "silu",
        res_dropout: float = 0.0,
        res_norm_type: NormTypes = "group",
        res_groups: int = 8,
        res_kernel_size: int = 3,
        attn_head_dim: int = 32,
        attn_n_heads: int = 1,
        attn_dropout_p: float = 0.0,
        attn_norm_type: NormTypes = "group",
        attn_groups: int = 32,
        attn_kernel_size: int = 1,
        attn_scales: Sequence[int] = (5,),
        context_args: dict = {},
        local_args: dict = {},
        encoder_args: dict = {},
        decoder_args: dict = {},
        **kwargs,
    ) -> "Autoencoder":
        """Build a model from architecture arguments, components included.

        The `res_*` and `attn_*` arguments are shared by both components; anything
        specific to one component goes into `encoder_args` or `decoder_args`, whose
        keys are the parameter names of `Encoder` and `Decoder`. Keys left out
        fall back to the defaults of `encoder_cls` and `decoder_cls`.

        Args:
            dimensions: Number of dimensions for the model.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            latent_dim: Number of channels in the latent space.
            res_act_fn: Activation function for the residual blocks
                (see `chuchichaestli.models.activations` for details).
            res_dropout: Dropout rate for the residual blocks.
            res_norm_type: Normalization type for the residual block
                (see `chuchichaestli.models.norm` for details).
            res_groups: Number of groups for the residual block normalization (if group norm).
            res_kernel_size: Kernel size for the residual blocks.
            attn_head_dim: Dimension of the attention heads.
            attn_n_heads: Number of attention heads.
            attn_dropout_p: Dropout probability of the scaled dot product attention.
            attn_norm_type: Normalization type for the convolutional attention block
                (see `chuchichaestli.models.norm` for details).
            attn_groups: Number of groups for the convolutional attention block normalization
                (if `attn_norm_type` is `"group"`).
            attn_kernel_size: Kernel size for the convolutional attention block.
            attn_scales: Scales for the multi-scale attention block.
            context_args: Keyword arguments for the context block in a transformer module.
            local_args: Keyword arguments for the local block in a transformer module.
            encoder_args: Arguments for the encoding component, overriding its defaults.
                Its `res_args` and `attn_args` override the shared values above.
            decoder_args: Arguments for the decoding component, with the same options.
            kwargs: Keyword arguments for the constructor, such as `latent_proj`.

        Returns:
            An assembled model of the class this was called on.
        """
        _reject_sequences(
            res_act_fn=res_act_fn,
            res_dropout=res_dropout,
            res_groups=res_groups,
            res_norm_type=res_norm_type,
            res_kernel_size=res_kernel_size,
            attn_head_dim=attn_head_dim,
            attn_n_heads=attn_n_heads,
            attn_dropout_p=attn_dropout_p,
            attn_groups=attn_groups,
            attn_kernel_size=attn_kernel_size,
        )

        res_args = {
            "res_act_fn": res_act_fn,
            "res_dropout": res_dropout,
            "res_groups": res_groups,
            "res_norm_type": res_norm_type,
            "res_kernel_size": res_kernel_size,
        }

        attn_args = {
            "n_heads": attn_n_heads,
            "head_dim": attn_head_dim,
            "dropout_p": attn_dropout_p,
            "norm_type": attn_norm_type,
            "kernel_size": attn_kernel_size,
            "groups": attn_groups,
            "scales": attn_scales,
            "context_args": context_args,
            "local_args": local_args,
        }

        enc_args = _merge_shared_args(
            encoder_args, res_args, attn_args, ENCODER_RESERVED, "encoder_args"
        )
        dec_args = _merge_shared_args(
            decoder_args, res_args, attn_args, DECODER_RESERVED, "decoder_args"
        )
        encoder = cls.encoder_cls(
            dimensions=dimensions,
            in_channels=in_channels,
            out_channels=latent_dim,
            **enc_args,
        )
        decoder = cls.decoder_cls(
            dimensions=dimensions,
            in_channels=encoder.latent_channels,
            n_channels=encoder.bottleneck_channels,
            out_channels=out_channels,
            **dec_args,
        )
        return cls(encoder=encoder, decoder=decoder, **kwargs)

    @property
    def channel_mults(self) -> int:
        """Total channel multiplication across the encoder levels."""
        return self.encoder.channel_mults

    @property
    def latent_dim(self) -> int:
        """Latent channel dimension."""
        return getattr(self.encoder, "latent_channels", self.encoder.out_channels)

    @property
    def levels(self) -> tuple[int, int]:
        """Number of stages in the encoder and decoder."""
        return self.encoder.levels, self.decoder.levels

    @property
    def f_comp(self) -> int:
        """Spatial compression factor of the encoder (number of spatial downsampling layers)."""
        return self.encoder.f

    @property
    def f_exp(self) -> int:
        """Spatial expansion factor of the decoder (number of spatial upsampling layers)."""
        return self.decoder.f

    def compute_latent_shape(
        self, input_shape: tuple[int, ...], no_batch_dim: bool = False
    ):
        """Compute the shape of the latent space."""
        batch_dim = input_shape[0] if not no_batch_dim else None
        spatial_dims = tuple(dim // self.f_comp for dim in input_shape[2:])
        if batch_dim is None:
            shape = (self.latent_dim, *spatial_dims)
        else:
            shape = (batch_dim, self.latent_dim, *spatial_dims)
        return shape

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input.

        Args:
            x: Input tensor.
            eps: Small constant value for numerical stability.

        Returns:
            Multivariate normal posterior distribution
        """
        z = self.encoder(x)
        z = self.latent_proj(z) if self.latent_proj is not None else z
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the input.

        Args:
            z: Input latent tensor.

        Returns:
            Image reconstructed from latent code.
        """
        z = self.latent_deproj(z) if self.latent_deproj is not None else z
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model, i.e. encode and decode."""
        code = self.encode(x)
        return self.decode(code)
