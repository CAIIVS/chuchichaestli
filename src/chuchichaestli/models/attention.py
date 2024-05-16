"""Attention layers.

Copyright 2024 The HuggingFace Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Additional modifications made by the Intelligent Vision Systems Group at ZHAW under the
GNU General Public License v3.0 which extends the conditions of the License for further
redistribution and use. See the GPLv3 license at

    http://www.gnu.org/licenses/gpl-3.0.html

This file is part of Chuchichaestli and has been modified for use in this project.
"""

from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chuchichaestli.models.downsampling import Downsample2D, Downsample3D
from chuchichaestli.models.normalization import SpatialNorm
from chuchichaestli.models.resnet import ResnetBlock2D, ResnetBlock3D
from chuchichaestli.models.upsampling import Upsample2D, Upsample3D


class SelfAttention1D(nn.Module):
    """Self-attention layer for 1D inputs."""

    def __init__(self, in_channels: int, n_head: int = 1, dropout_rate: float = 0.0):
        """Initialize the SelfAttention1D layer."""
        super().__init__()
        self.channels = in_channels
        self.group_norm = nn.GroupNorm(1, num_channels=in_channels)
        self.num_heads = n_head

        # Ensure that the channels can be divided by the number of heads
        if in_channels % n_head != 0:
            raise ValueError("in_channels must be divisible by n_head")

        self.query = nn.Linear(self.channels, self.channels)
        self.key = nn.Linear(self.channels, self.channels)
        self.value = nn.Linear(self.channels, self.channels)

        self.proj_attn = nn.Linear(self.channels, self.channels, bias=True)

        self.dropout = nn.Dropout(dropout_rate)

    def transpose_for_scores(self, x):
        """Transpose the input tensor for the attention mechanism."""
        new_x_shape = x.size()[:-1] + (self.num_heads, self.channels // self.num_heads)
        x = x.view(*new_x_shape).permute(0, 2, 1, 3)
        return x

    def forward(self, hidden_states):
        """Forward pass of the SelfAttention1D layer."""
        residual = hidden_states
        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.transpose(
            1, 2
        )  # Switch from (B, C, T) to (B, T, C)

        # Project inputs to Q, K, V
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        # Transpose for multi-head attention
        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        # Attention mechanism
        scale = self.channels**-0.5
        attn_output = F.scaled_dot_product_attention(
            query, key, value, scale=scale, dropout_p=0.0
        )

        # Concatenate heads and put through final linear layer
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(attn_output.size()[:-2] + (self.channels,))

        attn_output = self.proj_attn(attn_output)
        attn_output = attn_output.transpose(1, 2)
        attn_output = self.dropout(attn_output)

        output = attn_output + residual
        return output


class Attention(nn.Module):
    """A general cross-attention layer."""

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        cross_attention_norm: str = None,
        cross_attention_norm_num_groups: int = 32,
        norm_num_groups: int = None,
        spatial_norm_dim: int = None,
        out_bias: bool = True,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        out_dim: int = None,
    ) -> None:
        """Initialize the Attention layer."""
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection

        # Projection dimensions
        self.cross_attention_dim = (
            cross_attention_dim if cross_attention_dim is not None else query_dim
        )
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.out_dim = out_dim if out_dim is not None else query_dim

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(
                num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True
            )
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            self.spatial_norm = SpatialNorm(
                f_channels=query_dim, zq_channels=spatial_norm_dim
            )
        else:
            self.spatial_norm = None

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            # Removed option for added projection dimension.
            norm_cross_num_channels = self.cross_attention_dim
            self.norm_cross = nn.GroupNorm(
                num_channels=norm_cross_num_channels,
                num_groups=cross_attention_norm_num_groups,
                eps=1e-5,
                affine=True,
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )

        self.query = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.key = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.value = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.out = nn.ModuleList(
            [
                nn.Linear(self.inner_dim, self.out_dim, bias=out_bias),
                nn.Dropout(dropout),
            ]
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor = None,
        temb: torch.FloatTensor = None,
    ):
        """Forward pass of the Attention layer."""
        residual = hidden_states
        if self.spatial_norm is not None:
            hidden_states = self.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        elif input_ndim == 5:
            batch_size, channel, depth, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, depth * height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = self._prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, self.heads, -1, attention_mask.shape[-1]
            )

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = self.query(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif self.norm_cross:
            encoder_hidden_states = self._norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = self.key(encoder_hidden_states)
        value = self.value(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, self.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = self.out[0](hidden_states)
        # dropout
        hidden_states = self.out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        elif input_ndim == 5:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, depth, height, width
            )

        if self.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.rescale_output_factor

        return hidden_states

    def _norm_encoder_hidden_states(
        self, encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        r"""Normalize the encoder hidden states.

        Requires `self.norm_cross` to be specified when constructing the `Attention` class.

        Args:
            encoder_hidden_states (`torch.Tensor`): Hidden states of the encoder.

        Returns:
            `torch.Tensor`: The normalized encoder hidden states.
        """
        assert (
            self.norm_cross is not None
        ), "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

        if isinstance(self.norm_cross, nn.LayerNorm):
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        elif isinstance(self.norm_cross, nn.GroupNorm):
            # Group norm norms along the channels dimension and expects
            # input to be in the shape of (N, C, *). In this case, we want
            # to norm along the hidden dimension, so we need to move
            # (batch_size, sequence_length, hidden_size) ->
            # (batch_size, hidden_size, sequence_length)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
        else:
            raise ValueError("Unknown normalization layer")

        return encoder_hidden_states

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        target_length: int,
        batch_size: int,
        out_dim: int = 3,
    ) -> torch.Tensor:
        r"""Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`torch.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `torch.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (
                    attention_mask.shape[0],
                    attention_mask.shape[1],
                    target_length,
                )
                padding = torch.zeros(
                    padding_shape,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
                #       we want to instead pad by (0, remaining_length), where remaining_length is:
                #       remaining_length: int = target_length - current_length
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask


class AttnDownBlock(nn.Module):
    """A 2D and 3D down block with attention for the U-Net architecture."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        dimension: int = 2,
    ):
        """Initialize the AttnDownBlock.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            temb_channels (int): The number of channels in the temporal embedding.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            num_layers (int, optional): The number of ResNet blocks in the down block. Defaults to 1.
            resnet_eps (float, optional): The epsilon value for normalization in the ResNet blocks. Defaults to 1e-6.
            resnet_time_scale_shift (str, optional): The time scale shift method for the ResNet blocks. Defaults to "default".
            resnet_act_fn (str, optional): The activation function for the ResNet blocks. Defaults to "swish".
            resnet_groups (int, optional): The number of groups for group normalization in the ResNet blocks. Defaults to 32.
            attention_head_dim (int, optional): The dimension of a single attention head. Defaults to 1.
            output_scale_factor (float, optional): The scale factor for the output. Defaults to 1.0.
            add_downsample (bool, optional): Whether to add a downsampling layer. Defaults to True.
            downsample_padding (int, optional): The padding size for the downsampling layer. Defaults to 1.
            dimension (int, optional): The dimension of the block. Defaults to 2.
        """
        super().__init__()
        resnets = []
        attentions = []

        if dimension == 2:
            resnet_block_cls = ResnetBlock2D
            upsample_cls = Downsample2D
        elif dimension == 3:
            resnet_block_cls = ResnetBlock3D
            upsample_cls = Downsample3D

        if attention_head_dim is None:
            attention_head_dim = out_channels

        if attention_head_dim > out_channels:
            raise ValueError(
                f"Attention head dimension {attention_head_dim} must be less than or equal to the number of output "
                f"channels {out_channels}."
            )

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                resnet_block_cls(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                )
            )
            self.resnets = nn.ModuleList(resnets)

            attentions.append(
                Attention(
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                    residual_connection=True,
                    bias=True,
                )
            )
            self.attentions = nn.ModuleList(attentions)

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    upsample_cls(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                    )
                ]
            )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, ...]]:
        """Forward pass."""
        cross_attention_kwargs = (
            cross_attention_kwargs if cross_attention_kwargs is not None else {}
        )

        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, **cross_attention_kwargs)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class AttnUpBlock(nn.Module):
    """A 2D and 3D up block with attention for the U-Net architecture."""

    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        dimension: int = 2,
    ):
        """Initialize the AttnUpBlock.

        Args:
            in_channels (int): The number of input channels.
            prev_output_channel (int): The number of output channels from the previous block.
            out_channels (int): The number of output channels.
            temb_channels (int): The number of channels in the temporal embedding.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            num_layers (int, optional): The number of ResNet blocks in the up block. Defaults to 1.
            resnet_eps (float, optional): The epsilon value for normalization in the ResNet blocks. Defaults to 1e-6.
            resnet_time_scale_shift (str, optional): The time scale shift method for the ResNet blocks. Defaults to "default".
            resnet_act_fn (str, optional): The activation function for the ResNet blocks. Defaults to "swish".
            resnet_groups (int, optional): The number of groups for group normalization in the ResNet blocks. Defaults to 32.
            attention_head_dim (int, optional): The dimension of a single attention head. Defaults to 1.
            output_scale_factor (float, optional): The scale factor for the output. Defaults to 1.0.
            add_upsample (bool, optional): Whether to add an upsampling layer. Defaults to True.
            dimension (int, optional): The dimension of the block. Defaults to 2.
        """
        super().__init__()
        resnets = []
        attentions = []

        if dimension == 2:
            resnet_block_cls = ResnetBlock2D
            upsample_cls = Upsample2D
        elif dimension == 3:
            resnet_block_cls = ResnetBlock3D
            upsample_cls = Upsample3D

        if attention_head_dim > out_channels:
            raise ValueError(
                f"Attention head dimension {attention_head_dim} must be less than or equal to the number of output "
                f"channels {out_channels}."
            )

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                resnet_block_cls(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                )
            )
            self.resnets = nn.ModuleList(resnets)

            attentions.append(
                Attention(
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                    residual_connection=True,
                    bias=True,
                )
            )

        self.attentions = nn.ModuleList(attentions)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [upsample_cls(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: tuple[torch.FloatTensor, ...],
        temb: torch.FloatTensor | None = None,
        upsample_size: int | None = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        """Forward pass."""
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


AttnDownBlock2D = partial(AttnDownBlock, dimension=2)
AttnDownBlock3D = partial(AttnDownBlock, dimension=3)
AttnUpBlock2D = partial(AttnUpBlock, dimension=2)
AttnUpBlock3D = partial(AttnUpBlock, dimension=3)
