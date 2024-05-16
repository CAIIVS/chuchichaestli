"""3D U-Net building blocks.

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


import torch
from torch import nn

from chuchichaestli.models.attention import Attention, AttnDownBlock3D, AttnUpBlock3D
from chuchichaestli.models.downsampling import Downsample3D
from chuchichaestli.models.resnet import ResnetBlock3D
from chuchichaestli.models.upsampling import Upsample3D


class DownBlock3D(nn.Module):
    """A 3D down block for the U-Net architecture.

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
        output_scale_factor (float, optional): The scale factor for the output. Defaults to 1.0.
        add_downsample (bool, optional): Whether to add a downsampling layer. Defaults to True.
        downsample_padding (int, optional): The padding size for the downsampling layer. Defaults to 1.
        kwargs: Additional keyword arguments.
    """

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
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        **kwargs,  # noqa
    ):
        """Initialize the down block."""
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
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

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor = None,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, ...]]:
        """Forward pass.

        Args:
            hidden_states (torch.FloatTensor): The input hidden states.
            temb (torch.FloatTensor, optional): The temporal embedding. Defaults to None.

        Returns:
            tuple[torch.FloatTensor, tuple[torch.FloatTensor, ...]]: The output hidden states and intermediate output states.
        """
        output_states = ()

        for resnet in self.resnets:
            # Removed the gradient checkpointing code
            hidden_states = resnet(hidden_states, temb)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class MidBlock3D(nn.Module):
    """A 3D UNet mid-block [`UNetMidBlock3D`] with multiple residual blocks and optional attention blocks.

    Args:
        in_channels (`int`): The number of input channels.
        temb_channels (`int`): The number of temporal embedding channels.
        dropout (`float`, *optional*, defaults to 0.0): The dropout rate.
        num_layers (`int`, *optional*, defaults to 1): The number of residual blocks.
        resnet_eps (`float`, *optional*, 1e-6 ): The epsilon value for the resnet blocks.
        resnet_time_scale_shift (`str`, *optional*, defaults to `default`):
            The type of normalization to apply to the time embeddings. This can help to improve the performance of the
            model on tasks with long-range temporal dependencies.
        resnet_act_fn (`str`, *optional*, defaults to `swish`): The activation function for the resnet blocks.
        resnet_groups (`int`, *optional*, defaults to 32):
            The number of groups to use in the group normalization layers of the resnet blocks.
        attn_groups (`Optional[int]`, *optional*, defaults to None): The number of groups for the attention blocks.
        add_attention (`bool`, *optional*, defaults to `True`): Whether to add attention blocks.
        attention_head_dim (`int`, *optional*, defaults to 1):
            Dimension of a single attention head. The number of attention heads is determined based on this value and
            the number of input channels.
        output_scale_factor (`float`, *optional*, defaults to 1.0): The output scale factor.

    Returns:
        `torch.FloatTensor`: The output of the last residual block, which is a tensor of shape `(batch_size,
        in_channels, height, width)`.

    """

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attn_groups: int = None,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
    ):
        """Initialize the UNetMidBlock3D."""
        super().__init__()
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )
        self.add_attention = add_attention

        if attn_groups is None:
            attn_groups = (
                resnet_groups if resnet_time_scale_shift == "default" else None
            )

        # Removed spatial timescale code
        resnets = [
            ResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=(
                            temb_channels
                            if resnet_time_scale_shift == "spatial"
                            else None
                        ),
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                    )
                )
            else:
                attentions.append(None)

            # Removed spatial timescale code
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self, hidden_states: torch.FloatTensor, temb: torch.FloatTensor = None
    ) -> torch.FloatTensor:
        """Forward pass."""
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states, temb=temb)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class UpBlock3D(nn.Module):
    """A 3D up block for the U-Net architecture."""

    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        resolution_idx: int = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        **kwargs,  # noqa
    ):
        """UpBlock3D is a module that represents an upsampling block in a U-Net architecture.

        Args:
            in_channels (int): Number of input channels.
            prev_output_channel (int): Number of output channels from the previous block.
            out_channels (int): Number of output channels.
            temb_channels (int): Number of channels in the temporal embedding.
            resolution_idx (int, optional): Index of the resolution. Defaults to None.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            num_layers (int, optional): Number of ResNet blocks in the block. Defaults to 1.
            resnet_eps (float, optional): Epsilon value for normalization layers in ResNet blocks. Defaults to 1e-6.
            resnet_time_scale_shift (str, optional): Time scale shift for the temporal embedding normalization. Defaults to "default".
            resnet_act_fn (str, optional): Activation function for the ResNet blocks. Defaults to "swish".
            resnet_groups (int, optional): Number of groups for group normalization in ResNet blocks. Defaults to 32.
            output_scale_factor (float, optional): Scale factor for the output. Defaults to 1.0.
            add_upsample (bool, optional): Whether to add an upsampling layer. Defaults to True.
            kwargs: Additional keyword arguments.
        """
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
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

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample3D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: tuple[torch.FloatTensor, ...],
        temb: torch.FloatTensor = None,
        upsample_size: int = None,
    ) -> torch.FloatTensor:
        """Forward pass.

        Args:
            hidden_states (torch.FloatTensor): Input hidden states.
            res_hidden_states_tuple (tuple[torch.FloatTensor, ...]): Tuple of residual hidden states.
            temb (torch.FloatTensor, optional): Temporal embedding. Defaults to None.
            upsample_size (int, optional): Size of the upsampling. Defaults to None.

        Returns:
            torch.FloatTensor: Output hidden states.
        """
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # Removed the free-U and gradient checkpointing code
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


BLOCK_MAP_3D = {
    "DownBlock": DownBlock3D,
    "AttnDownBlock": AttnDownBlock3D,
    "MidBlock": MidBlock3D,
    "UpBlock": UpBlock3D,
    "AttnUpBlock": AttnUpBlock3D,
}
