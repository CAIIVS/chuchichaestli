"""Implementation of the Attention Gate mechanism from "Attention U-Net: Learning Where to Look for the Pancreas".

This file is part of Chuchichaestli.

Chuchichaestli is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Chuchichaestli is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Chuchichaestli.  If not, see <http://www.gnu.org/licenses/>.

Developed by the Intelligent Vision Systems Group at ZHAW.
"""

import torch
from torch import nn
from torch.nn import functional as F

from chuchichaestli.models.resnet import ResnetBlock2D, ResnetBlock3D
from chuchichaestli.models.upsampling import Upsample2D, Upsample3D

from functools import partial


class AttentionGate(nn.Module):
    """Attention Gate mechanism from "Attention U-Net: Learning Where to Look for the Pancreas".

    C.f. https://arxiv.org/abs/1804.03999
    """

    def __init__(
        self,
        dimension: int = 2,
        num_channels_x: int = 1,
        num_channels_g: int = 1,
        num_channels_inter: int = 1,
        subsample_factor: tuple[int, ...] = 2,
    ):
        """Initialize the AttentionGate."""
        super().__init__()

        if dimension == 1:
            conv_cls = nn.Conv1d
            self.upsample_mode = "linear"
        elif dimension == 2:
            conv_cls = nn.Conv2d
            self.upsample_mode = "bilinear"
        elif dimension == 3:
            conv_cls = nn.Conv3d
            self.upsample_mode = "trilinear"
        else:
            raise ValueError(f"Invalid dimension: {dimension}")

        # TODO: What about a "multi-scale attention gate"?

        self.W_g = conv_cls(
            num_channels_g,
            num_channels_inter,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.W_x = conv_cls(
            num_channels_x,
            num_channels_inter,
            kernel_size=subsample_factor,
            stride=subsample_factor,
            padding=0,
            bias=False,
        )  # Eq. 1 doesn't have bias for W_x
        self.psi = conv_cls(
            num_channels_inter, 1, kernel_size=1, stride=1, padding=0, bias=True
        )

        self.W_out = conv_cls(
            num_channels_x,
            num_channels_x,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.sigma1 = nn.ReLU()
        self.sigma2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor, g: torch.TensorType):
        """Forward pass of the AttentionGate.

        Args:
            x: The input features.
            g: The gating signal.

        Returns:
            The attended features.
        """
        input_size = x.size()

        theta_x = self.W_x(x)
        phi_g = self.W_g(g)
        phi_g = F.interpolate(phi_g, size=theta_x.size()[2:], mode=self.upsample_mode)
        q = self.psi(self.sigma1(theta_x + phi_g))
        alpha = F.interpolate(
            self.sigma2(q), size=input_size[2:], mode=self.upsample_mode
        )
        x_hat = alpha.expand_as(x) * x
        x_hat = self.W_out(x_hat)
        return x_hat


class AttnGateUpBlock(nn.Module):
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
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        dimension: int = 2,
        **kwargs,  # noqa
    ):
        """Initialize the AttnGateUpBlock.

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
            output_scale_factor (float, optional): The scale factor for the output. Defaults to 1.0.
            add_upsample (bool, optional): Whether to add an upsampling layer. Defaults to True.
            dimension (int, optional): The dimension of the block. Defaults to 2.
            kwargs: Additional keyword arguments.
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

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            attentions.append(
                AttentionGate(
                    dimension=dimension,
                    num_channels_x=res_skip_channels,
                    num_channels_g=prev_output_channel,
                    num_channels_inter=res_skip_channels // 2,
                )
            )

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
        self.attentions = nn.ModuleList(attentions)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [upsample_cls(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

    def forward(
        self,
        hidden_states_and_gate: torch.FloatTensor,
        res_hidden_states_tuple: tuple[torch.FloatTensor, ...],
        temb: torch.FloatTensor | None = None,
        gate: torch.FloatTensor | None = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """Forward pass."""
        if isinstance(hidden_states_and_gate, tuple):
            hidden_states, gate = hidden_states_and_gate
        else:
            hidden_states = hidden_states_and_gate
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states_attn = attn(g=gate, x=res_hidden_states)
            hidden_states = torch.cat([hidden_states, hidden_states_attn], dim=1)
            hidden_states = resnet(hidden_states, temb)

        gate = hidden_states
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states, gate


AttnGateUpBlock2D = partial(AttnGateUpBlock, dimension=2)
AttnGateUpBlock3D = partial(AttnGateUpBlock, dimension=3)
