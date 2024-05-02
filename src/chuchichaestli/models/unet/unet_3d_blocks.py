"""3D U-Net building blocks."""

import torch
from torch import nn

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


class AttnDownBlock3D(nn.Module):
    """A 3D U-Net down block with attention."""

    pass


class MidBlock3D(nn.Module):
    """A middle block for a 3D U-Net architecture."""

    pass


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


class AttnUpBlock3D(nn.Module):
    """A 3D U-Net up block with attention."""

    pass


BLOCK_MAP_3D = {
    "DownBlock": DownBlock3D,
    "AttnDownBlock": AttnDownBlock3D,
    "MidBlock": MidBlock3D,
    "UpBlock": UpBlock3D,
    "AttnUpBlock": AttnUpBlock3D,
}
