"""1D U-Net blocks."""

import torch
from torch import nn

from chuchichaestli.models.attention import SelfAttention1D
from chuchichaestli.models.downsampling import Downsample1D
from chuchichaestli.models.resnet import ResnetBlock1D
from chuchichaestli.models.upsampling import Upsample1D


class DownBlock1D(nn.Module):
    """A 2D down block for the U-Net architecture.

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
        **kwargs,  # noqa
    ):
        """Initialize the down block."""
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock1D(
                    in_channels=in_channels,
                    mid_channels=in_channels,
                    out_channels=out_channels,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample1D(
                        out_channels,
                        use_conv=True,
                    )
                ]
            )
        else:
            self.downsamplers = None

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
            hidden_states = resnet(hidden_states)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class AttnDownBlock1D(nn.Module):
    """A 1D U-Net down block with attention."""

    pass


class MidBlock1D(nn.Module):
    """A 1D U-Net middle block.

    This block is used in the U-Net architecture for processing intermediate feature maps.
    It consists of a series of ResnetBlock1D and SelfAttention1D layers.

    Args:
        mid_channels (int): The number of channels in the intermediate feature maps.
        in_channels (int): The number of input channels.
        out_channels (int, optional): The number of output channels. If not provided, it is set to `in_channels`.

    Attributes:
        down (Downsample1D): The downsampling layer used at the beginning of the block.
        resnets (nn.ModuleList): The list of ResnetBlock1D layers.
        attentions (nn.ModuleList): The list of SelfAttention1D layers.
        up (Upsample1D): The upsampling layer used at the end of the block.

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
        """Initialize the 1D U-Net middle block."""
        super().__init__()

        # The orginal implementation uses a fixed cubic kernel for downsampling.
        self.down = Downsample1D(in_channels, use_conv=True)
        resnets = [
            ResnetBlock1D(in_channels, in_channels, in_channels),
        ]

        attentions = []
        for _ in range(num_layers):
            resnets.append(ResnetBlock1D(in_channels, in_channels, in_channels))
            if add_attention:
                attentions.append(SelfAttention1D(in_channels, in_channels))
            else:
                attentions.append(nn.Identity())

        # The orginal implementation uses a fixed cubic kernel for downsampling.
        self.up = Upsample1D(in_channels, use_conv=True)

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self, hidden_states: torch.FloatTensor, temb: torch.FloatTensor = None
    ) -> torch.FloatTensor:
        """Forward pass of the 1D U-Net middle block.

        Args:
            hidden_states (torch.FloatTensor): The input feature maps.
            temb (torch.FloatTensor, optional): The temporal embedding tensor.

        Returns:
            torch.FloatTensor: The output feature maps.

        """
        hidden_states = self.down(hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states)

        hidden_states = self.up(hidden_states)

        return hidden_states


class UpBlock1D(nn.Module):
    """A 2D up block for the U-Net architecture."""

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
        **kwargs,  # noqa
    ):
        """UpBlock2D is a module that represents an upsampling block in a U-Net architecture.

        Args:
            in_channels (int): Number of input channels.
            prev_output_channel (int): Number of output channels from the previous block.
            out_channels (int): Number of output channels.
            temb_channels (int): Number of channels in the temporal embedding.
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
                ResnetBlock1D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    mid_channels=out_channels,
                    out_channels=out_channels,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample1D(out_channels, use_conv=True)])
        else:
            self.upsamplers = None

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
            hidden_states = resnet(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class AttnUpBlock1D(nn.Module):
    """A 1D U-Net up block with attention."""

    pass


BLOCK_MAP_1D = {
    "DownBlock": DownBlock1D,
    "AttnDownBlock": AttnDownBlock1D,
    "MidBlock": MidBlock1D,
    "UpBlock": UpBlock1D,
    "AttnUpBlock": AttnUpBlock1D,
}
