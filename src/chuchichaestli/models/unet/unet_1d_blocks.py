"""1D U-Net blocks."""

import torch
from torch import nn

from chuchichaestli.models.attention import SelfAttention1D
from chuchichaestli.models.downsampling import Downsample1D
from chuchichaestli.models.resnet import ResnetBlock1D
from chuchichaestli.models.upsampling import Upsample1D


class DownBlock1D(nn.Module):
    """A 1D U-Net down block."""

    def __init__(
        self, num_layers, out_channels: int, in_channels: int, mid_channels: int = None
    ):
        """Initialize the 1D U-Net down block."""
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        # The orginal implementation uses a fixed cubic kernel for downsampling.
        # This could be achieved by fixing the kernel parameter of the convolution
        # in the Downsample1D class.
        self.down = Downsample1D(in_channels, use_conv=True)

        self.resnets = []
        for i in range(num_layers):
            if i == 0:
                self.resnets.append(
                    ResnetBlock1D(in_channels, mid_channels, mid_channels)
                )
            elif i == num_layers - 1:
                self.resnets.append(
                    ResnetBlock1D(mid_channels, mid_channels, mid_channels)
                )
            else:
                (
                    self.resnets.append(
                        ResnetBlock1D(mid_channels, mid_channels, out_channels)
                    ),
                )

        self.resnets = nn.ModuleList(self.resnets)

    def forward(
        self, hidden_states: torch.FloatTensor, temb: torch.FloatTensor = None
    ) -> torch.FloatTensor:
        """Forward pass of the 1D U-Net down block."""
        hidden_states = self.down(hidden_states)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        return hidden_states, (hidden_states,)


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

    def __init__(self, mid_channels: int, in_channels: int, out_channels: int = None):
        """Initialize the 1D U-Net middle block."""
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        # The orginal implementation uses a fixed cubic kernel for downsampling.
        self.down = Downsample1D(in_channels, use_conv=True)
        resnets = [
            ResnetBlock1D(in_channels, mid_channels, mid_channels),
            ResnetBlock1D(mid_channels, mid_channels, mid_channels),
            ResnetBlock1D(mid_channels, mid_channels, mid_channels),
            ResnetBlock1D(mid_channels, mid_channels, mid_channels),
            ResnetBlock1D(mid_channels, mid_channels, mid_channels),
            ResnetBlock1D(mid_channels, mid_channels, out_channels),
        ]
        attentions = [
            SelfAttention1D(mid_channels, mid_channels // 32),
            SelfAttention1D(mid_channels, mid_channels // 32),
            SelfAttention1D(mid_channels, mid_channels // 32),
            SelfAttention1D(mid_channels, mid_channels // 32),
            SelfAttention1D(mid_channels, mid_channels // 32),
            SelfAttention1D(out_channels, out_channels // 32),
        ]
        # The orginal implementation uses a fixed cubic kernel for downsampling.
        self.up = Upsample1D(out_channels, use_conv=True)

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
    """A 1D U-Net up block."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        """Initialize the 1D U-Net up block."""
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels

        resnets = [
            ResnetBlock1D(2 * in_channels, mid_channels, mid_channels),
            ResnetBlock1D(mid_channels, mid_channels, mid_channels),
            ResnetBlock1D(mid_channels, mid_channels, out_channels),
        ]

        self.resnets = nn.ModuleList(resnets)
        # The orginal implementation uses a fixed cubic kernel for downsampling.
        # C.f. comment in Upblock1D
        self.up = Upsample1D(out_channels, use_conv=True)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: tuple[torch.FloatTensor, ...],
        temb: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """Forward pass of the 1D U-Net up block."""
        res_hidden_states = res_hidden_states_tuple[-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        hidden_states = self.up(hidden_states)

        return hidden_states


class AttnUpBlock1D(nn.Module):
    """A 1D U-Net up block with attention."""

    pass


BLOCK_MAP_1D = {
    "DownBlock": DownBlock1D,
    "AttnDownBlock": AttnDownBlock1D,
    "UpBlock": UpBlock1D,
    "AttnUpBlock": AttnUpBlock1D,
}
