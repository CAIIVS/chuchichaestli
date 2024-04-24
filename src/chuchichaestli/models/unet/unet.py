"""UNet model implementation."""

from torch import nn

from chuchichaestli.models.unet.unet_1d_blocks import BLOCK_MAP_1D
from chuchichaestli.models.unet.unet_2d_blocks import BLOCK_MAP_2D
from chuchichaestli.models.unet.unet_3d_blocks import BLOCK_MAP_3D

DIM_TO_BLOCK_MAP = {
    1: BLOCK_MAP_1D,
    2: BLOCK_MAP_2D,
    3: BLOCK_MAP_3D,
}


class UNet(nn.Module):
    """UNet model implementation."""

    def __init__(
        self,
        sample_size: int | tuple[int, ...] = None,
        dimensions: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        time_embedding_type: str = "positional",
        down_block_types: tuple[str, ...] = (
            "DownBlock",
            "AttnDownBlock",
            "AttnDownBlock",
            "AttnDownBlock",
        ),
        up_block_types: tuple[str, ...] = (
            "UpBlock",
            "AttnUpBlock",
            "AttnUpBlock",
            "AttnUpBlock",
        ),
        block_out_channels: tuple[int, ...] = (224, 448, 672, 896),
    ):
        """UNet model implementation."""
        super(__class__, self).__init__()

        if dimensions not in (1, 2, 3):
            raise ValueError("Only 1D, 2D, and 3D UNets are supported.")

        if len(down_block_types) != len(up_block_types):
            raise ValueError("Down and up block types must have the same length.")

        if len(down_block_types) != len(block_out_channels):
            raise ValueError(
                "Down block types and out channels must have the same length."
            )
