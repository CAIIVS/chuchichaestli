"""3D U-Net building blocks."""

from torch import nn


class DownBlock3D(nn.Module):
    """A 3D U-Net down block."""

    pass


class AttnDownBlock3D(nn.Module):
    """A 3D U-Net down block with attention."""

    pass


class UpBlock3D(nn.Module):
    """A 3D U-Net up block."""

    pass


class AttnUpBlock3D(nn.Module):
    """A 3D U-Net up block with attention."""

    pass


BLOCK_MAP_3D = {
    "DownBlock": DownBlock3D,
    "AttnDownBlock": AttnDownBlock3D,
    "UpBlock": UpBlock3D,
    "AttnUpBlock": AttnUpBlock3D,
}
