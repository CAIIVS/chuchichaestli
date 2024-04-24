"""1D U-Net blocks."""

from torch import nn


class DownBlock1D(nn.Module):
    """A 1D U-Net down block."""

    pass


class AttnDownBlock1D(nn.Module):
    """A 1D U-Net down block with attention."""

    pass


class UpBlock1D(nn.Module):
    """A 1D U-Net up block."""

    pass


class AttnUpBlock1D(nn.Module):
    """A 1D U-Net up block with attention."""

    pass


BLOCK_MAP_1D = {
    "DownBlock": DownBlock1D,
    "AttnDownBlock": AttnDownBlock1D,
    "UpBlock": UpBlock1D,
    "AttnUpBlock": AttnUpBlock1D,
}
