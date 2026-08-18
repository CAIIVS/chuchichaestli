# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Channel dimension expansion/collapse transforms."""

from typing import Any
import torch
from torchvision.transforms.v2 import Transform

__all__ = ["ChannelExpand", "ChannelCollapse"]


class ChannelExpand(Transform):
    """Insert (and optionally replicate) a channel dimension.

    Maps `(W, H)`/`(B, W, H)` tensors to a channel-first `(B, C, W, H)` (or
    channel-last) layout.
    """

    _transformed_types = (torch.Tensor,)

    def __init__(self, channel_first: bool = True, replicate: int = 0):
        """Constructor.

        Args:
            channel_first: If `True`, insert the channel dim first, else last.
            replicate: If `>= 2`, replicate the new channel this many times.
        """
        super().__init__()
        self.channel_first = channel_first
        self.replicate = replicate

    def transform(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Expand the channel dimension."""
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(self.channel_first - 1)
        elif x.ndim == 3:
            x = x.unsqueeze(1) if self.channel_first else x.unsqueeze(-1)
        if self.replicate > 1:
            dim = 1 if self.channel_first else 3
            x = torch.cat((x,) * self.replicate, dim=dim)
        return x

    def revert(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Collapse the channel dimension."""
        if self.replicate > 1:
            x = x[:, 0, :, :] if self.channel_first else x[:, :, :, 0]
        if x.ndim == 4:
            if self.channel_first and x.shape[1] == 1:
                x = x.squeeze(1)
            elif not self.channel_first and x.shape[-1] == 1:
                x = x.squeeze(-1)
        return x


class ChannelCollapse(ChannelExpand):
    """Inverse of `ChannelExpand`."""

    def transform(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Collapse the channel dimension."""
        return super().revert(x, params)

    def revert(self, x: Any, params: dict[str, Any] | None = None) -> Any:
        """Expand the channel dimension."""
        return super().transform(x, params)
