# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Peak-Signal-to-Noise Ratio evaluation metric."""

import torch
from chuchichaestli.metrics.mse import MSE


class PSNR(MSE):
    """Peak-signal-to-noise ratio."""

    @torch.inference_mode()
    def update(
        self, data: torch.Tensor, prediction: torch.Tensor, update_range: bool = True
    ):
        """Compute metric on new input and update current state.

        Args:
            data: Observed data.
            prediction: Predicted data.
            update_range: If True, ranges are automatically updated based on
              new observation.
        """
        super().update(data, prediction, update_range=update_range)
        return self

    @torch.inference_mode()
    def compute(self) -> float:
        """Return current metric state total."""
        mse = super().compute()
        if mse is None:
            return None
        elif mse == 0:
            return torch.inf
        psnr = 10 * torch.log10(torch.pow(self.data_range, 2) / mse)
        self.value = psnr
        return psnr.item()
