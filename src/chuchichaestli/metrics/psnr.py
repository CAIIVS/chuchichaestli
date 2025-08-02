"""Peak-Signal-to-Noise Ratio evaluation metric.

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
