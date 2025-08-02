"""MSE evaluation metric (cannot be used for gradient propagation).

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
from chuchichaestli.metrics.base import EvalMetric


class MSE(EvalMetric):
    """Mean-squared error."""

    @torch.inference_mode()
    def update(
        self, data: torch.Tensor, prediction: torch.Tensor, update_range: bool = False
    ):
        """Compute metric on new input and update current state.

        Args:
            data: Observed data.
            prediction: Predicted data.
            update_range: If True, ranges are automatically updated based on
              new observation.
        """
        super().update(data, prediction, update_range=update_range)
        square_error = torch.pow(data[~self.is_nan] - prediction[~self.is_nan], 2)
        # sum square errors from batch
        aggregate_batch = torch.sum(square_error)
        self.aggregate = self.aggregate + aggregate_batch
        return self

    @torch.inference_mode()
    def compute(self) -> float:
        """Return current metric state total."""
        if self.n_observations == 0:
            return None
        mse = self.aggregate / self.n_observations
        self.value = mse
        return mse.item()
