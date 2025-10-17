# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""MSE evaluation metric.

Note:
  This metric cannot reliably be used for gradient propagation;
  use `torch.nn.MSELoss` for training instead.
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
