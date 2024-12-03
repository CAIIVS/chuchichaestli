"""Base injectible diffusion model for diffusion processes.

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
from torch import Tensor
from chuchichaestli.diffusion.ddpm.base import DiffusionProcess
from chuchichaestli.injectables.typedefs import STEP_OUTPUT
from chuchichaestli.injectables.typedefs import Fätch
from chuchichaestli.injectables.bätchfätch import Identity

import torch.nn as nn
import lightning as L

# --------------------------------------------------------------------------------------
# module
# --------------------------------------------------------------------------------------


class CondBase(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        scheduler: DiffusionProcess,
        train_loss: nn.modules.loss._Loss,
        valid_loss: nn.modules.loss._Loss,
        train_fätch: Fätch | None,
        valid_fätch: Fätch | None,
    ):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.valid_loss = train_loss
        self.valid_loss = valid_loss
        self.train_fätch = train_fätch or Identity()
        self.valid_fätch = valid_fätch or Identity()

    def forward(self, inputs) -> Tensor:
        if isinstance(inputs, dict):
            output = self.model(**inputs)
        else:
            output = self.model(inputs)
        return output

    def training_step(self, batch, batch_idx: int) -> STEP_OUTPUT:
        inputs = self.train_fätch(batch)

        # sample noise, timesteps
        x_t, noise, timesteps = self.scheduler.noise_step(inputs)

        # predict noise
        output = self.forward({"x": x_t, "t": timesteps})

        # compute loss
        loss = self.valid_loss(output, noise)
        return STEP_OUTPUT(
            loss=loss,
            inputs=inputs,
            output=output,
            target=noise,
        )

    def validation_step(self, batch, batch_idx: int, dataloader_idx=0) -> STEP_OUTPUT:
        inputs, target = self.valid_fätch(batch)

        # generate sample
        output = self.predict_step(target)

        # compute loss
        loss = self.valid_loss(output, target)
        return STEP_OUTPUT(
            loss=loss,
            inputs=inputs,
            output=output,
            target=target,
        )

    @torch.no_grad()
    def predict_step(self, condition: Tensor) -> Tensor:
        return next(iter(self.scheduler.generate(self.model, condition)))
