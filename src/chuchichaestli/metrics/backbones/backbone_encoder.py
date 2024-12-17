"""This file is part of Chuchichaestli.

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

from abc import ABC, abstractmethod

import torch.nn as nn


class Encoder(ABC, nn.Module):
    """Abstract Encoder class for pre-trained FD backbones."""
    def __init__(self, *args, **kwargs):
        """Constructor."""
        nn.Module.__init__(self)
        self.setup(*args, **kwargs)
        self.name = 'encoder'

    @abstractmethod
    def setup(self, *args, **kwargs):
        """Setup."""
        pass

    @abstractmethod
    def transform(self, x):
        """Converts a PIL Image to an input for the model."""
        pass

    def forward(self, *args, **kwargs):
        """Forward pass."""
        return self.model(*args, **kwargs)
