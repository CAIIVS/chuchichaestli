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

from chuchichaestli.metrics.backbones.inception import InceptionEncoder
from chuchichaestli.metrics.backbones.swav import ResNet50Encoder  # , ResNet18Encoder
from chuchichaestli.metrics.backbones.clip import CLIPEncoder
from chuchichaestli.metrics.backbones.dinov2 import DINOv2Encoder
from chuchichaestli.metrics.backbones.load_backbone_encoder import load_encoder, MODELS
from chuchichaestli.metrics.backbones.resizer import pil_resize


__all__ = [
    "InceptionEncoder",
    "ResNet50Encoder",
    "CLIPEncoder",
    "DINOv2Encoder",
    "load_encoder",
    "MODELS",
    "pil_resize"
]
