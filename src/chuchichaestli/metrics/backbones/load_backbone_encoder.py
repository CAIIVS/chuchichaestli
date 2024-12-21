"""Load various (pretrained) encoders."""

import inspect

from chuchichaestli.metrics.backbones.backbone_encoder import Encoder
from chuchichaestli.metrics.backbones.inception import InceptionEncoder
from chuchichaestli.metrics.backbones.swav import ResNet50Encoder
from chuchichaestli.metrics.backbones.clip import CLIPEncoder
from chuchichaestli.metrics.backbones.dinov2 import DINOv2Encoder

MODELS = {
    "inception": InceptionEncoder,
    "sinception": InceptionEncoder,
    "swav": ResNet50Encoder,
    "clip": CLIPEncoder,
    "dinov2": DINOv2Encoder,
}


def load_encoder(model_name, device, **kwargs):
    """Load feature extractor"""

    model_cls = MODELS[model_name]

    # Get names of model_cls.setup arguments
    signature = inspect.signature(model_cls.setup)
    arguments = list(signature.parameters.keys())
    arguments = arguments[1:]  # Omit `self` arg

    # Initialize model using the `arguments` that have been passed in the `kwargs` dict
    encoder = model_cls(**{arg: kwargs[arg] for arg in arguments if arg in kwargs})
    encoder.name = model_name

    assert isinstance(
        encoder, Encoder
    ), "Can only get representations with Encoder subclasses!"

    return encoder.to(device)
