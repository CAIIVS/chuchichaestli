# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""LPIPS loss metric, including various feature extraction backbones.

Note:
  The original implementation of LPIPS (https://arxiv.org/abs/1801.03924) uses a
  scaling layer, which adjusts the standardization based on the ImageNet stats
  `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]`. Note, these
  scalings are already included in the `FeatureExtractor` module via
  `torchvision.models.Weights.transforms()`.
"""

from functools import partial, singledispatchmethod
from pathlib import Path
import torch
from torch.nn import Module, ModuleList
from torch.nn.functional import interpolate
from torchvision.transforms.v2.functional import normalize
from torchvision import models as tv
from torchvision.models.feature_extraction import create_feature_extractor
from torch.fx.graph_module import GraphModule
from chuchichaestli.metrics.base import sanitize_ndim, as_tri_channel, as_batched_slices
from chuchichaestli.models.blocks import BaseConvBlock
from chuchichaestli.utils import partialclass
from typing import Any, Literal
from collections.abc import Iterable, Sequence, Callable


__all__ = [
    "LPIPSLoss",
    "LPIPSVGG16",
    "LPIPSAlexNet",
    "LPIPSSqueezeNet",
    "LPIPSResNet18",
    "LPIPSViT",
    "LPIPSConvNeXt",
    "LPIPSSwinV2",
    "LPIPSEmbedding",
]


class FeatureExtractor(Module):
    """Generic feature extractor base class."""

    def __init__(
        self,
        model_name: str,
        default_weights: tv.Weights | str,
        feature_nodes: dict,
        weights: tv.Weights | str | None = None,
        use_default_transforms: bool = True,
        input_size: tuple[int, int] = (224, 224),
        mode: str = "bilinear",
        antialias: bool = False,
    ):
        """Constructor.

        Args:
            model_name: The model name of a feature extraction backbone.
            default_weights: The pretrained weights for the specific model;
              for details and possible values, see
              https://docs.pytorch.org/vision/main/_modules/torchvision/models/_api.html.
            feature_nodes: Features to be extracted; the dict keys are node names,
              and the values user-specified keys that should be sortable via `sorted`.
            weights: The user-specified pretrained weights in the child class.
            use_default_transforms: If `True`, uses standard transforms for
              preprocessing (VGG16_Weights.IMAGENET1K_V1.transforms).
            input_size: If `use_default_transforms=False`, this is the resize target size.
            mode: If `use_default_transforms=False`, a simple interpolation in
              given mode is performed.
            antialias: If `use_default_transforms=False` and `True`,
              antialiasing is used during interpolation.
        """
        self.extractor: GraphModule
        super().__init__()
        self.model_name = model_name
        self.weights = default_weights if weights is None else weights
        model = tv.get_model(model_name, weights=self.weights)
        self.extractor = create_feature_extractor(model, feature_nodes)
        self.input_size = input_size
        if use_default_transforms:
            if isinstance(self.weights, str):
                weights = tv.get_model_weights("inception_v3")[self.weights]
                self.transforms = weights.transforms()
            else:
                self.transforms = self.weights.transforms()
        else:
            self.transforms = partial(
                interpolate,
                size=self.input_size,
                mode=mode,
                align_corners=False,
                antialias=antialias,
            )

    @property
    def feature_channels(self):
        """Feature channels of the extractor's output."""
        if not hasattr(self, "_feature_channels"):
            sample = torch.rand((1, 1, *self.input_size))
            features = self(sample)
            channels = [feature.shape[1] for feature in features]
            self._feature_channels = channels
        return self._feature_channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Feature extraction."""
        x = sanitize_ndim(x)
        x = as_tri_channel(x)
        x = self.transforms(x)
        # Extract features
        features = self.extractor(x)
        outputs = []
        for k in sorted(features.keys()):
            feature = features[k]
            # vision transformers flatten features
            if feature.ndim == 3 and "vit" in self.model_name:
                feature = feature[:, 1:, :]  # omit CLS token
                B, N, D = feature.shape
                H = W = int(N**0.5)
                feature = feature.permute(0, 2, 1).contiguous().view(B, D, H, W)
            # swin transformers are channel-last
            elif "swin" in self.model_name:
                feature = feature.permute(0, 3, 1, 2)
            # elif feature.ndim == 2:
            #     feature = feature.unsqueeze(-1).unsqueeze(-1)
            outputs.append(feature)
        return outputs


class LPIPSVGG16(FeatureExtractor):
    """VGG16 feature extractor for LPIPSLoss."""

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(
            model_name="vgg16",
            default_weights=tv.VGG16_Weights.IMAGENET1K_V1,
            feature_nodes={
                "features.3": 0,
                "features.8": 1,
                "features.15": 2,
                "features.22": 3,
                "features.29": 4,
            },
            **kwargs,
        )


class LPIPSAlexNet(FeatureExtractor):
    """AlexNet feature extractor for LPIPSLoss."""

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(
            model_name="alexnet",
            default_weights=tv.AlexNet_Weights.IMAGENET1K_V1,
            feature_nodes={
                "features.1": 0,
                "features.4": 1,
                "features.7": 2,
                "features.9": 3,
                "features.11": 4,
            },
            **kwargs,
        )


class LPIPSSqueezeNet(FeatureExtractor):
    """SqueezeNet feature extractor for LPIPSLoss."""

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(
            model_name="squeezenet1_1",
            default_weights=tv.SqueezeNet1_1_Weights.IMAGENET1K_V1,
            feature_nodes={
                "features.1": 0,
                # "features.3": 1,
                "features.4": 2,
                # "features.6": 3,
                "features.7": 4,
                "features.9": 5,
                "features.10": 6,
                "features.11": 7,
                "features.12": 8,
            },
            **kwargs,
        )


class LPIPSResNet18(FeatureExtractor):
    """ResNet18 feature extractor for LPIPSLoss."""

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(
            model_name="resnet18",
            default_weights=tv.ResNet18_Weights.IMAGENET1K_V1,
            feature_nodes={
                "relu": 0,
                "layer1.1.relu": 1,
                "layer2.1.relu": 2,
                "layer3.1.relu": 3,
                "layer4.1.relu": 4,
            },
            **kwargs,
        )


class LPIPSConvNeXt(FeatureExtractor):
    """ConvNeXt-Tiny feature extractor for LPIPSLoss."""

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(
            model_name="convnext_tiny",
            default_weights=tv.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
            feature_nodes={
                "features.1": 0,
                "features.3": 1,
                "features.5": 2,
                "features.7": 3,
            },
            **kwargs,
        )


class LPIPSViT(FeatureExtractor):
    """ViT-B-16 feature extractor for LPIPSLoss."""

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(
            model_name="vit_b_16",
            default_weights=tv.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1,
            feature_nodes={
                "conv_proj": 0,
                "encoder.layers.encoder_layer_1.ln_1": 1,
                "encoder.layers.encoder_layer_2.ln_1": 2,
                "encoder.layers.encoder_layer_5.ln_1": 3,
                "encoder.layers.encoder_layer_8.ln_1": 4,
                "encoder.ln": 5,
            },
            input_size=(384, 384),
            **kwargs,
        )


class LPIPSSwinV2(FeatureExtractor):
    """Small SwinTransformer-V2 feature extractor for LPIPSLoss."""

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(
            model_name="swin_v2_s",
            default_weights=tv.Swin_V2_S_Weights.IMAGENET1K_V1,
            feature_nodes={
                "features.1": 0,
                "features.3": 1,
                "features.5.9": 2,
                "features.5": 3,
                "features.7": 4,
            },
            input_size=(256, 256),
            **kwargs,
        )


LPIPSEmbeddingBlock = partialclass(
    "LPIPSEmbeddingBlock",
    BaseConvBlock,
    act_fn="softplus",
    act_last=True,
    norm=False,
    dropout_p=0.5,
    kernel_size=1,
    stride=1,
    padding=0,
    bias=False,
)


class LPIPSEmbedding(ModuleList):
    """Linear deep embedding for LPIPS extracted features."""

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int | list[int] = 1,
        softplus: bool = False,
        dropout: bool = False,
        clamp_weights: bool = False,
    ):
        """Constructor.

        Args:
            in_channels: Input channels for each linear embedding layer.
            out_channels: Output channels for each linear embedding layer.
            softplus: If `True`, uses a softplus activation after each embedding layer.
            dropout: If `True`, uses dropout in the embedding.
            clamp_weights: If `True`, the weights of the embedding layers are clamped;
              this is necessary to avoid negative values
        """
        if not isinstance(out_channels, Iterable):
            out_channels = [out_channels] * len(in_channels)
        layers = []
        for in_c, out_c in zip(in_channels, out_channels):
            layer = LPIPSEmbeddingBlock(2, in_c, out_c, act=softplus, dropout=dropout)
            layers.append(layer)
        super().__init__(layers)
        if clamp_weights:
            self.clamp_weights()

    def clamp_weights(self):
        """Clamp weights of the convolutions (to avoid negative outputs)."""
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.weight.data.clamp_(min=0.0)

    def forward(self, x: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        """Forward method for a sequence of LPIPS feature loss tensors."""
        assert len(x) == len(self)
        scores = [layer(xi) for layer, xi in zip(self, x)]
        return scores

    def save(self, path: Path | str = Path("lpips_embedding.pth")) -> Path:
        """Save the model weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    def load(self, path: Path | str = Path("lpips_embedding.pth")):
        """Load the model weights."""
        path = Path(path)
        if path.exists():
            states = torch.load(path, weights_only=True)
            self.load_state_dict(states)
            self.eval()
        # else:
        #     raise ValueError(f"Model weights not found at {path}!")


class LPIPSNonEmbedding(Module):
    """A simple weighted-sum channel reduction embedding for LPIPS features."""

    def __init__(self, in_channels: list[int]):
        super().__init__()
        self.channels = in_channels
        self.weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.ones(in_c).view(1, in_c, 1, 1))
                for in_c in self.channels
            ]
        )

    def clamp_weights(self):
        """Dummy function."""
        pass

    def forward(self, x: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        """Forward method for a sequence of LPIPS feature loss tensors."""
        assert len(x) == len(self.weights)
        scores = [
            (weights * xi).sum(dim=1, keepdim=True)
            for xi, weights, C in zip(x, self.weights, self.channels)
        ]
        return scores

    def save(self, path: Path | str = Path("lpips_non_embedding.pth")) -> Path:
        """Save the model weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    def load(self, path: Path | str = Path("lpips_non_emedding.pth")):
        """Load the model weights."""
        path = Path(path)
        if path.exists():
            states = torch.load(path, weights_only=True)
            self.load_state_dict(states)
            self.eval()
        # else:
        #     raise ValueError(f"Model weights not found at {path}!")


class LPIPSLoss(Module):
    """LPIPS loss implementation."""

    def __init__(
        self,
        model: FeatureExtractor
        | Literal[
            "vgg16",
            "vgg",
            "alexnet",
            "squeezenet",
            "resnet18",
            "resnet",
            "convnext",
            "vit",
            "swinv2",
        ]
        | None = None,
        embedding: LPIPSEmbedding | bool | None = True,
        pretrained_weights: Path | str | None = None,
        finetune: bool = False,
        reduction: Callable | None = torch.mean,
        device: torch.device | None = None,
        spatial_average: bool = True,
    ):
        """Constructor.

        Args:
            model: Model from which to extract features; either pre-initialized or one of
              ["vgg16", "vgg", "alexnet", "squeezenet", "resnet18", "resnet",
               "convnext", "vit", "swinv2"].
            embedding: Linear embedding layers for feature mapping.
              An embedding can be passed directly, but needs to use as many layers as
              features are extracted from the inputs.
              If `True`, linear embedding layers are used to calculate the loss,
              if `False`, weighted sum across channel dimensions is used,
              if `None`, the raw latent least-squares are used.
            pretrained_weights: weights for the embedding layers to be loaded.
              If `None`, the embedding layers are randomly initialized.
            finetune: If `False`, freeze gradients of the embedding.
            reduction: Reduction function, e.g. `torch.mean` or `torch.sum`.
            device: Tensor allocation/computation device.
            spatial_average: If `True`, averages across image dimensions.
        """
        super().__init__()
        device = torch.get_default_device() if device is None else device
        if model is None:
            model = LPIPSVGG16()
        elif isinstance(model, str):
            match model.lower():
                case "vgg16" | "vgg":
                    model = LPIPSVGG16()
                case "alexnet":
                    model = LPIPSAlexNet()
                case "squeezenet":
                    model = LPIPSSqueezeNet()
                case "resnet18" | "resnet":
                    model = LPIPSResNet18()
                case "convnext":
                    model = LPIPSConvNeXt()
                case "vit":
                    model = LPIPSViT()
                case "swinv2":
                    model = LPIPSSwinV2()
                case _:
                    model = LPIPSVGG16()
        self.model = model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        if isinstance(embedding, bool) or embedding is None:
            embedding = (
                LPIPSEmbedding(self.model.feature_channels)
                if embedding
                else LPIPSNonEmbedding(self.model.feature_channels)
            )
        if pretrained_weights is not None:
            embedding.load(pretrained_weights)
        self.embedding = embedding.to(device)
        if not finetune and self.embedding:
            self.embedding.eval()
            # self.embedding.requires_grad_(False)
        else:
            self.embedding.train()
        self.spatial_average = spatial_average
        self.reduction = reduction if reduction is not None else torch.nn.Identity()

    def __repr__(self) -> str:
        """String representation of LPIPSLoss."""
        return (
            f"LPIPSLoss(model={self.model.__class__.__name__}, "
            f"embedding={self.embedding.__class__.__name__}, "
            f"spatial_avg={self.spatial_average})"
        )

    @singledispatchmethod
    @staticmethod
    def _normalize(
        x: Any,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """Normalize input tensor(s)."""
        raise NotImplementedError(
            "Input must be a `torch.Tensor` or `Sequence[torch.Tensor]`!"
        )

    @_normalize.register(torch.Tensor)
    @staticmethod
    def _normalize_tensor(
        x: torch.Tensor,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """Normalize input tensor."""
        norm = torch.sqrt(torch.sum(torch.pow(x, 2), dim=1, keepdim=True) + epsilon)
        return normalize(x, 0, norm)

    @_normalize.register(list)
    @_normalize.register(tuple)
    @staticmethod
    def _normalize_tensors(
        x: Sequence[torch.Tensor],
        epsilon: float = 1e-8,
    ) -> Sequence[torch.Tensor]:
        """Normalize input tensors."""
        norms = [
            torch.sqrt(torch.sum(torch.pow(xi, 2), dim=1, keepdim=True) + epsilon)
            for xi in x
        ]
        return [normalize(xi, 0, norm) for xi, norm in zip(x, norms)]

    @singledispatchmethod
    @staticmethod
    def _upsample(
        x: Any,
        size: Any,
        mode: Literal[
            "nearest",
            "linear",
            "bilinear",
            "bicubic",
            "trilinear",
            "area",
            "nearest-exact",
        ] = "bilinear",
        align_corners: bool | None = False,
        antialias: bool = False,
    ):
        """Upsample input tensor(s) to specified shape(s)."""
        raise NotImplementedError(
            "Input must be a `torch.Tensor` or `Sequence[torch.Tensor]`!"
        )

    @_upsample.register(torch.Tensor)
    @staticmethod
    def _upsample_tensor(
        x: torch.Tensor,
        size: tuple[int, int] = (64, 64),
        mode: Literal[
            "nearest",
            "linear",
            "bilinear",
            "bicubic",
            "trilinear",
            "area",
            "nearest-exact",
        ] = "bilinear",
        align_corners: bool | None = False,
        antialias: bool = False,
    ) -> torch.Tensor:
        """Upsample input tensor to specified shape."""
        return interpolate(
            x, size=size, mode=mode, align_corners=align_corners, antialias=antialias
        )

    @_upsample.register(list)
    @_upsample.register(tuple)
    @staticmethod
    def _upsample_tensors(
        x: Sequence[torch.Tensor],
        size: tuple[int, int] | Sequence[tuple[int, int]] = (64, 64),
        mode: Literal[
            "nearest",
            "linear",
            "bilinear",
            "bicubic",
            "trilinear",
            "area",
            "nearest-exact",
        ] = "bilinear",
        align_corners: bool | None = False,
        antialias: bool = False,
    ) -> Sequence[torch.Tensor]:
        """Upsample input tensors to specified shapes."""
        if size and isinstance(size, tuple) and isinstance(size[0], int):
            size = [size for _ in x]
        return [
            interpolate(
                xi, size=s, mode=mode, align_corners=align_corners, antialias=antialias
            )
            for xi, s in zip(x, size)
        ]

    @singledispatchmethod
    @staticmethod
    def _image_average(x: Any, keepdim: bool = True):
        """Spatially average image dimensions of tensor(s)."""
        raise NotImplementedError(
            "Input must be a `torch.Tensor` or `Sequence[torch.Tensor]`!"
        )

    @_image_average.register(torch.Tensor)
    @staticmethod
    def _image_average_tensor(x: torch.Tensor, keepdim: bool = True):
        """Spatially average image dimensions of tensor."""
        return x.mean((2, 3), keepdim=keepdim)

    @_image_average.register(list)
    @_image_average.register(tuple)
    @staticmethod
    def _image_average_tensor(x: Sequence[torch.Tensor], keepdim: bool = True):
        """Spatially average image dimensions of tensors."""
        return [xi.mean((2, 3), keepdim=keepdim) for xi in x]

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, as_scores: bool = False, **kwargs
    ) -> torch.Tensor | Sequence[torch.Tensor]:
        """Forward method."""
        if x1.shape != x2.shape:
            raise ValueError(f"Input shapes must match, got {x1.shape} and {x2.shape}")
        if x1.ndim == 5:
            sample = kwargs.get("sample", 0)
            x1 = as_batched_slices(x1, sample=sample)
            x2 = as_batched_slices(x2, sample=sample)
        reduction = kwargs.get("reduction", self.reduction)
        if reduction is None:
            reduction = torch.nn.Identity()
        x1, x2 = sanitize_ndim(x1), sanitize_ndim(x2)
        x1, x2 = as_tri_channel(x1), as_tri_channel(x2)
        w, h = x1.shape[2:]
        feat1, feat2 = self._normalize(self.model(x1)), self._normalize(self.model(x2))
        scores = [(f1 - f2) ** 2 for f1, f2 in zip(feat1, feat2)]
        scores = self.embedding(scores)
        scores = (
            self._upsample(scores, size=(w, h))
            if not self.spatial_average
            else self._image_average(scores)
        )
        if as_scores:
            return [reduction(s.squeeze()) for s in scores]
        loss = sum(scores).squeeze()
        return reduction(loss)
