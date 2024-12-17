"""Learned Perceptual Image Patch Similarity.

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

from collections.abc import Sequence
from typing import Any, ClassVar, Optional, Union, Literal

import torch
from torch import Tensor

from torchmetrics.functional.image.lpips import (
    _LPIPS,
    _lpips_compute,
    _lpips_update,
    _NoTrainLpips,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TORCHVISION_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["LearnedPerceptualImagePatchSimilarity.plot"]

if _TORCHVISION_AVAILABLE:

    def _download_lpips() -> None:
        _LPIPS(pretrained=True, net="vgg")

    if _SKIP_SLOW_DOCTEST and not _try_proceed_with_timeout(_download_lpips):
        __doctest_skip__ = [
            "LearnedPerceptualImagePatchSimilarity",
            "LearnedPerceptualImagePatchSimilarity.plot",
        ]
else:
    __doctest_skip__ = [
        "LearnedPerceptualImagePatchSimilarity",
        "LearnedPerceptualImagePatchSimilarity.plot",
    ]


class LearnedPerceptualImagePatchSimilarity(Metric):
    """The Learned Perceptual Image Patch Similarity (`LPIPS_`) calculates perceptual similarity between two images.

    LPIPS essentially computes the similarity between the activations of two image patches for some pre-defined network.
    This measure has been shown to match human perception well. A low LPIPS score means that image patches are
    perceptual similar.

    Both input image patches are expected to have shape ``(N, 3, H, W)``. The minimum size of `H, W` depends on the
    chosen backbone (see `net_type` arg).

    .. hint::
        Using this metrics requires you to have ``torchvision`` package installed. Either install as
        ``pip install torchmetrics[image]`` or ``pip install torchvision``.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``img1`` (:class:`~torch.Tensor`): tensor with images of shape ``(N, C, H, W)`` or ``(N, C, D, H, W)``

    - ``img2`` (:class:`~torch.Tensor`): tensor with images of shape ``(N, C, H, W)`` or ``(N, C, D, H, W)``

    As output of `forward` and `compute` the metric returns the following output

    - ``lpips`` (:class:`~torch.Tensor`): returns float scalar tensor with average LPIPS value over samples

    Args:
        net_type: str indicating backbone network type to use. Choose between `'alex'`, `'vgg'` or `'squeeze'`
        reduction: str indicating how to reduce over the batch dimension. Choose between `'sum'` or `'mean'`.
        normalize: by default this is ``False`` meaning that the input is expected to be in the [-1,1] range. If set
            to ``True`` will instead expect input to be in the ``[0,1]`` range.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If ``torchvision`` package is not installed
        ValueError:
            If ``net_type`` is not one of ``"vgg"``, ``"alex"`` or ``"squeeze"``
        ValueError:
            If ``reduction`` is not one of ``"mean"`` or ``"sum"``

    Example:
        >>> from torch import rand
        >>> from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        >>> lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
        >>> # LPIPS needs the images to be in the [-1, 1] range.
        >>> img1 = (rand(10, 3, 100, 100) * 2) - 1
        >>> img2 = (rand(10, 3, 100, 100) * 2) - 1
        >>> lpips(img1, img2)
        tensor(0.1024)

    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    sum_scores: Tensor
    total: Tensor
    feature_network: str = "net"

    # due to the use of named tuple in the backbone the net variable cannot be scripted
    __jit_ignored_attributes__: ClassVar[list[str]] = ["net"]

    def __init__(
        self,
        net_type: Literal["vgg", "alex", "squeeze"] = "vgg",
        reduction: Literal["sum", "mean"] = "mean",
        normalize: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError(
                "LPIPS metric requires that torchvision is installed."
                " Either install as `pip install torchmetrics[image]` or `pip install torchvision`."
            )

        valid_net_type = ("vgg", "alex", "squeeze")
        if net_type not in valid_net_type:
            raise ValueError(
                f"Argument `net_type` must be one of {valid_net_type}, but got {net_type}."
            )
        self.net = _NoTrainLpips(net=net_type)

        valid_reduction = ("mean", "sum")
        if reduction not in valid_reduction:
            raise ValueError(
                f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}"
            )
        self.reduction = reduction

        if not isinstance(normalize, bool):
            raise ValueError(
                f"Argument `normalize` should be an bool but got {normalize}"
            )
        self.normalize = normalize

        self.add_state("sum_scores", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0.0), dist_reduce_fx="sum")

    def store_loss(self, img1: Tensor, img2: Tensor) -> None:
        loss, total = _lpips_update(img1, img2, net=self.net, normalize=self.normalize)
        self.sum_scores += loss.sum()
        self.total += total

    def update(self, img1: Tensor, img2: Tensor) -> None:
        """Update internal states with lpips score."""
        if img1.dim() < 4 or img1.dim() > 5:
            raise ValueError(
                "Expected input to be a 4D or 5D tensor with shape (B, C, H, W) or (B, C, D, H, W)"
            )

        if img1.dim() == 5:
            # make batches of 2D images from 3D images
            # batch size is the product of the first two dimensions
            # TODO make smaller batches if too large for GPU
            B, C, D, H, W = img1.shape
            img1 = img1.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            _B, _C, _D, _H, _W = img2.shape
            img2 = img2.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            if C == 1:
                img1 = img1.repeat(1, 3, 1, 1)
                img2 = img2.repeat(1, 3, 1, 1)
            self.store_loss(img1, img2)
        else:
            B, C, H, W = img1.shape
            if C == 1:
                img1 = img1.repeat(1, 3, 1, 1)
                img2 = img2.repeat(1, 3, 1, 1)
            self.store_loss(img1, img2)

    def compute(self) -> Tensor:
        """Compute final perceptual similarity metric."""
        return _lpips_compute(self.sum_scores, self.total, self.reduction)

    def plot(
        self,
        val: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        ax: Optional[_AX_TYPE] = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            >>> metric = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
            >>> metric.update(torch.rand(10, 3, 100, 100), torch.rand(10, 3, 100, 100))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            >>> metric = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
            >>> values = [ ]
            >>> for _ in range(3):
            ...     values.append(metric(torch.rand(10, 3, 100, 100), torch.rand(10, 3, 100, 100)))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
