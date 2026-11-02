# SPDX-FileCopyrightText: 2024-present Members of CAIIVS
# SPDX-FileNotice: Part of chuchichaestli
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the image exporter."""

import subprocess
import sys
import pytest
import torch
from torch.utils.data import TensorDataset
from chuchichaestli.utils.visualization.images import _to_unit, save_images


def test_a_lone_row_is_numbered_by_image(tmp_path):
    """One row has nothing to be told apart from, so only the image counts."""
    paths = save_images(torch.rand(3, 1, 8, 8), tmp_path / "s.png")
    assert [p.name for p in paths] == ["s_0.png", "s_1.png", "s_2.png"]


def test_a_dataset_of_pairs_gives_a_row_each(tmp_path):
    """What `Predict` publishes exports as predictions beside their truths."""
    pairs = TensorDataset(torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8))
    paths = save_images(pairs, tmp_path / "s.png", labels=("pred", "truth"))
    assert [p.name for p in paths] == [
        "s_pred_0.png",
        "s_pred_1.png",
        "s_truth_0.png",
        "s_truth_1.png",
    ]
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)


def test_an_unlabelled_row_is_numbered_by_position(tmp_path):
    """Without labels the row index keeps two rows' files apart."""
    pairs = TensorDataset(torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8))
    paths = save_images(pairs, tmp_path / "s.png")
    assert [p.name for p in paths] == [
        "s_0_0.png",
        "s_0_1.png",
        "s_1_0.png",
        "s_1_1.png",
    ]


def test_a_dataset_of_dicts_gives_a_row_per_key(tmp_path):
    """Datasets built with `return_as` hand out dicts, not tuples."""
    items = [{"x": torch.rand(1, 8, 8), "c": torch.rand(1, 8, 8)} for _ in range(1)]
    paths = save_images(items, tmp_path / "s.png")
    assert [p.name for p in paths] == ["s_x_0.png", "s_c_0.png"]


def test_a_mapping_names_its_rows(tmp_path):
    """Names given with the images need not be repeated as labels."""
    images = {"input": torch.rand(1, 1, 8, 8), "output": torch.rand(1, 1, 8, 8)}
    paths = save_images(images, tmp_path / "s.png")
    assert [p.name for p in paths] == ["s_input_0.png", "s_output_0.png"]


def test_a_label_is_made_safe_for_a_file_name(tmp_path):
    """A label reads as prose on the axes, but a file name cannot."""
    pairs = TensorDataset(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8))
    paths = save_images(pairs, tmp_path / "s.png", labels=("ground truth", "a/b"))
    assert [p.name for p in paths] == ["s_ground-truth_0.png", "s_a-b_0.png"]


def test_nothing_to_plot_says_so(tmp_path):
    """An empty batch is a mistake worth naming, not a blank file."""
    with pytest.raises(ValueError, match="at least one image"):
        save_images(torch.rand(0, 1, 8, 8), tmp_path / "s.png")


def test_there_must_be_a_label_per_row(tmp_path):
    """Labelling the wrong row is worse than not labelling at all."""
    with pytest.raises(ValueError, match="takes as many labels"):
        save_images(torch.rand(4, 1, 8, 8), tmp_path / "s.png", labels=("a", "b"))


def test_limit_caps_what_each_row_writes(tmp_path):
    """A file per image of a 1000-image validation set is not a preview."""
    pairs = TensorDataset(torch.rand(32, 1, 8, 8), torch.rand(32, 1, 8, 8))
    assert len(save_images(pairs, tmp_path / "s.png", limit=3)) == 6


@pytest.mark.parametrize("normalize", ["image", "batch", "shared", "none"])
def test_normalization_lands_in_the_unit_range(normalize):
    """Both backends are handed `[0, 1]`, whichever mode asked for it."""
    batch = _to_unit([torch.randn(4, 1, 8, 8) * 10], normalize)[0]
    assert float(batch.min()) >= 0.0 and float(batch.max()) <= 1.0


def test_an_explicit_range_scales_by_what_it_was_given():
    """A fixed range is what makes two exports comparable."""
    batch = _to_unit([torch.tensor([[[[0.0, 1.0, 2.0]]]])], (0.0, 2.0))[0]
    assert torch.allclose(batch, torch.tensor([[[[0.0, 0.5, 1.0]]]]))


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_plain_pixels_write_every_channel_count(channels, tmp_path):
    """Single channel, RGB and RGBA all come out as a file with pixels in it."""
    [path] = save_images(torch.rand(1, channels, 8, 8), tmp_path / "s.png")
    assert path.is_file() and path.stat().st_size > 0


def test_plain_pixels_need_no_plotting_backend(tmp_path):
    """Pixels are torchvision's job; matplotlib is only for more than pixels."""
    code = (
        "import sys, torch; "
        "from chuchichaestli.utils.visualization.images import save_images; "
        "save_images(torch.rand(2, 3, 8, 8), sys.argv[1]); "
        "print('matplotlib' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "s.png")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False"


def test_an_unknown_suffix_names_what_is_accepted(tmp_path):
    """As elsewhere in the package, the error lists the alternatives."""
    with pytest.raises(ValueError, match="Unsupported image format: '.gif'"):
        save_images(torch.rand(4, 1, 8, 8), tmp_path / "s.gif")


def test_an_unknown_normalization_names_what_is_accepted(tmp_path):
    """A typo in the mode should not silently write unscaled pixels."""
    with pytest.raises(ValueError, match="Unsupported image normalization"):
        save_images(torch.rand(4, 1, 8, 8), tmp_path / "s.png", normalize="best")


def test_labels_switch_to_the_figure_backend(tmp_path):
    """Only a figure can write an axis label, so asking for one picks it."""
    pytest.importorskip("matplotlib")
    pairs = TensorDataset(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8))
    paths = save_images(
        pairs, tmp_path / "s.png", labels=("sampled", "truth"), cmap="magma"
    )
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)


@pytest.mark.parametrize("suffix", [".pdf", ".svg"])
def test_a_vector_format_is_written_as_a_figure(suffix, tmp_path):
    """Raw pixels cannot be vectorized, so the figure backend takes over."""
    pytest.importorskip("matplotlib")
    [path] = save_images(torch.rand(1, 1, 8, 8), tmp_path / f"s{suffix}")
    assert path.is_file() and path.stat().st_size > 0


def test_a_closure_draws_the_image_itself(tmp_path):
    """The point of `draw`: the exporter opens the axes, the caller fills it."""
    pytest.importorskip("matplotlib")
    shapes: list[tuple[int, ...]] = []

    def draw(ax, image):
        shapes.append(tuple(image.shape))
        ax.contourf(image[0].numpy(), levels=3)

    save_images(torch.rand(2, 1, 8, 8), tmp_path / "s.png", draw=draw)
    assert shapes == [(1, 8, 8), (1, 8, 8)]


def test_a_closure_owns_its_axes_and_figure(tmp_path):
    """Labels, a legend and a colorbar are all the closure's to add."""
    pytest.importorskip("matplotlib")
    xlabels: list[str] = []

    def draw(ax, image):
        shown = ax.imshow(image[0].numpy())
        ax.set_xlabel("x [px]")
        ax.get_figure().colorbar(shown, ax=ax)
        xlabels.append(ax.get_xlabel())

    save_images(torch.rand(2, 1, 8, 8), tmp_path / "s.png", draw=draw)
    assert xlabels == ["x [px]", "x [px]"]


@pytest.mark.parametrize(("height", "width"), [(16, 48), (48, 16), (32, 32)])
def test_the_figure_takes_its_aspect_from_the_image(height, width, tmp_path):
    """A canvas of the image's own shape, not a square it sits adrift in."""
    pytest.importorskip("matplotlib")
    from PIL import Image

    [path] = save_images(
        torch.rand(1, 1, height, width), tmp_path / "s.png", cmap="magma", pad=0.0
    )
    pixels = Image.open(path).size
    assert pixels[0] / pixels[1] == pytest.approx(width / height, rel=0.02)


def test_the_longer_side_is_what_size_sets(tmp_path):
    """Mixed aspects still share a bounding box, so a folder of them lines up."""
    pytest.importorskip("matplotlib")
    from PIL import Image

    longest = []
    for height, width in [(16, 48), (48, 16)]:
        [path] = save_images(
            torch.rand(1, 1, height, width),
            tmp_path / f"s{height}.png",
            cmap="magma",
            size=2.0,
            pad=0.0,
            dpi=100,
        )
        longest.append(max(Image.open(path).size))
    assert longest == [200, 200]


def test_padding_widens_the_canvas(tmp_path):
    """Margin around the image is the caller's to set, in inches."""
    pytest.importorskip("matplotlib")
    from PIL import Image

    image = torch.rand(1, 1, 16, 16)
    [tight] = save_images(image, tmp_path / "a.png", cmap="magma", pad=0.0)
    [loose] = save_images(image, tmp_path / "b.png", cmap="magma", pad=0.25)
    assert Image.open(loose).size[0] > Image.open(tight).size[0]
