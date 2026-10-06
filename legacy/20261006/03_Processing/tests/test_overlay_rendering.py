from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from uav_rgb.config import load_config
from uav_rgb.visualization import class_masks_to_color, make_multilabel_overlay


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "03_Processing/configs/daechung_aug512.yaml"
SCRIPT_PATH = PROJECT_ROOT / "03_Processing/scripts/render_overlays.py"
SPEC = importlib.util.spec_from_file_location("render_overlays_for_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
RENDERER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDERER)


def example_arrays():
    image = np.array(
        [[[30, 60, 90], [100, 110, 120], [150, 160, 170]],
         [[70, 80, 90], [20, 40, 60], [180, 190, 200]]],
        dtype=np.uint8,
    )
    masks = {
        1: np.array([[False, True, False], [True, False, False]]),
        2: np.array([[False, False, True], [True, True, False]]),
        3: np.array([[False, False, True], [True, False, True]]),
    }
    return image, masks


def test_stronger_overlay_keeps_background_overlap_mean_and_inputs():
    config = load_config(CONFIG_PATH)
    assert config.overlay_alpha == 0.80
    image, masks = example_arrays()
    image_before = image.copy()
    masks_before = {key: value.copy() for key, value in masks.items()}
    colors = class_masks_to_color(masks, config.palette)
    assert colors[0, 2].tolist() == [127, 190, 127]  # DLM + SPL
    assert colors[1, 0].tolist() == [170, 148, 106]  # All three classes
    overlay = make_multilabel_overlay(image, masks, config.palette)
    foreground = np.logical_or.reduce(list(masks.values()))
    expected = image.copy()
    expected[foreground] = (
        image[foreground].astype(np.float32) * 0.20
        + colors[foreground].astype(np.float32) * 0.80
    ).astype(np.uint8)
    np.testing.assert_array_equal(overlay, expected)
    np.testing.assert_array_equal(overlay[~foreground], image[~foreground])
    np.testing.assert_array_equal(image, image_before)
    for class_id in masks:
        np.testing.assert_array_equal(masks[class_id], masks_before[class_id])


@pytest.mark.parametrize("alpha", [-0.01, 1.01, float("nan"), float("inf"), -float("inf")])
def test_overlay_and_helper_reject_invalid_opacity(alpha):
    image, masks = example_arrays()
    with pytest.raises(ValueError, match="alpha"):
        make_multilabel_overlay(image, masks, load_config(CONFIG_PATH).palette, alpha=alpha)
    with pytest.raises(ValueError, match="alpha"):
        RENDERER.valid_alpha(alpha)


@pytest.fixture
def saved_inputs(tmp_path):
    image, masks = example_arrays()
    images = tmp_path / "images"
    images.mkdir()
    image_path = images / "sample.png"
    Image.fromarray(image).save(image_path)
    mask_root = tmp_path / "masks"
    mask_dir = mask_root / image_path.stem
    mask_dir.mkdir(parents=True)
    for name, class_id in load_config(CONFIG_PATH).class_ids.items():
        Image.fromarray(masks[class_id].astype(np.uint8) * 255).save(mask_dir / f"{name}.png")
    return image_path, mask_root, image, masks


@pytest.mark.parametrize("alpha", [None, 0.0, 0.65, 1.0])
@pytest.mark.parametrize("use_directory", [False, True])
def test_helper_renders_exactly_without_changing_sources(saved_inputs, tmp_path, alpha, use_directory):
    image_path, mask_root, image, masks = saved_inputs
    source_files = [image_path, *sorted(mask_root.rglob("*.png"))]
    original_bytes = {path: path.read_bytes() for path in source_files}
    destination = tmp_path / "new_overlays"
    outputs = RENDERER.render_overlays(
        images=image_path.parent if use_directory else image_path,
        masks_dir=mask_root,
        output_dir=destination,
        config_path=CONFIG_PATH,
        alpha=alpha,
    )
    assert outputs == [destination / "sample_overlay.png"]
    assert list(destination.iterdir()) == outputs
    config = load_config(CONFIG_PATH)
    expected = make_multilabel_overlay(
        image, masks, config.palette,
        alpha=config.overlay_alpha if alpha is None else alpha,
    )
    with Image.open(outputs[0]) as opened:
        assert opened.size == (3, 2)
        np.testing.assert_array_equal(np.asarray(opened), expected)
    assert all(path.read_bytes() == original_bytes[path] for path in source_files)


def test_helper_refuses_existing_output_directory(saved_inputs, tmp_path):
    image_path, mask_root, _, _ = saved_inputs
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "sample_overlay.png"
    sentinel.write_bytes(b"existing original overlay")
    with pytest.raises(FileExistsError, match="must not already exist"):
        RENDERER.render_overlays(images=image_path, masks_dir=mask_root, output_dir=destination)
    assert sentinel.read_bytes() == b"existing original overlay"


@pytest.mark.parametrize("problem", ["missing", "rgb", "nonbinary", "shape"])
def test_helper_preflights_bad_masks_before_creating_output(saved_inputs, tmp_path, problem):
    image_path, mask_root, _, _ = saved_inputs
    mask_path = mask_root / "sample" / "DLM.png"
    if problem == "missing":
        mask_path.unlink()
        error = FileNotFoundError
    else:
        if problem == "rgb":
            pixels = np.zeros((2, 3, 3), dtype=np.uint8)
        elif problem == "nonbinary":
            pixels = np.full((2, 3), 128, dtype=np.uint8)
        else:
            pixels = np.zeros((1, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(mask_path)
        error = ValueError
    destination = tmp_path / "new_overlays"
    with pytest.raises(error):
        RENDERER.render_overlays(images=image_path, masks_dir=mask_root, output_dir=destination)
    assert not destination.exists()


def test_helper_rejects_duplicate_image_stems(saved_inputs, tmp_path):
    image_path, mask_root, image, _ = saved_inputs
    Image.fromarray(image).save(image_path.with_suffix(".jpg"))
    destination = tmp_path / "new_overlays"
    with pytest.raises(ValueError, match="unique stems"):
        RENDERER.render_overlays(images=image_path.parent, masks_dir=mask_root, output_dir=destination)
    assert not destination.exists()


def test_helper_rejects_invalid_alpha_before_creating_output(saved_inputs, tmp_path):
    image_path, mask_root, _, _ = saved_inputs
    destination = tmp_path / "new_overlays"
    with pytest.raises(ValueError, match="alpha"):
        RENDERER.render_overlays(
            images=image_path, masks_dir=mask_root, output_dir=destination, alpha=float("nan")
        )
    assert not destination.exists()
