"""Display opacity must never alter mask pixels, source RGB, or class priority."""
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo512.data import CLASSES, CLASS_COLORS, OVERLAY_ALPHA, overlay_metadata, render_overlay


def fixture_images():
    rgb = Image.new("RGB", (4, 2), (60, 80, 100))
    masks = {label: Image.new("L", rgb.size, 0) for label in CLASSES}
    # Column 0: background; 1: DLM; 2: DLM+SPL; 3: all three.
    masks["DLM"].paste(255, (1, 0, 4, 2))
    masks["SPL"].paste(255, (2, 0, 4, 2))
    masks["CRC"].paste(255, (3, 0, 4, 2))
    return rgb, masks


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 1.0])
def test_blend_once_preserves_priority_background_inputs_and_masks(alpha):
    rgb, masks = fixture_images()
    source_before = rgb.tobytes()
    masks_before = {label: mask.tobytes() for label, mask in masks.items()}
    actual = render_overlay(rgb, masks, alpha=alpha)
    assert actual.mode == "RGB" and actual.size == rgb.size
    expected = np.array(rgb)
    for column, label in ((1, "DLM"), (2, "SPL"), (3, "CRC")):
        expected[:, column] = ((1 - alpha) * np.array([60, 80, 100])
                               + alpha * np.array(CLASS_COLORS[label])).astype(np.uint8)
    np.testing.assert_array_equal(np.array(actual), expected)
    assert rgb.tobytes() == source_before
    assert {label: mask.tobytes() for label, mask in masks.items()} == masks_before
    assert overlay_metadata(alpha)["alpha"] == alpha


def test_default_opacity_matches_metadata():
    rgb, masks = fixture_images()
    assert OVERLAY_ALPHA == overlay_metadata()["alpha"] == 0.5
    assert render_overlay(rgb, masks).tobytes() == render_overlay(rgb, masks, alpha=0.5).tobytes()


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_opacity_rejected(alpha):
    rgb, masks = fixture_images()
    with pytest.raises(ValueError, match="alpha"):
        render_overlay(rgb, masks, alpha=alpha)


def test_soft_mask_rejected_without_modifying_input():
    rgb, masks = fixture_images()
    masks["CRC"].putpixel((0, 0), 128)
    before = masks["CRC"].tobytes()
    with pytest.raises(ValueError, match="binary"):
        render_overlay(rgb, masks)
    assert masks["CRC"].tobytes() == before


def test_mismatched_mask_dimensions_rejected():
    rgb, masks = fixture_images()
    masks["CRC"] = Image.new("L", (1, 1), 0)
    with pytest.raises(ValueError, match="dimensions"):
        render_overlay(rgb, masks)
