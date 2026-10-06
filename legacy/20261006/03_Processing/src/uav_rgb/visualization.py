from __future__ import annotations

import numpy as np

from .metrics import ClassMasks, class_membership_count, validate_class_masks


def mask_to_color(mask: np.ndarray, palette: dict[int, tuple[int, int, int]]) -> np.ndarray:
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls_id, rgb in palette.items():
        color[mask == cls_id] = rgb
    return color


def make_overlay(image_rgb: np.ndarray, color_mask: np.ndarray, alpha: float = 0.80) -> np.ndarray:
    return np.clip(image_rgb * (1 - alpha) + color_mask * alpha, 0, 255).astype(np.uint8)


def class_masks_to_color(
    class_masks: ClassMasks,
    palette: dict[int, tuple[int, int, int]],
) -> np.ndarray:
    """Render independent masks, averaging colors where classes overlap."""

    height, width = validate_class_masks(class_masks)
    color_sum = np.zeros((height, width, 3), dtype=np.float32)
    color_count = np.zeros((height, width, 1), dtype=np.float32)
    for class_id, mask in class_masks.items():
        rgb = palette.get(int(class_id), (255, 255, 255))
        color_sum[mask] += np.asarray(rgb, dtype=np.float32)
        color_count[mask] += 1.0
    np.divide(color_sum, color_count, out=color_sum, where=color_count > 0)
    return np.clip(color_sum, 0, 255).astype(np.uint8)


def make_multilabel_overlay(
    image_rgb: np.ndarray,
    class_masks: ClassMasks,
    palette: dict[int, tuple[int, int, int]],
    alpha: float = 0.80,
) -> np.ndarray:
    """Blend damage colors at 80% opacity by default; keep background intact.

    Overlapping labels retain their averaged display color. This affects
    visualization only, never the independent masks or quantitative outputs.
    """

    height, width = validate_class_masks(class_masks)
    if image_rgb.shape != (height, width, 3):
        raise ValueError(
            f"image_rgb must have shape {(height, width, 3)} for these class masks, got {image_rgb.shape}"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    color_mask = class_masks_to_color(class_masks, palette)
    foreground = class_membership_count(class_masks) > 0
    overlay = np.asarray(image_rgb, dtype=np.uint8).copy()
    blended = np.clip(
        image_rgb[foreground].astype(np.float32) * (1.0 - alpha)
        + color_mask[foreground].astype(np.float32) * alpha,
        0,
        255,
    )
    overlay[foreground] = blended.astype(np.uint8)
    return overlay


def class_masks_to_channel_image(
    class_masks: ClassMasks,
    channel_class_ids: tuple[int, int, int] = (1, 2, 3),
) -> np.ndarray:
    """Encode three independent masks losslessly as RGB channels."""

    height, width = validate_class_masks(class_masks)
    encoded = np.zeros((height, width, 3), dtype=np.uint8)
    for channel, class_id in enumerate(channel_class_ids):
        mask = class_masks.get(int(class_id))
        if mask is not None:
            encoded[..., channel] = mask.astype(np.uint8) * 255
    return encoded
