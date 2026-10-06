from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


ClassMasks = Mapping[int, np.ndarray]


def validate_class_masks(class_masks: ClassMasks) -> tuple[int, int]:
    """Validate the canonical multilabel representation and return ``(height, width)``.

    A class-mask mapping contains foreground classes only.  Every value is an
    independent boolean image, so pixels may belong to more than one class.
    """

    if not isinstance(class_masks, Mapping):
        raise TypeError("class_masks must be a mapping of positive class IDs to boolean HxW arrays")
    if not class_masks:
        raise ValueError("class_masks must contain at least one foreground class mask")

    shape: tuple[int, int] | None = None
    for raw_class_id, mask in class_masks.items():
        if not isinstance(raw_class_id, (int, np.integer)) or isinstance(raw_class_id, (bool, np.bool_)):
            raise TypeError("class mask keys must be integer class IDs")
        class_id = int(raw_class_id)
        if class_id <= 0:
            raise ValueError("class_masks contains foreground masks only; class IDs must be > 0")
        if not isinstance(mask, np.ndarray):
            raise TypeError(f"class mask {class_id} must be a numpy array")
        if mask.ndim != 2:
            raise ValueError(f"class mask {class_id} must have shape HxW, got {mask.shape}")
        if mask.dtype != np.bool_:
            raise TypeError(f"class mask {class_id} must have boolean dtype, got {mask.dtype}")
        current_shape = (int(mask.shape[0]), int(mask.shape[1]))
        if shape is None:
            shape = current_shape
        elif current_shape != shape:
            raise ValueError(
                f"all class masks must have the same HxW shape; expected {shape}, "
                f"class {class_id} has {current_shape}"
            )

    assert shape is not None
    return shape


def class_membership_count(class_masks: ClassMasks) -> np.ndarray:
    """Return how many foreground classes contain each pixel."""

    height, width = validate_class_masks(class_masks)
    memberships = np.zeros((height, width), dtype=np.uint16)
    for mask in class_masks.values():
        memberships += mask.astype(np.uint16, copy=False)
    return memberships


def summarize_class_masks(
    class_masks: ClassMasks,
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    """Summarize independent SAM masks without discarding class overlaps.

    Background is the complement of the union of all foreground masks.  A
    pixel in two damage masks is counted once in each damage class and never
    in background.
    """

    memberships = class_membership_count(class_masks)
    total = int(memberships.size)
    background = memberships == 0
    overlap = memberships > 1

    ordered_ids = list(class_names)
    ordered_ids.extend(sorted(class_id for class_id in class_masks if class_id not in class_names))
    if 0 not in ordered_ids:
        ordered_ids.insert(0, 0)

    rows: list[dict[str, Any]] = []
    for cls_id in ordered_ids:
        class_id = int(cls_id)
        if class_id == 0:
            mask = background
            cls_name = class_names.get(0, "BG")
        else:
            mask = class_masks.get(class_id)
            if mask is None:
                mask = np.zeros_like(background)
            cls_name = class_names.get(class_id, str(class_id))

        count = int(np.count_nonzero(mask))
        ratio = (count / total * 100.0) if total > 0 else 0.0
        rows.append(
            {
                "class_id": class_id,
                "class_name": cls_name,
                "pixel_count": count,
                "pixel_ratio": round(count / total, 8) if total > 0 else 0.0,
                "area_percent": round(ratio, 6),
                "overlap_pixel_count": int(np.count_nonzero(mask & overlap)) if class_id != 0 else 0,
                "mask_semantics": "multilabel_binary_union",
            }
        )
    return rows


def summarize_pixels(mask: np.ndarray, class_names: dict[int, str]) -> list[dict[str, Any]]:
    total = int(mask.size)
    rows: list[dict[str, Any]] = []
    for cls_id, cls_name in class_names.items():
        count = int((mask == cls_id).sum())
        ratio = (count / total * 100.0) if total > 0 else 0.0
        rows.append(
            {
                "class_id": int(cls_id),
                "class_name": cls_name,
                "pixel_count": count,
                "pixel_ratio": round(count / total, 8) if total > 0 else 0.0,
                "area_percent": round(ratio, 6),
            }
        )
    return rows


def foreground_ratio(mask: np.ndarray | ClassMasks) -> float:
    if isinstance(mask, Mapping):
        memberships = class_membership_count(mask)
        return float(np.count_nonzero(memberships) / memberships.size) if memberships.size else 0.0
    if mask.size == 0:
        return 0.0
    return float((mask != 0).sum() / mask.size)
