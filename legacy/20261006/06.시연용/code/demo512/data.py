"""Image hashing and transparent damage overlays for inference outputs."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


CLASSES = ("CRC", "DLM", "SPL")
CLASS_COLORS = {"CRC": (0, 255, 0), "DLM": (0, 0, 255), "SPL": (255, 255, 0)}
# Last class wins only in the RGB visualization; binary masks remain independent.
OVERLAY_ORDER = ("DLM", "SPL", "CRC")
OVERLAY_ALPHA = 0.5


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def overlay_metadata(alpha: float = OVERLAY_ALPHA) -> dict[str, Any]:
    """Shared display-only styling; alpha is color opacity, not transparency."""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("overlay alpha must be a finite number in 0..1")
    return {
        "class_colors_rgb": {label: list(color) for label, color in CLASS_COLORS.items()},
        "paint_order": list(OVERLAY_ORDER),
        "overlap_rule": "last listed class wins in RGB overlay only; all binary masks and instances preserved",
        "alpha": float(alpha),
        "blending_rule": "select the top class color, then blend once with original RGB; background unchanged",
    }


def render_overlay(image: Image.Image, masks: dict[str, Image.Image],
                   alpha: float = OVERLAY_ALPHA) -> Image.Image:
    """Blend class colors once with RGB; CRC > SPL > DLM, masks unchanged.

    Alpha 0 leaves the original visible, 1 paints opaque class colors. Masks
    must be source-size binary images: display opacity never changes GT or
    prediction pixels, and overlapping classes do not compound the opacity.
    """
    overlay_metadata(alpha)
    original = image.convert("RGB")
    overlay = original.copy()
    for label in OVERLAY_ORDER:
        mask = masks[label]
        if mask.mode != "L" or mask.size != original.size:
            raise ValueError(f"{label} overlay mask must be an L-mode image matching RGB dimensions")
        values = np.asarray(mask)
        if not np.all((values == 0) | (values == 255)):
            raise ValueError(f"{label} overlay mask must contain only binary values 0 and 255")
        overlay.paste(CLASS_COLORS[label], (0, 0), mask)
    return Image.blend(original, overlay, float(alpha))
