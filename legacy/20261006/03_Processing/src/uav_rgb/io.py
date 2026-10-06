from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


VALID_IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def read_image_rgb(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.dtype != np.uint8:
        max_value = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else float(np.nanmax(img))
        if max_value <= 0:
            return np.zeros_like(img, dtype=np.uint8)
        img = np.clip(img.astype(np.float32) / max_value * 255.0, 0, 255).astype(np.uint8)
    return img


def list_images(path: str | Path) -> list[Path]:
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() not in VALID_IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension: {p}")
        return [p]
    if not p.exists():
        raise FileNotFoundError(p)
    return sorted(
        x for x in p.rglob("*")
        if x.is_file() and x.suffix.lower() in VALID_IMAGE_EXTS
    )
