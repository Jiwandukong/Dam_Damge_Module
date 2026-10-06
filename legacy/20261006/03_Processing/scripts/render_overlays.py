#!/usr/bin/env python3
"""Render source-resolution overlays from saved masks without model inference."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "03_Processing" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from uav_rgb.config import CrackSegConfig, load_config
from uav_rgb.io import list_images
from uav_rgb.visualization import make_multilabel_overlay


DEFAULT_CONFIG = PROJECT_ROOT / "03_Processing/configs/daechung_aug512.yaml"


def valid_alpha(value: str | float) -> float:
    """Validate mask opacity for both the command line and Python callers."""

    alpha = float(value)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and between 0 and 1")
    return alpha


def load_saved_masks(
    masks_dir: Path,
    image_path: Path,
    config: CrackSegConfig,
    shape_hw: tuple[int, int],
) -> dict[int, np.ndarray]:
    """Require original-size, single-channel binary PNGs for all three labels."""

    masks = {}
    for name, class_id in config.class_ids.items():
        path = masks_dir / image_path.stem / f"{name}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Required class mask not found: {path}")
        with Image.open(path) as opened:
            if opened.format != "PNG" or opened.mode not in {"1", "L"}:
                raise ValueError(f"Class mask must be a single-channel binary PNG: {path}")
            pixels = np.asarray(opened)
        if pixels.shape != shape_hw:
            raise ValueError(
                f"Class mask shape {pixels.shape} does not match source image "
                f"shape {shape_hw}: {path}"
            )
        if pixels.dtype != np.bool_ and not np.all((pixels == 0) | (pixels == 255)):
            raise ValueError(f"Class mask must contain only 0 and 255: {path}")
        masks[class_id] = pixels.astype(bool)
    return masks


def render_overlays(
    *,
    images: str | Path,
    masks_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path = DEFAULT_CONFIG,
    alpha: float | None = None,
) -> list[Path]:
    """Write only overlays into a new directory; never replace existing files.

    Input masks and photographs are read-only. The checkpoint, geometry, CSV,
    Excel, and original overlays are not loaded or modified. All masks are
    validated before creating the destination directory.
    """

    destination = Path(output_dir).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Output directory must not already exist: {destination}")
    config = load_config(config_path)
    opacity = valid_alpha(config.overlay_alpha if alpha is None else alpha)
    source_paths = list_images(images)
    if not source_paths:
        raise ValueError(f"No source images found: {images}")
    stems = [path.stem for path in source_paths]
    if len(set(stems)) != len(stems):
        raise ValueError("Source images must have unique stems to avoid output collisions")
    mask_root = Path(masks_dir)

    # Preflight the entire batch without holding all full-size images in RAM.
    for source in source_paths:
        with Image.open(source) as opened:
            width, height = opened.size
        load_saved_masks(mask_root, source, config, (height, width))

    destination.mkdir(parents=True, exist_ok=False)
    results = []
    for source in source_paths:
        # Match UAVRGBPipeline.run: Pillow RGB conversion, no EXIF transpose,
        # resizing, preprocessing, or model inference.
        with Image.open(source) as opened:
            image_rgb = np.asarray(opened.convert("RGB"))
        masks = load_saved_masks(mask_root, source, config, image_rgb.shape[:2])
        overlay = make_multilabel_overlay(image_rgb, masks, config.palette, alpha=opacity)
        output = destination / f"{source.stem}_overlay.png"
        with output.open("xb") as stream:
            Image.fromarray(overlay).save(stream, format="PNG")
        results.append(output)
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="One source image or an image directory.")
    parser.add_argument(
        "--masks-dir", required=True,
        help="Existing masks directory containing <image stem>/CRC.png, DLM.png, and SPL.png.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="New directory for PNG overlays; existing directories are never overwritten.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--alpha", type=valid_alpha, default=None,
        help="Mask opacity from 0 to 1; omitted uses visualization.alpha from the config.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = render_overlays(
        images=args.images,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        alpha=args.alpha,
    )
    for path in paths:
        print(path)
    print(f"Rendered {len(paths)} overlays. Masks and quantitative results were not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
