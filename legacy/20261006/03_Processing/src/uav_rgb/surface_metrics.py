from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.measure import find_contours


@dataclass(frozen=True)
class OrientedPixelGeometry:
    """Deterministic minimum-area rectangle around a raster component.

    Coordinates use the source-image convention: ``x`` increases to the
    right and ``y`` increases downwards.  Lengths include the full footprint
    of each foreground pixel, so a component containing one pixel measures
    1 x 1 rather than 0 x 0.
    """

    center_x_px: float
    center_y_px: float
    length_px: float
    width_px: float
    major_dx: float
    major_dy: float
    minor_dx: float
    minor_dy: float
    angle_deg: float
    box_xy: tuple[tuple[float, float], ...]


def measure_oriented_pixel_geometry(
    submask: np.ndarray,
    offset_x: int | float = 0,
    offset_y: int | float = 0,
) -> OrientedPixelGeometry | None:
    """Measure a binary component with a deterministic oriented rectangle.

    The mask is doubled before extracting its padded 0.5-level contour.  At
    that resolution, even an isolated source pixel is a 2 x 2 plateau, whose
    contour represents the complete 1 x 1 source-pixel footprint.  This also
    avoids losing the outer half-pixel when ``submask`` is a tight component
    bounding box.
    """

    mask = np.asarray(submask)
    if mask.ndim != 2:
        raise ValueError("submask must be a two-dimensional array")
    mask = mask.astype(bool, copy=False)
    if not np.any(mask):
        return None

    # A direct 0.5-level contour of one foreground sample is a diamond.  Two
    # nearest-neighbour samples per source axis turn every source pixel into a
    # plateau and recover its square [-0.5, +0.5] footprint exactly.
    doubled = np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)
    padded = np.pad(doubled, pad_width=1, mode="constant", constant_values=False)
    contours = find_contours(
        padded.astype(np.uint8, copy=False),
        level=0.5,
        fully_connected="high",
    )
    if not contours:
        return None

    points: list[np.ndarray] = []
    for contour in contours:
        # Remove the one-sample padding, map doubled-grid sample centers back
        # to source-pixel centers, and finally restore the global image offset.
        x = (contour[:, 1] - 1.5) * 0.5 + float(offset_x)
        y = (contour[:, 0] - 1.5) * 0.5 + float(offset_y)
        points.append(np.column_stack((x, y)))
    contour_xy = np.ascontiguousarray(np.vstack(points), dtype=np.float32)

    rect = cv2.minAreaRect(contour_xy)
    raw_box = cv2.boxPoints(rect).astype(np.float64)
    box = _ordered_box(raw_box)

    first_axis = box[1] - box[0]
    second_axis = box[2] - box[1]
    first_length = float(np.linalg.norm(first_axis))
    second_length = float(np.linalg.norm(second_axis))
    if first_length <= 0.0 or second_length <= 0.0:
        # The doubled-pixel contour should make this unreachable for every
        # non-empty raster mask, but keep a clear failure mode for bad native
        # OpenCV output instead of returning NaNs.
        raise RuntimeError("minimum-area rectangle has a degenerate side")

    first_unit = _canonical_axis(first_axis / first_length)
    second_unit = _canonical_axis(second_axis / second_length)
    square_tolerance = max(first_length, second_length, 1.0) * 1e-6

    if abs(first_length - second_length) <= square_tolerance:
        # A square has no intrinsic major axis.  Select the box edge closest
        # to +x; at an exact angular tie, prefer the edge with non-negative y.
        # This makes the result independent of OpenCV's start corner and
        # width/height swap conventions while retaining the rectangle axis.
        first_key = _square_axis_key(first_unit)
        second_key = _square_axis_key(second_unit)
        if first_key <= second_key:
            major = first_unit
            minor = second_unit
        else:
            major = second_unit
            minor = first_unit
        length = max(first_length, second_length)
        width = min(first_length, second_length)
    elif first_length > second_length:
        major = first_unit
        minor = second_unit
        length = first_length
        width = second_length
    else:
        major = second_unit
        minor = first_unit
        length = second_length
        width = first_length

    center = np.mean(box, axis=0)
    angle_deg = math.degrees(math.atan2(float(major[1]), float(major[0])))
    return OrientedPixelGeometry(
        center_x_px=_clean(float(center[0])),
        center_y_px=_clean(float(center[1])),
        length_px=_clean(length),
        width_px=_clean(width),
        major_dx=_clean(float(major[0])),
        major_dy=_clean(float(major[1])),
        minor_dx=_clean(float(minor[0])),
        minor_dy=_clean(float(minor[1])),
        angle_deg=_clean(angle_deg),
        box_xy=tuple(
            (_clean(float(point[0])), _clean(float(point[1])))
            for point in box
        ),
    )


def _ordered_box(box: np.ndarray) -> np.ndarray:
    """Return corners clockwise in image coordinates, starting top-leftmost."""

    center = np.mean(box, axis=0)
    angles = np.arctan2(box[:, 1] - center[1], box[:, 0] - center[0])
    ordered = box[np.argsort(angles, kind="stable")]
    start = min(
        range(len(ordered)),
        key=lambda index: (
            round(float(ordered[index, 1]), 12),
            round(float(ordered[index, 0]), 12),
        ),
    )
    return np.roll(ordered, -start, axis=0)


def _canonical_axis(axis: np.ndarray) -> np.ndarray:
    """Give an unoriented rectangle axis one stable image-coordinate sign."""

    result = np.asarray(axis, dtype=np.float64).copy()
    norm = float(np.linalg.norm(result))
    if norm <= 0.0:
        raise ValueError("rectangle axis must be non-zero")
    result /= norm

    epsilon = 1e-7
    if abs(float(result[0])) <= epsilon:
        result[0] = 0.0
        if result[1] < 0.0:
            result *= -1.0
    elif result[0] < 0.0:
        result *= -1.0

    if abs(float(result[1])) <= epsilon:
        result[1] = 0.0
    return result / np.linalg.norm(result)


def _square_axis_key(axis: np.ndarray) -> tuple[float, int, float]:
    angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    return (round(abs(angle), 10), 0 if angle >= 0.0 else 1, -float(axis[0]))


def _clean(value: float) -> float:
    return 0.0 if abs(value) < 1e-10 else float(value)
