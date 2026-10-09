"""Map the measured GPR layout in metres into world coordinates."""

import json
import math
from pathlib import Path

DEFAULT_TRANSFORM = Path(__file__).resolve().parents[1] / "Data/Calibration/transform.json"
LINE_Y_OFFSETS = {1: 0.0, 2: 0.8, 3: 1.2}
LINE_X_POSITIONS = {4: 1.0, 5: 4.0, 6: 5.5, 7: 9.0, 8: 12.0, 9: 15.0, 10: 18.0, 11: 21.0, 12: 25.0}
UNIT_REVIEW_STATUS = "line_surface_projection_unit_review"


def horizontal_frame(direction, across_hint):
    """Build orthonormal axes with an exactly horizontal along-line axis."""
    length = math.hypot(direction[0], direction[1])
    if length < 1e-9:
        raise ValueError("Horizontal survey direction is degenerate")
    along = [direction[0] / length, direction[1] / length, 0.0]
    component = sum(a * b for a, b in zip(across_hint, along))
    across = [across_hint[i] - component * along[i] for i in range(3)]
    length = math.hypot(*across)
    if length < 1e-9:
        raise ValueError("Across-line direction is degenerate")
    across = [x / length for x in across]
    normal = [along[1]*across[2]-along[2]*across[1],
              along[2]*across[0]-along[0]*across[2],
              along[0]*across[1]-along[1]*across[0]]
    return [[along[i], across[i], normal[i]] for i in range(3)]


def load_transform(path=DEFAULT_TRANSFORM):
    transform = json.loads(Path(path).read_text())
    scale = float(transform["scale"])
    rotation = transform["R"]
    translation = transform["t"]
    values = [scale, float(transform.get("y0", 0))] + [v for row in rotation for v in row] + translation
    if len(rotation) != 3 or any(len(row) != 3 for row in rotation) or len(translation) != 3:
        raise ValueError("Expected a 3x3 rotation and a three-component translation")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Coordinate transform contains non-finite values")
    if not math.isclose(scale, 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Metre-to-metre display requires scale=1; recompute calibration instead of compressing the measured layout")
    for i in range(3):
        for j in range(3):
            dot = sum(rotation[k][i] * rotation[k][j] for k in range(3))
            if not math.isclose(dot, float(i == j), rel_tol=0, abs_tol=1e-7):
                raise ValueError("Rotation must preserve measured distances")
    determinant = sum(rotation[0][i] * (rotation[1][(i+1)%3] * rotation[2][(i+2)%3]
                      - rotation[1][(i+2)%3] * rotation[2][(i+1)%3]) for i in range(3))
    if not math.isclose(determinant, 1, abs_tol=1e-7):
        raise ValueError("Expected a proper rotation, not a reflected coordinate frame")
    if abs(rotation[2][0]) > 1e-12:
        raise ValueError("Horizontal LINE_001~003 must preserve world Z; the along-line axis cannot tilt")
    horizontal_scale = float(transform.get('horizontal_distance_scale', 1.0))
    if not math.isfinite(horizontal_scale) or horizontal_scale <= 0:
        raise ValueError('Horizontal endpoint interpolation requires a finite positive distance ratio')
    return transform


def plane_position(line_no, along_m):
    line_no = int(line_no)
    along_m = float(along_m)
    if not math.isfinite(along_m) or along_m < 0:
        raise ValueError("Along-line distance must be finite and non-negative")
    if line_no in LINE_Y_OFFSETS:
        return [along_m, LINE_Y_OFFSETS[line_no]]
    if line_no in LINE_X_POSITIONS:
        return [LINE_X_POSITIONS[line_no], along_m]
    raise ValueError(f"Unknown survey line: {line_no}")


def map_plane(x_m, y_m, transform):
    plane = [float(x_m), float(y_m) + float(transform.get("y0", 0)), 0.0]
    return [float(transform["t"][i]) + float(transform["scale"]) *
            sum(float(transform["R"][i][j]) * plane[j] for j in range(3)) for i in range(3)]


def candidate_world(line_no, along_m, transform):
    return map_plane(*display_plane_position(line_no, along_m, transform), transform)


def display_plane_position(line_no, along_m, transform):
    """Fit horizontal along-scan distances while keeping vertical stations fixed."""
    x, y = plane_position(line_no, along_m)
    if int(line_no) in LINE_Y_OFFSETS:
        x *= float(transform.get('horizontal_distance_scale', 1.0))
    return [x, y]


def map_line_plane(line_no, x_m, y_m, transform):
    if int(line_no) in LINE_Y_OFFSETS:
        x_m = float(x_m) * float(transform.get('horizontal_distance_scale', 1.0))
    return map_plane(x_m, y_m, transform)


def mapping_status(transform):
    if transform.get("calibration_status") == "unit_scale_review":
        return UNIT_REVIEW_STATUS
    if transform.get("calibration_status") == "start_end_review":
        return "line_surface_projection_start_end_review"
    if transform.get('calibration_status') == 'selected_anchor_fit':
        return 'line_surface_projection_anchor_fit'
    return "line_surface_projection"
