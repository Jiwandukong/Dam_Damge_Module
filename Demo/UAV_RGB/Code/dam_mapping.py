"""Native gridded glTF camera-ray mapping and existing local surface measurements.
Coordinates: EPSG:5186 X/Y, altitude Z; native glTF axes X/Z/-Y.
Camera and local directional GSD formulas are reused from the existing pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json
import math
import re
import cv2
import numpy as np
import warp as wp
from PIL import Image
from skimage.measure import find_contours

wp.config.enable_cuda = False

SURFACE_MEASUREMENT_METHOD = 'mesh_ray_local_directional_gsd_min_area_rectangle'

GSD_REVIEW_RATIO = 10.0
@wp.kernel
def _mesh_raycast_kernel(
    mesh_id: wp.uint64,
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    hit_flags: wp.array(dtype=wp.int32),
    hit_t: wp.array(dtype=wp.float32),
    hit_face: wp.array(dtype=wp.int32),
    hit_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    query = wp.mesh_query_ray(mesh_id, origins[tid], directions[tid], 1.0e6)
    if query.result:
        hit_flags[tid] = 1
        hit_t[tid] = query.t
        hit_face[tid] = query.face
        hit_points[tid] = origins[tid] + directions[tid] * query.t
    else:
        hit_flags[tid] = 0
        hit_t[tid] = -1.0
        hit_face[tid] = -1
        hit_points[tid] = wp.vec3(0.0, 0.0, 0.0)

class WarpMeshRaycaster:
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        device: str = "cuda:0",
        origin_offset: np.ndarray | None = None,
    ):
        wp.init()
        if str(device).startswith("cuda") and not wp.is_cuda_available():
            raise RuntimeError("Warp CUDA device was requested, but CUDA is not available.")

        self.device = str(device)
        vertices_f64 = np.asarray(vertices, dtype=np.float64)
        if origin_offset is None:
            origin_offset = 0.5 * (vertices_f64.min(axis=0) + vertices_f64.max(axis=0))
        self.origin_offset = np.asarray(origin_offset, dtype=np.float64)

        vertices_f32 = (vertices_f64 - self.origin_offset).astype(np.float32)
        faces_i32 = np.asarray(faces, dtype=np.int32).reshape(-1)
        self.points = wp.array(vertices_f32, dtype=wp.vec3, device=self.device)
        self.indices = wp.array(faces_i32, dtype=wp.int32, device=self.device)
        constructor = "lbvh" if self.device.startswith("cuda") else "sah"
        self.mesh = wp.Mesh(
            points=self.points,
            velocities=None,
            indices=self.indices,
            bvh_constructor=constructor,
        )

    def intersect_rays(self, origins: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        origins_f32 = (np.asarray(origins, dtype=np.float64) - self.origin_offset).astype(np.float32)
        directions_f32 = np.asarray(directions, dtype=np.float32)
        n = int(len(origins_f32))
        if n == 0:
            return (
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0, 3), dtype=np.float32),
            )

        origins_wp = wp.array(origins_f32, dtype=wp.vec3, device=self.device)
        directions_wp = wp.array(directions_f32, dtype=wp.vec3, device=self.device)
        hit_flags = wp.empty(n, dtype=wp.int32, device=self.device)
        hit_t = wp.empty(n, dtype=wp.float32, device=self.device)
        hit_face = wp.empty(n, dtype=wp.int32, device=self.device)
        hit_points = wp.empty(n, dtype=wp.vec3, device=self.device)

        wp.launch(
            kernel=_mesh_raycast_kernel,
            dim=n,
            inputs=[self.mesh.id, origins_wp, directions_wp, hit_flags, hit_t, hit_face, hit_points],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        points = hit_points.numpy().astype(np.float64) + self.origin_offset
        return hit_flags.numpy(), hit_t.numpy(), hit_face.numpy(), points

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": "warp",
            "device": self.device,
            "origin_offset": tuple(float(v) for v in self.origin_offset),
        }
@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    focal_length_px: float
    cx: float
    cy: float
    dewarp_data: str | None = None

    def ray_camera(self, pixel_x: float, pixel_y: float) -> np.ndarray:
        x = (float(pixel_x) - self.cx) / self.focal_length_px
        y = (float(pixel_y) - self.cy) / self.focal_length_px
        ray = np.array([x, y, 1.0], dtype=np.float64)
        return ray / np.linalg.norm(ray)

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "focal_length_px": self.focal_length_px,
            "cx": self.cx,
            "cy": self.cy,
            "dewarp_data": self.dewarp_data,
            "distortion_applied": False,
            "distortion_note": (
                "DJI DewarpData is recorded but this rough mapper uses calibrated "
                "pinhole rays without lens-distortion correction."
            ),
        }

@dataclass(frozen=True)
class CameraPose:
    source_path: str
    center_xyz: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    position_source: str
    orientation_source: str
    rtk_std_lon_m: float | None = None
    rtk_std_lat_m: float | None = None
    rtk_std_hgt_m: float | None = None

    def ray_world(self, camera_ray: np.ndarray) -> np.ndarray:
        rotation = camera_to_world_matrix(self.yaw_deg, self.pitch_deg, self.roll_deg)
        ray = rotation @ camera_ray
        return ray / np.linalg.norm(ray)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "center_x_m": self.center_xyz[0],
            "center_y_m": self.center_xyz[1],
            "center_z_m": self.center_xyz[2],
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "position_source": self.position_source,
            "orientation_source": self.orientation_source,
            "horizontal_crs": "EPSG:5186",
            "coordinate_axes": "X=easting, Y=northing, Z=altitude; metres; Z-up",
            "vertical_datum_verified": False,
            "vertical_datum_note": (
                "Timestamp.MRK Ellh or XMP AbsoluteAltitude is used directly. "
                "Its vertical datum has not been verified against the OBJ Z datum."
            ),
            "rtk_std_lon_m": self.rtk_std_lon_m,
            "rtk_std_lat_m": self.rtk_std_lat_m,
            "rtk_std_hgt_m": self.rtk_std_hgt_m,
            "orientation_note": (
                "DJI gimbal yaw/pitch/roll are interpreted as optical-axis "
                "orientation in the EPSG:5186/Z-up world frame."
            ),
        }

def read_dji_xmp(path: str | Path) -> dict[str, str]:
    data = Path(path).read_bytes()
    start = data.find(b"<x:xmpmeta")
    end = data.find(b"</x:xmpmeta>")
    if start < 0 or end < 0:
        raise ValueError(f"DJI XMP metadata was not found in {path}")
    text = data[start : end + len(b"</x:xmpmeta>")].decode(
        "utf-8", errors="ignore"
    )
    return {
        key: value
        for key, value in re.findall(
            r'drone-dji:([A-Za-z0-9_]+)="([^"]*)"', text
        )
    }

def intrinsics_from_xmp(
    image_path: str | Path, xmp: dict[str, str]
) -> CameraIntrinsics:
    with Image.open(image_path) as image:
        width, height = image.size
    try:
        focal = float(xmp["CalibratedFocalLength"])
        cx = float(xmp["CalibratedOpticalCenterX"])
        cy = float(xmp["CalibratedOpticalCenterY"])
    except KeyError as exc:
        raise ValueError(
            f"Missing calibrated camera intrinsic in DJI XMP: {exc}"
        ) from exc
    return CameraIntrinsics(
        width=int(width),
        height=int(height),
        focal_length_px=focal,
        cx=cx,
        cy=cy,
        dewarp_data=xmp.get("DewarpData"),
    )

def pose_from_xmp(
    image_path: str | Path,
    xmp: dict[str, str],
    mrk_path: str | Path | None = None,
) -> CameraPose:
    lat = float(xmp["GpsLatitude"])
    lon = float(xmp["GpsLongitude"])
    alt = float(xmp["AbsoluteAltitude"])
    position_source = "jpg_xmp"
    if mrk_path is not None:
        photo_index = image_index_from_name(Path(image_path).name)
        mrk_positions = read_mrk_positions(mrk_path)
        if photo_index in mrk_positions:
            lat, lon, alt = mrk_positions[photo_index]
            position_source = "timestamp_mrk"

    center_x, center_y = wgs84_to_epsg5186(lon, lat)
    yaw = float(xmp.get("GimbalYawDegree", xmp.get("FlightYawDegree", 0.0)))
    pitch = float(
        xmp.get("GimbalPitchDegree", xmp.get("FlightPitchDegree", 0.0))
    )
    roll = float(xmp.get("GimbalRollDegree", xmp.get("FlightRollDegree", 0.0)))
    return CameraPose(
        source_path=Path(image_path).name,
        center_xyz=(float(center_x), float(center_y), float(alt)),
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        position_source=position_source,
        orientation_source="jpg_xmp_gimbal",
        rtk_std_lon_m=parse_optional_float(xmp.get("RtkStdLon")),
        rtk_std_lat_m=parse_optional_float(xmp.get("RtkStdLat")),
        rtk_std_hgt_m=parse_optional_float(xmp.get("RtkStdHgt")),
    )

def read_mrk_positions(path: str | Path) -> dict[int, tuple[float, float, float]]:
    positions: dict[int, tuple[float, float, float]] = {}
    for line in Path(path).read_text(
        encoding="utf-8", errors="ignore"
    ).splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) < 12:
            continue
        try:
            index = int(parts[0])
            lat_i = parts.index("Lat")
            lon_i = parts.index("Lon")
            ellh_i = parts.index("Ellh")
            positions[index] = (
                float(parts[lat_i - 1]),
                float(parts[lon_i - 1]),
                float(parts[ellh_i - 1]),
            )
        except (ValueError, IndexError):
            continue
    return positions

def image_index_from_name(name: str) -> int:
    match = re.search(r"_(\d{4})_", name)
    if match is None:
        raise ValueError(f"Could not parse DJI image index from {name}")
    return int(match.group(1))

def wgs84_to_epsg5186(lon: float, lat: float) -> tuple[float, float]:
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise ImportError("pyproj is required for EPSG:5186 conversion") from exc
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5186", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return float(x), float(y)

def camera_to_world_matrix(
    yaw_deg: float, pitch_deg: float, roll_deg: float
) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    forward = np.array(
        [
            math.cos(pitch) * math.sin(yaw),
            math.cos(pitch) * math.cos(yaw),
            math.sin(pitch),
        ],
        dtype=np.float64,
    )
    forward /= np.linalg.norm(forward)
    right = np.array([math.cos(yaw), -math.sin(yaw), 0.0], dtype=np.float64)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    if abs(roll) > 1e-12:
        cosine = math.cos(roll)
        sine = math.sin(roll)
        right, down = (
            right * cosine + down * sine,
            -right * sine + down * cosine,
        )
    return np.column_stack([right, down, forward])

def parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
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
@dataclass(frozen=True)
class SurfaceMeasurementStencil:
    """Four mesh-ray samples used to estimate local directional pixel scale."""

    center_xy: tuple[float, float]
    xy: np.ndarray
    length_span_px: float
    width_span_px: float
    length_scheme: str
    width_scheme: str

def build_surface_measurement_stencil(
    geometry: OrientedPixelGeometry,
    *,
    center_xy: np.ndarray,
    image_width: int,
    image_height: int,
    half_span_px: float = 2.0,
) -> SurfaceMeasurementStencil | None:
    """Build bounded samples along the instance's rotated major/minor axes.

    A symmetric four-pixel baseline is preferred for mesh stability.  The
    resulting 3D chord is divided by its exact image-pixel span, so the output
    remains a one-pixel GSD.  Near an image edge, a one-sided baseline is used.
    """

    center = np.asarray(center_xy, dtype=np.float64)
    if center.shape != (2,):
        raise ValueError("center_xy must contain exactly x and y")
    if image_width < 1 or image_height < 1:
        raise ValueError("image dimensions must be positive")
    if not np.isfinite(center).all():
        raise ValueError("center_xy must be finite")
    if not 0.0 <= float(center[0]) <= float(image_width - 1):
        return None
    if not 0.0 <= float(center[1]) <= float(image_height - 1):
        return None
    half_span = float(half_span_px)
    if not np.isfinite(half_span) or half_span <= 0.0:
        raise ValueError("half_span_px must be a positive finite value")

    major = np.array([geometry.major_dx, geometry.major_dy], dtype=np.float64)
    minor = np.array([geometry.minor_dx, geometry.minor_dy], dtype=np.float64)
    length_segment = _bounded_axis_segment(
        center,
        major,
        image_width=image_width,
        image_height=image_height,
        half_span_px=half_span,
    )
    width_segment = _bounded_axis_segment(
        center,
        minor,
        image_width=image_width,
        image_height=image_height,
        half_span_px=half_span,
    )
    if length_segment is None or width_segment is None:
        return None

    length_start, length_end, length_span, length_scheme = length_segment
    width_start, width_end, width_span, width_scheme = width_segment
    return SurfaceMeasurementStencil(
        center_xy=(float(center[0]), float(center[1])),
        xy=np.vstack((length_start, length_end, width_start, width_end)),
        length_span_px=float(length_span),
        width_span_px=float(width_span),
        length_scheme=length_scheme,
        width_scheme=width_scheme,
    )

def surface_measurement_from_hits(
    instance_row: Mapping[str, Any],
    geometry: OrientedPixelGeometry | None,
    stencil: SurfaceMeasurementStencil | None,
    hits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert four local mesh hits into approximate instance dimensions."""

    if geometry is None:
        return empty_surface_measurement(instance_row, "oriented_geometry_unavailable")
    if stencil is None:
        return empty_surface_measurement(
            instance_row,
            "measurement_stencil_outside_image",
            geometry=geometry,
        )
    if len(hits) != 4:
        return empty_surface_measurement(
            instance_row,
            "measurement_ray_count_mismatch",
            geometry=geometry,
            ray_count=len(hits),
            hit_count=sum(bool(hit.get("mesh_ray_hit")) for hit in hits),
        )

    hit_count = sum(bool(hit.get("mesh_ray_hit")) for hit in hits)
    points = [_mesh_hit_point(hit) for hit in hits]
    length_vector = None
    width_vector = None
    if points[0] is not None and points[1] is not None:
        length_vector = (points[1] - points[0]) / stencil.length_span_px
    if points[2] is not None and points[3] is not None:
        width_vector = (points[3] - points[2]) / stencil.width_span_px

    gsd_length = _positive_norm_or_none(length_vector)
    gsd_width = _positive_norm_or_none(width_vector)
    gsd_area = None
    if length_vector is not None and width_vector is not None:
        candidate_area = float(np.linalg.norm(np.cross(length_vector, width_vector)))
        if np.isfinite(candidate_area) and candidate_area > 0.0:
            gsd_area = candidate_area

    is_crack = int(instance_row.get("class_id", -1)) == 1
    valid = (
        gsd_length is not None and gsd_width is not None
        if is_crack
        else gsd_area is not None
    )
    missing: list[str] = []
    if gsd_length is None:
        missing.append("length_axis_mesh_hits_invalid")
    if gsd_width is None:
        missing.append("width_axis_mesh_hits_invalid")
    if not is_crack and gsd_area is None and not missing:
        missing.append("surface_jacobian_area_invalid")

    length_px = float(geometry.length_px) if is_crack else None
    width_px = float(geometry.width_px) if is_crack else None
    length_m = length_px * gsd_length if length_px is not None and gsd_length is not None else None
    width_m = width_px * gsd_width if width_px is not None and gsd_width is not None else None
    area_m2 = (
        float(instance_row["area_px"]) * gsd_area
        if not is_crack and gsd_area is not None
        else None
    )
    schemes = {stencil.length_scheme, stencil.width_scheme}
    quality = (
        "local_central_difference"
        if valid and schemes == {"central"}
        else "local_edge_adjusted"
        if valid
        else "invalid"
    )
    return {
        "length_px": round(length_px, 6) if length_px is not None else None,
        "length_m": round(length_m, 8) if length_m is not None else None,
        "width_px": round(width_px, 6) if width_px is not None else None,
        "width_m": round(width_m, 8) if width_m is not None else None,
        "area_m2": round(area_m2, 10) if area_m2 is not None else None,
        "area_m2_source": (
            "mesh_ray_local_surface_jacobian" if area_m2 is not None else None
        ),
        "gsd_length_m_per_px": (
            round(gsd_length, 10) if gsd_length is not None else None
        ),
        "gsd_width_m_per_px": (
            round(gsd_width, 10) if gsd_width is not None else None
        ),
        "gsd_area_m2_per_px": round(gsd_area, 12) if gsd_area is not None else None,
        "measurement_angle_deg": round(float(geometry.angle_deg), 6) if is_crack else None,
        "measurement_center_x_px": round(float(stencil.center_xy[0]), 4),
        "measurement_center_y_px": round(float(stencil.center_xy[1]), 4),
        "measurement_box_xy_json": json.dumps(
            [[round(x, 4), round(y, 4)] for x, y in geometry.box_xy],
            separators=(",", ":"),
        ),
        "measurement_method": SURFACE_MEASUREMENT_METHOD,
        "measurement_quality": quality,
        "measurement_valid": bool(valid),
        "measurement_is_approximate": True,
        "measurement_ray_count": 4,
        "measurement_hit_count": int(hit_count),
        "measurement_hit_ratio": round(float(hit_count / 4.0), 6),
        "measurement_length_baseline_px": round(float(stencil.length_span_px), 6),
        "measurement_width_baseline_px": round(float(stencil.width_span_px), 6),
        "measurement_length_stencil": stencil.length_scheme,
        "measurement_width_stencil": stencil.width_scheme,
        "measurement_miss_reason": None if valid else ";".join(missing),
    }

def empty_surface_measurement(
    instance_row: Mapping[str, Any],
    reason: str,
    *,
    geometry: OrientedPixelGeometry | None = None,
    ray_count: int = 0,
    hit_count: int = 0,
) -> dict[str, Any]:
    is_crack = int(instance_row.get("class_id", -1)) == 1
    length_px = (
        float(geometry.length_px)
        if is_crack and geometry is not None
        else instance_row.get("length_px")
        if is_crack
        else None
    )
    width_px = (
        float(geometry.width_px)
        if is_crack and geometry is not None
        else instance_row.get("width_px")
        if is_crack
        else None
    )
    return {
        "length_px": round(float(length_px), 6) if length_px is not None else None,
        "length_m": None,
        "width_px": round(float(width_px), 6) if width_px is not None else None,
        "width_m": None,
        "area_m2": None,
        "area_m2_source": None,
        "gsd_length_m_per_px": None,
        "gsd_width_m_per_px": None,
        "gsd_area_m2_per_px": None,
        "measurement_angle_deg": (
            round(float(geometry.angle_deg), 6)
            if is_crack and geometry is not None
            else instance_row.get("measurement_angle_deg")
            if is_crack
            else None
        ),
        "measurement_center_x_px": None,
        "measurement_center_y_px": None,
        "measurement_box_xy_json": (
            json.dumps(
                [[round(x, 4), round(y, 4)] for x, y in geometry.box_xy],
                separators=(",", ":"),
            )
            if geometry is not None
            else "[]"
        ),
        "measurement_method": SURFACE_MEASUREMENT_METHOD,
        "measurement_quality": "invalid",
        "measurement_valid": False,
        "measurement_is_approximate": True,
        "measurement_ray_count": int(ray_count),
        "measurement_hit_count": int(hit_count),
        "measurement_hit_ratio": (
            round(float(hit_count / ray_count), 6) if ray_count > 0 else 0.0
        ),
        "measurement_length_baseline_px": None,
        "measurement_width_baseline_px": None,
        "measurement_length_stencil": None,
        "measurement_width_stencil": None,
        "measurement_miss_reason": str(reason),
    }

def _bounded_axis_segment(
    center: np.ndarray,
    direction: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    half_span_px: float,
) -> tuple[np.ndarray, np.ndarray, float, str] | None:
    axis = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    axis /= norm
    positive_limit = _image_travel_limit(
        center,
        axis,
        image_width=image_width,
        image_height=image_height,
    )
    negative_limit = _image_travel_limit(
        center,
        -axis,
        image_width=image_width,
        image_height=image_height,
    )
    half_span = float(half_span_px)
    if positive_limit >= half_span and negative_limit >= half_span:
        t_start, t_end, scheme = -half_span, half_span, "central"
    elif positive_limit >= 2.0 * half_span:
        t_start, t_end, scheme = 0.0, 2.0 * half_span, "forward"
    elif negative_limit >= 2.0 * half_span:
        t_start, t_end, scheme = -2.0 * half_span, 0.0, "backward"
    else:
        t_start = -min(negative_limit, half_span)
        t_end = min(positive_limit, half_span)
        scheme = "clipped"
    span = float(t_end - t_start)
    if not np.isfinite(span) or span <= 1e-9:
        return None
    lower = np.array([0.0, 0.0], dtype=np.float64)
    upper = np.array([float(image_width - 1), float(image_height - 1)], dtype=np.float64)
    start = np.clip(center + t_start * axis, lower, upper)
    end = np.clip(center + t_end * axis, lower, upper)
    actual_span = float(np.dot(end - start, axis))
    if not np.isfinite(actual_span) or actual_span <= 1e-9:
        return None
    return start, end, actual_span, scheme

def _image_travel_limit(
    center: np.ndarray,
    direction: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> float:
    limits: list[float] = []
    for coordinate, delta, upper in (
        (float(center[0]), float(direction[0]), float(image_width - 1)),
        (float(center[1]), float(direction[1]), float(image_height - 1)),
    ):
        if delta > 1e-12:
            limits.append((upper - coordinate) / delta)
        elif delta < -1e-12:
            limits.append(coordinate / -delta)
    return max(0.0, min(limits)) if limits else float("inf")

def _mesh_hit_point(hit: Mapping[str, Any]) -> np.ndarray | None:
    if not hit.get("mesh_ray_hit"):
        return None
    try:
        point = np.array(
            [hit["world_x_m"], hit["world_y_m"], hit["world_z_m"]],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None
    return point if np.isfinite(point).all() else None

def _positive_norm_or_none(vector: np.ndarray | None) -> float | None:
    if vector is None:
        return None
    value = float(np.linalg.norm(vector))
    return value if np.isfinite(value) and value > 0.0 else None

def pixels_to_world_rays(intrinsics: CameraIntrinsics, pose: CameraPose, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    camera_rays = np.empty((len(xy), 3), dtype=np.float64)
    camera_rays[:, 0] = (xy[:, 0] - intrinsics.cx) / intrinsics.focal_length_px
    camera_rays[:, 1] = (xy[:, 1] - intrinsics.cy) / intrinsics.focal_length_px
    camera_rays[:, 2] = 1.0
    camera_rays /= np.linalg.norm(camera_rays, axis=1)[:, None]
    rotation = camera_to_world_matrix(pose.yaw_deg, pose.pitch_deg, pose.roll_deg)
    world_rays = camera_rays @ rotation.T
    return world_rays / np.linalg.norm(world_rays, axis=1)[:, None]
def obj_to_gltf(point):
    return [float(point[0]), float(point[2]), -float(point[1])]

class DisplaySurface:
    """Raycast the exact supplied display model, keeping source mesh/triangle IDs."""
    def __init__(self, path):
        self.path = Path(path)
        self.g = g = json.loads(self.path.read_text())
        buffers = [(self.path.parent/b['uri']).read_bytes() for b in g['buffers']]
        types = {5123: '<u2', 5125: '<u4', 5126: '<f4'}
        widths = {'SCALAR': 1, 'VEC3': 3}
        def acc(i):
            a = g['accessors'][i]; v = g['bufferViews'][a['bufferView']]
            dt = np.dtype(types[a['componentType']]); n = widths[a['type']]
            return np.ndarray((a['count'], n), dtype=dt, buffer=buffers[v['buffer']],
                              offset=v.get('byteOffset', 0)+a.get('byteOffset', 0),
                              strides=(v.get('byteStride', dt.itemsize*n), dt.itemsize)).copy()
        roots = g['scenes'][g.get('scene', 0)]['nodes']
        if len(roots) != 1:
            raise ValueError('Expected one scene root')
        root = g['nodes'][roots[0]]
        self.translation = np.array(root['translation'], dtype=np.float64)
        self.triangle_records = []
        pieces = []
        for node_id, node in enumerate(g['nodes']):
            if 'mesh' not in node:
                continue
            if any(k in node for k in ['translation', 'rotation', 'scale', 'matrix']):
                raise ValueError('Unexpected descendant transform in supplied asset')
            name = node.get('name', '')
            if '지형' in name or '수면' in name or (name.startswith('SPW_') and node.get('extras', {}).get('face') == 1):
                continue
            mesh_id = node['mesh']
            for primitive_id, p in enumerate(g['meshes'][mesh_id]['primitives']):
                v = acc(p['attributes']['POSITION']).astype(np.float64)+self.translation
                v = v[:, [0, 2, 1]]
                v[:, 1] *= -1
                ix = acc(p['indices']).reshape(-1, 3)
                pieces.append(v[ix])
                for j in range(len(ix)):
                    self.triangle_records.append(dict(mesh_id=mesh_id, triangle_id=j, primitive_id=primitive_id,
                                                      node_id=node_id, node_guid=node['extras']['guid'],
                                                      region_guid=node['extras']['guid'] if 'area' in node.get('extras', {}) else None,
                                                      mesh_name=name))
        vertices = np.concatenate(pieces).reshape(-1, 3)
        faces = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
        self.raycaster = WarpMeshRaycaster(vertices, faces, device='cpu')

    def cast(self, context, xy):
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        origins = np.repeat(np.array(context.pose.center_xyz)[None], len(xy), axis=0)
        directions = pixels_to_world_rays(context.intrinsics, context.pose, xy)
        flags, distance, faces, points = self.raycaster.intersect_rays(origins, directions)
        results = []
        for i in range(len(xy)):
            if not flags[i]:
                results.append(None)
                continue
            result = dict(self.triangle_records[int(faces[i])])
            result.update(gltf_xyz=obj_to_gltf(points[i]), world_obj_xyz=[float(x) for x in points[i]],
                          ray_distance_m=float(distance[i]))
            results.append(result)
        return results
def review_measurement(spatial: dict, focal_px: float) -> dict:
    """Conservative display gate, NOT a new GSD estimate or accuracy claim.

    For the user's roughly frontal imagery assumption, a local scale over 10x
    camera-to-hit distance / focal length needs manual review. Raw legacy
    measurements are preserved even when omitted from the public CSV.
    """
    reasons = []
    if not spatial.get("xyz_valid"):
        reasons.append(spatial.get("xyz_miss_reason") or "representative_ray_no_hit")
    if not spatial.get("measurement_valid"):
        reasons.append(spatial.get("measurement_miss_reason") or "local_gsd_unavailable")
    distance = spatial.get("mesh_ray_t_m")
    reference = float(distance) / float(focal_px) if distance is not None and distance > 0 else None
    scales = [spatial.get("gsd_length_m_per_px"), spatial.get("gsd_width_m_per_px")]
    scale = max((float(value) for value in scales if value is not None), default=None)
    ratio = scale / reference if reference and scale is not None else None
    if ratio is not None and (not math.isfinite(ratio) or ratio > GSD_REVIEW_RATIO):
        reasons.append("local_gsd_exceeds_10x_frontal_reference_manual_review")
    return {"physical_values_exported": not reasons,
            "reasons": reasons, "frontal_reference_m_per_px": reference,
            "max_directional_gsd_to_frontal_ratio": ratio,
            "review_ratio_threshold": GSD_REVIEW_RATIO}
