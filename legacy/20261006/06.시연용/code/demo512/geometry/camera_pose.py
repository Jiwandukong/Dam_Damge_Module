"""DJI camera metadata and EPSG:5186 pose conversion for mesh ray mapping."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


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
