"""Load and verify the repository's externally distributed mesh contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml


MESH_ASSET_KEY = "daecheong_dam_mesh"
DEFAULT_BOUNDS_TOLERANCE_M = 1e-6


class AssetContractError(ValueError):
    """Raised when an asset or loaded geometry violates its manifest contract."""


@dataclass(frozen=True)
class MeshAssetContract:
    """Validated ``assets.yaml`` contract for the georeferenced OBJ."""

    schema_version: str
    asset_key: str
    repository_path: str
    source_filename: str | None
    size_bytes: int
    sha256: str
    vertex_count_loaded: int
    face_count_loaded: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    coordinate_contract: dict[str, Any]

    def repository_file(self, project_root: str | Path) -> Path:
        return Path(project_root) / PurePosixPath(self.repository_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "asset_key": self.asset_key,
            "repository_path": self.repository_path,
            "source_filename": self.source_filename,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "geometry": {
                "vertex_count_loaded": self.vertex_count_loaded,
                "face_count_loaded": self.face_count_loaded,
                "bounds_min": list(self.bounds_min),
                "bounds_max": list(self.bounds_max),
            },
            "coordinate_contract": dict(self.coordinate_contract),
        }


def load_mesh_asset_contract(
    manifest_path: str | Path,
    *,
    asset_key: str = MESH_ASSET_KEY,
) -> MeshAssetContract:
    """Read and strictly validate the mesh entry in ``assets.yaml``."""

    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"asset manifest not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "asset manifest root")
    schema_version = str(root.get("schema_version", ""))
    if not schema_version:
        raise AssetContractError("asset manifest schema_version is required")
    assets = _mapping(root.get("assets"), "asset manifest assets")
    entry = _mapping(assets.get(asset_key), f"assets.{asset_key}")
    geometry = _mapping(entry.get("geometry"), f"assets.{asset_key}.geometry")
    coordinate_contract = _mapping(
        entry.get("coordinate_contract"),
        f"assets.{asset_key}.coordinate_contract",
    )

    repository_path = str(entry.get("repository_path", ""))
    relative_path = PurePosixPath(repository_path)
    if (
        not repository_path
        or relative_path.is_absolute()
        or ".." in relative_path.parts
    ):
        raise AssetContractError(
            f"assets.{asset_key}.repository_path must be a safe project-relative path"
        )

    size_bytes = _positive_int(entry.get("size_bytes"), f"assets.{asset_key}.size_bytes")
    sha256 = str(entry.get("sha256", "")).lower()
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise AssetContractError(f"assets.{asset_key}.sha256 must be a SHA-256 hex digest")

    bounds_min = _bounds(geometry.get("bounds_min"), f"assets.{asset_key}.geometry.bounds_min")
    bounds_max = _bounds(geometry.get("bounds_max"), f"assets.{asset_key}.geometry.bounds_max")
    if any(lower > upper for lower, upper in zip(bounds_min, bounds_max)):
        raise AssetContractError(f"assets.{asset_key}.geometry bounds_min exceeds bounds_max")

    source_filename_value = entry.get("source_filename")
    return MeshAssetContract(
        schema_version=schema_version,
        asset_key=asset_key,
        repository_path=repository_path,
        source_filename=(
            str(source_filename_value) if source_filename_value is not None else None
        ),
        size_bytes=size_bytes,
        sha256=sha256,
        vertex_count_loaded=_positive_int(
            geometry.get("vertex_count_loaded"),
            f"assets.{asset_key}.geometry.vertex_count_loaded",
        ),
        face_count_loaded=_positive_int(
            geometry.get("face_count_loaded"),
            f"assets.{asset_key}.geometry.face_count_loaded",
        ),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        coordinate_contract=dict(coordinate_contract),
    )


def verify_mesh_file(
    mesh_path: str | Path,
    contract: MeshAssetContract,
    *,
    chunk_size: int = 16 * 1024 * 1024,
) -> dict[str, Any]:
    """Require exact OBJ byte size and SHA-256 before loading the mesh."""

    path = Path(mesh_path)
    if not path.is_file():
        raise FileNotFoundError(f"mesh asset not found: {path}")
    actual_size = path.stat().st_size
    if actual_size != contract.size_bytes:
        raise AssetContractError(
            "mesh size mismatch: "
            f"actual={actual_size}, expected={contract.size_bytes}, path={path}"
        )
    actual_sha256 = sha256_file(path, chunk_size=chunk_size)
    if actual_sha256 != contract.sha256:
        raise AssetContractError(
            "mesh SHA-256 mismatch: "
            f"actual={actual_sha256}, expected={contract.sha256}, path={path}"
        )
    return {
        "verified": True,
        "size_bytes": actual_size,
        "sha256": actual_sha256,
        "expected_size_bytes": contract.size_bytes,
        "expected_sha256": contract.sha256,
    }


def verify_loaded_mesh_geometry(
    *,
    vertex_count: int,
    face_count: int,
    bounds: Any,
    contract: MeshAssetContract,
    bounds_tolerance_m: float = DEFAULT_BOUNDS_TOLERANCE_M,
) -> dict[str, Any]:
    """Require loaded mesh topology and bounds to match ``assets.yaml``."""

    tolerance = float(bounds_tolerance_m)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("bounds_tolerance_m must be a finite non-negative number")
    actual_bounds = np.asarray(bounds, dtype=np.float64)
    if actual_bounds.shape != (2, 3) or not np.isfinite(actual_bounds).all():
        raise AssetContractError("loaded mesh bounds must be a finite 2x3 array")

    actual_vertex_count = _loaded_positive_int(vertex_count, "loaded mesh vertex_count")
    actual_face_count = _loaded_positive_int(face_count, "loaded mesh face_count")
    expected_bounds = np.asarray(
        [contract.bounds_min, contract.bounds_max], dtype=np.float64
    )
    failures: list[str] = []
    if actual_vertex_count != contract.vertex_count_loaded:
        failures.append(
            "vertex_count "
            f"actual={actual_vertex_count}, expected={contract.vertex_count_loaded}"
        )
    if actual_face_count != contract.face_count_loaded:
        failures.append(
            f"face_count actual={actual_face_count}, expected={contract.face_count_loaded}"
        )
    if not np.allclose(
        actual_bounds,
        expected_bounds,
        rtol=0.0,
        atol=tolerance,
    ):
        failures.append(
            "bounds "
            f"actual={actual_bounds.tolist()}, expected={expected_bounds.tolist()}, "
            f"tolerance_m={tolerance}"
        )
    if failures:
        raise AssetContractError("loaded mesh geometry mismatch: " + "; ".join(failures))

    return {
        "verified": True,
        "vertex_count_loaded": actual_vertex_count,
        "face_count_loaded": actual_face_count,
        "bounds_min": actual_bounds[0].tolist(),
        "bounds_max": actual_bounds[1].tolist(),
        "bounds_tolerance_m": tolerance,
        "expected_vertex_count_loaded": contract.vertex_count_loaded,
        "expected_face_count_loaded": contract.face_count_loaded,
        "expected_bounds_min": list(contract.bounds_min),
        "expected_bounds_max": list(contract.bounds_max),
    }


def sha256_file(path: str | Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AssetContractError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AssetContractError(f"{name} must be a positive integer")
    return value


def _loaded_positive_int(value: Any, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise AssetContractError(f"{name} must be a positive integer")
    return int(value)


def _bounds(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise AssetContractError(f"{name} must contain three finite numbers")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise AssetContractError(f"{name} must contain three finite numbers") from exc
    if not all(math.isfinite(item) for item in result):
        raise AssetContractError(f"{name} must contain three finite numbers")
    return result  # type: ignore[return-value]
