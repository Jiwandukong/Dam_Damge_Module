from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from uav_rgb.asset_contract import (
    AssetContractError,
    load_mesh_asset_contract,
    sha256_file,
    verify_loaded_mesh_geometry,
    verify_mesh_file,
)
import uav_rgb.pipeline as pipeline_module


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_manifest(
    path: Path,
    mesh_bytes: bytes,
    *,
    repository_path: str = "01_RawData/geometry/test.obj",
    vertex_count: int = 3,
    face_count: int = 1,
    bounds_min: list[float] | None = None,
    bounds_max: list[float] | None = None,
) -> Path:
    data = {
        "schema_version": "1.0",
        "assets": {
            "daecheong_dam_mesh": {
                "repository_path": repository_path,
                "source_filename": "test.obj",
                "size_bytes": len(mesh_bytes),
                "sha256": hashlib.sha256(mesh_bytes).hexdigest(),
                "geometry": {
                    "vertex_count_loaded": vertex_count,
                    "face_count_loaded": face_count,
                    "bounds_min": bounds_min or [1.0, 2.0, 3.0],
                    "bounds_max": bounds_max or [4.0, 5.0, 6.0],
                },
                "coordinate_contract": {
                    "horizontal_crs": "EPSG:5186",
                    "vertical_axis": "Z-up",
                },
            }
        },
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_load_and_verify_mesh_contract(tmp_path):
    mesh_bytes = b"v 1 2 3\nv 4 5 6\nv 2 3 4\nf 1 2 3\n"
    mesh = tmp_path / "test.obj"
    mesh.write_bytes(mesh_bytes)
    manifest = _write_manifest(tmp_path / "assets.yaml", mesh_bytes)

    contract = load_mesh_asset_contract(manifest)
    verification = verify_mesh_file(mesh, contract, chunk_size=3)

    assert contract.repository_file(tmp_path) == (
        tmp_path / "01_RawData/geometry/test.obj"
    )
    assert contract.to_dict()["geometry"]["vertex_count_loaded"] == 3
    assert verification["verified"] is True
    assert verification["size_bytes"] == len(mesh_bytes)
    assert verification["sha256"] == hashlib.sha256(mesh_bytes).hexdigest()
    assert sha256_file(mesh, chunk_size=2) == verification["sha256"]


@pytest.mark.parametrize("repository_path", ["../outside.obj", "/tmp/outside.obj"])
def test_manifest_rejects_unsafe_repository_path(tmp_path, repository_path):
    manifest = _write_manifest(
        tmp_path / "assets.yaml",
        b"mesh",
        repository_path=repository_path,
    )
    with pytest.raises(AssetContractError, match="safe project-relative"):
        load_mesh_asset_contract(manifest)


def test_manifest_rejects_non_integer_geometry_count(tmp_path):
    manifest = _write_manifest(tmp_path / "assets.yaml", b"mesh")
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    data["assets"]["daecheong_dam_mesh"]["geometry"]["face_count_loaded"] = 1.0
    manifest.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(AssetContractError, match="positive integer"):
        load_mesh_asset_contract(manifest)


def test_mesh_file_rejects_size_and_sha_mismatches(tmp_path):
    original = b"correct mesh bytes"
    manifest = _write_manifest(tmp_path / "assets.yaml", original)
    contract = load_mesh_asset_contract(manifest)

    wrong_size = tmp_path / "wrong-size.obj"
    wrong_size.write_bytes(original + b"!")
    with pytest.raises(AssetContractError, match="mesh size mismatch"):
        verify_mesh_file(wrong_size, contract)

    wrong_sha = tmp_path / "wrong-sha.obj"
    wrong_sha.write_bytes(b"x" * len(original))
    with pytest.raises(AssetContractError, match="mesh SHA-256 mismatch"):
        verify_mesh_file(wrong_sha, contract)


def test_loaded_geometry_contract_and_bounds_tolerance(tmp_path):
    manifest = _write_manifest(tmp_path / "assets.yaml", b"mesh")
    contract = load_mesh_asset_contract(manifest)
    bounds = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0000005]])

    report = verify_loaded_mesh_geometry(
        vertex_count=3,
        face_count=1,
        bounds=bounds,
        contract=contract,
    )
    assert report["verified"] is True
    assert report["bounds_tolerance_m"] == 1e-6

    with pytest.raises(AssetContractError, match="vertex_count"):
        verify_loaded_mesh_geometry(
            vertex_count=4,
            face_count=1,
            bounds=bounds,
            contract=contract,
        )
    with pytest.raises(AssetContractError, match="bounds"):
        verify_loaded_mesh_geometry(
            vertex_count=3,
            face_count=1,
            bounds=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.01]]),
            contract=contract,
        )


def test_repository_manifest_is_parseable():
    contract = load_mesh_asset_contract(
        PROJECT_ROOT / "01_RawData/manifests/assets.yaml"
    )
    assert contract.size_bytes == 184_020_785
    assert contract.vertex_count_loaded == 1_006_692
    assert contract.face_count_loaded == 2_012_989
    assert contract.coordinate_contract["horizontal_crs"] == "EPSG:5186"


def test_pipeline_rejects_checkpoint_sha_bypass_before_io(tmp_path):
    with pytest.raises(ValueError, match="mandatory and cannot be disabled"):
        pipeline_module.UAVRGBPipeline(
            checkpoint_path=tmp_path / "missing.pt",
            config_path=tmp_path / "missing.yaml",
            mesh_path=tmp_path / "missing.obj",
            mrk_path=None,
            verify_checkpoint_sha256=False,
        )


def test_pipeline_preflight_uses_manifest_for_file_and_loaded_geometry(
    tmp_path, monkeypatch
):
    mesh_bytes = b"not parsed because MeshSurfaceIndex is replaced"
    mesh = tmp_path / "test.obj"
    mesh.write_bytes(mesh_bytes)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test checkpoint")
    manifest = _write_manifest(tmp_path / "assets.yaml", mesh_bytes)
    seen: dict[str, object] = {}

    class FakeInferenceEngine:
        @classmethod
        def from_config(cls, checkpoint_path, config, *, device, verify_sha256):
            seen["checkpoint_path"] = checkpoint_path
            seen["verify_sha256"] = verify_sha256
            return SimpleNamespace(checkpoint={"sha256": "fake"}, device="cpu")

    class FakeSurface:
        def __init__(self, path, *, ray_backend, warp_device):
            seen["mesh_path"] = path
            self.vertex_count = 3
            self.face_count = 1
            self.bounds = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            self.ray_backend = "trimesh"

    monkeypatch.setattr(pipeline_module, "InferenceEngine", FakeInferenceEngine)
    monkeypatch.setattr(pipeline_module, "MeshSurfaceIndex", FakeSurface)
    pipeline = pipeline_module.UAVRGBPipeline(
        checkpoint_path=checkpoint,
        config_path=PROJECT_ROOT / "03_Processing/configs/daechung_aug512.yaml",
        mesh_path=mesh,
        mrk_path=None,
        asset_manifest_path=manifest,
    )

    assert seen["verify_sha256"] is True
    assert pipeline.mesh_file_verification["verified"] is True
    assert pipeline.mesh_geometry_verification["verified"] is True
    assert pipeline.mesh_asset_contract.sha256 == hashlib.sha256(mesh_bytes).hexdigest()
