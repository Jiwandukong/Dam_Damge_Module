#!/usr/bin/env python3
"""Preflight the mission image, MRK, OBJ and model contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "03_Processing" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from uav_rgb.asset_contract import (
    load_mesh_asset_contract,
    verify_loaded_mesh_geometry,
    verify_mesh_file,
)
from uav_rgb.camera_pose import image_index_from_name, read_dji_xmp, read_mrk_positions
from uav_rgb.config import EXPECTED_CHECKPOINT_SIZE
from uav_rgb.io import list_images
from uav_rgb.mesh_ray import MeshSurfaceIndex


def main() -> int:
    mission = PROJECT_ROOT / "01_RawData/missions/DJI_202507021616_left03"
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default=str(mission / "images"))
    parser.add_argument(
        "--mrk",
        default=str(mission / "navigation/DJI_202507021616_003_Timestamp.MRK"),
    )
    parser.add_argument("--mesh", default=None)
    parser.add_argument(
        "--asset-manifest",
        default=str(PROJECT_ROOT / "01_RawData/manifests/assets.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "02_Model/sam3_aug512/checkpoints/checkpoint_final.pt"),
    )
    args = parser.parse_args()

    mrk = Path(args.mrk).resolve()
    asset_manifest = Path(args.asset_manifest).resolve()
    mesh_contract = load_mesh_asset_contract(asset_manifest)
    mesh = Path(
        args.mesh if args.mesh is not None else mesh_contract.repository_file(PROJECT_ROOT)
    ).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    for path in (mrk, mesh, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    positions = read_mrk_positions(mrk)
    mesh_file_verification = verify_mesh_file(mesh, mesh_contract)
    mesh_surface = MeshSurfaceIndex(mesh, ray_backend="trimesh")
    mesh_geometry_verification = verify_loaded_mesh_geometry(
        vertex_count=mesh_surface.vertex_count,
        face_count=mesh_surface.face_count,
        bounds=mesh_surface.bounds,
        contract=mesh_contract,
    )
    image_rows = []
    for path in list_images(args.images):
        xmp = read_dji_xmp(path)
        index = image_index_from_name(path.name)
        with Image.open(path) as image:
            size = image.size
        required_xmp = all(
            key in xmp
            for key in (
                "CalibratedFocalLength",
                "CalibratedOpticalCenterX",
                "CalibratedOpticalCenterY",
                "GimbalYawDegree",
                "GimbalPitchDegree",
                "GimbalRollDegree",
            )
        )
        image_rows.append(
            {
                "file": path.name,
                "index": index,
                "size_wh": size,
                "mrk_match": index in positions,
                "required_xmp": required_xmp,
                "requires_edge_padding": bool(size[0] % 512 or size[1] % 512),
            }
        )
    report = {
        "ok": all(
            row["mrk_match"] and row["required_xmp"] for row in image_rows
        )
        and mesh_file_verification["verified"]
        and mesh_geometry_verification["verified"]
        and checkpoint.stat().st_size == EXPECTED_CHECKPOINT_SIZE,
        "images": image_rows,
        "mrk_record_count": len(positions),
        "mesh": {
            "path": str(mesh),
            "asset_manifest": str(asset_manifest),
            "contract": mesh_contract.to_dict(),
            "file_verification": mesh_file_verification,
            "loaded_geometry_verification": mesh_geometry_verification,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "expected_size_bytes": EXPECTED_CHECKPOINT_SIZE,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
