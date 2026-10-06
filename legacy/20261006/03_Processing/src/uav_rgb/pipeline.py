"""End-to-end SAM3 detection, OBJ ray mapping, GSD and report orchestration."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .asset_contract import (
    load_mesh_asset_contract,
    verify_loaded_mesh_geometry,
    verify_mesh_file,
)
from .config import MODEL_VARIANT, CrackSegConfig, load_config
from .inference import InferenceEngine
from .io import list_images
from .mesh_ray import MeshSurfaceIndex, build_mesh_ray_context_with_surface
from .reporting import save_image_result, write_combined_outputs


ProgressCallback = Callable[[str], None]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ASSET_MANIFEST = PROJECT_ROOT / "01_RawData" / "manifests" / "assets.yaml"


class UAVRGBPipeline:
    """Load heavyweight assets once and process one mission deterministically."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        config_path: str | Path,
        mesh_path: str | Path,
        mrk_path: str | Path | None,
        asset_manifest_path: str | Path | None = None,
        device: str = "auto",
        verify_checkpoint_sha256: bool = True,
        mesh_ray_backend: str = "auto",
        warp_device: str = "cuda:0",
        representative_mode: str = "centroid",
        instance_sample_count: int = 128,
        node_sample_count: int = 256,
        ray_batch_size: int = 256,
        min_instance_area_px: int = 8,
        profile: bool = False,
        progress: ProgressCallback | None = None,
    ) -> None:
        if verify_checkpoint_sha256 is not True:
            raise ValueError(
                "checkpoint SHA-256 verification is mandatory and cannot be disabled"
            )
        self.checkpoint_path = absolute_path(checkpoint_path)
        self.config_path = absolute_path(config_path)
        self.mesh_path = absolute_path(mesh_path)
        self.mrk_path = absolute_path(mrk_path) if mrk_path is not None else None
        self.asset_manifest_path = absolute_path(
            asset_manifest_path or DEFAULT_ASSET_MANIFEST
        )
        self.representative_mode = representative_mode
        self.instance_sample_count = int(instance_sample_count)
        self.node_sample_count = int(node_sample_count)
        self.ray_batch_size = int(ray_batch_size)
        self.min_instance_area_px = int(min_instance_area_px)
        self.profile = bool(profile)
        self.progress = progress or (lambda message: None)

        self._validate_inputs()
        self.mesh_asset_contract = load_mesh_asset_contract(self.asset_manifest_path)
        self.progress("OBJ 자산 크기 및 SHA-256 검증")
        self.mesh_file_verification = verify_mesh_file(
            self.mesh_path,
            self.mesh_asset_contract,
        )
        self.config: CrackSegConfig = load_config(self.config_path)
        self.progress("SAM3 체크포인트 검증 및 모델 로딩")
        model_start = time.perf_counter()
        self.engine = InferenceEngine.from_config(
            self.checkpoint_path,
            self.config,
            device=device,
            verify_sha256=True,
        )
        self.model_load_sec = time.perf_counter() - model_start

        self.progress("EPSG:5186 OBJ 메시 로딩")
        mesh_start = time.perf_counter()
        self.surface = MeshSurfaceIndex(
            self.mesh_path,
            ray_backend=mesh_ray_backend,
            warp_device=warp_device,
        )
        self.mesh_load_sec = time.perf_counter() - mesh_start
        self.mesh_geometry_verification = verify_loaded_mesh_geometry(
            vertex_count=self.surface.vertex_count,
            face_count=self.surface.face_count,
            bounds=self.surface.bounds,
            contract=self.mesh_asset_contract,
        )

    def _validate_inputs(self) -> None:
        for path, label in (
            (self.checkpoint_path, "checkpoint"),
            (self.config_path, "config"),
            (self.mesh_path, "mesh"),
            (self.asset_manifest_path, "asset manifest"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")
        if self.mesh_path.suffix.lower() != ".obj":
            raise ValueError("This project requires the georeferenced OBJ; GLB is not supported")
        if self.mrk_path is not None:
            if not self.mrk_path.is_file():
                raise FileNotFoundError(f"MRK not found: {self.mrk_path}")
            if self.mrk_path.suffix.lower() != ".mrk":
                raise ValueError("mrk_path must have a .MRK extension")
        if self.min_instance_area_px < 1:
            raise ValueError("min_instance_area_px must be >= 1")
        if self.representative_mode not in {"centroid", "median"}:
            raise ValueError("representative_mode must be centroid or median")

    def run(
        self,
        *,
        images: str | Path,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        image_paths = list_images(images)
        if not image_paths:
            raise ValueError(f"no supported images found: {images}")
        output_root = Path(output_dir).resolve()
        if output_root.exists() and any(output_root.iterdir()):
            raise FileExistsError(
                f"output directory must be new or empty to prevent mixed runs: {output_root}"
            )
        output_root.mkdir(parents=True, exist_ok=True)

        started = datetime.now(timezone.utc)
        total_start = time.perf_counter()
        all_rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        per_image_paths: dict[str, dict[str, str]] = {}
        inference_times: dict[str, float] = {}
        report_times: dict[str, float] = {}

        for image_number, image_path in enumerate(image_paths, start=1):
            self.progress(
                f"[{image_number}/{len(image_paths)}] {image_path.name}: SAM3 512 타일 추론"
            )
            with Image.open(image_path) as opened:
                pil_rgb = opened.convert("RGB")
                image_rgb = np.asarray(pil_rgb).copy()

            tile_state = {"last": 0, "total": 0}

            def tile_progress(current: int, total: int) -> None:
                tile_state["total"] = total
                if current == total or current == 1 or current - tile_state["last"] >= 10:
                    self.progress(
                        f"[{image_number}/{len(image_paths)}] {image_path.name}: "
                        f"tile {current}/{total}"
                    )
                    tile_state["last"] = current

            inference_start = time.perf_counter()
            prediction = self.engine.predict(
                pil_rgb,
                image_name=image_path.name,
                progress_callback=tile_progress,
            )
            inference_times[image_path.name] = time.perf_counter() - inference_start

            self.progress(
                f"[{image_number}/{len(image_paths)}] {image_path.name}: OBJ Ray 좌표/GSD"
            )
            geo_context = build_mesh_ray_context_with_surface(
                image_path=image_path,
                surface=self.surface,
                mrk_path=self.mrk_path,
                instance_sample_count=self.instance_sample_count,
                node_sample_count=self.node_sample_count,
                ray_batch_size=self.ray_batch_size,
                representative_mode=self.representative_mode,
                profile=self.profile,
            )
            report_start = time.perf_counter()
            result = save_image_result(
                image_rgb=image_rgb,
                image_path=image_path,
                prediction=prediction,
                config=self.config,
                geo3d_context=geo_context,
                output_dir=output_root,
                min_instance_area_px=self.min_instance_area_px,
                damage_id_start=len(all_rows) + 1,
            )
            report_times[image_path.name] = time.perf_counter() - report_start
            all_rows.extend(result["rows"])
            summaries.append(result["summary"])
            per_image_paths[image_path.name] = {
                key: relative_path(value, output_root)
                for key, value in result["paths"].items()
            }

        finished = datetime.now(timezone.utc)
        run_metadata: dict[str, Any] = {
            "schema_version": "1.0",
            "model_variant": MODEL_VARIANT,
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
            "image_count": len(image_paths),
            "damage_count": len(all_rows),
            "paths": {
                "checkpoint": portable_project_path(self.checkpoint_path),
                "config": portable_project_path(self.config_path),
                "mesh": portable_project_path(self.mesh_path),
                "asset_manifest": portable_project_path(self.asset_manifest_path),
                "mrk": portable_project_path(self.mrk_path) if self.mrk_path else None,
                "output": portable_project_path(output_root),
            },
            "asset_identity": {
                "checkpoint_size_bytes": self.checkpoint_path.stat().st_size,
                "checkpoint_sha256": self.engine.checkpoint.get("sha256"),
                "mesh_size_bytes": self.mesh_file_verification["size_bytes"],
                "mesh_sha256": self.mesh_file_verification["sha256"],
                "mesh_contract": {
                    "manifest": self.mesh_asset_contract.to_dict(),
                    "file_verification": dict(self.mesh_file_verification),
                    "loaded_geometry_verification": dict(
                        self.mesh_geometry_verification
                    ),
                },
            },
            "runtime": {
                "device": self.engine.device,
                "mesh_ray_backend": self.surface.ray_backend,
                "model_load_sec": round(self.model_load_sec, 6),
                "mesh_load_sec": round(self.mesh_load_sec, 6),
                "inference_sec_by_image": {
                    key: round(value, 6) for key, value in inference_times.items()
                },
                "report_sec_by_image": {
                    key: round(value, 6) for key, value in report_times.items()
                },
                "total_sec_before_final_write": round(
                    time.perf_counter() - total_start, 6
                ),
            },
            "processing_contract": {
                "source_resized": False,
                "tile_size": self.config.tile_size,
                "stride": self.config.stride,
                "edge_policy": self.config.edge_policy,
                "padding_mode": self.config.padding_mode,
                "mask_coordinates": "original source image pixels",
                "minimum_instance_area_px": self.min_instance_area_px,
                "instance_coordinate_mode": self.representative_mode,
                "horizontal_crs": "EPSG:5186",
                "mesh_axes": "X/Y EPSG:5186 metres; Z-up metres",
                "quantification_is_approximate": True,
            },
            "per_image_outputs": per_image_paths,
        }
        combined_paths = write_combined_outputs(
            rows=all_rows,
            run_metadata=run_metadata,
            output_dir=output_root,
        )
        self.progress(f"완료: {combined_paths['excel']}")
        return {
            "output_dir": output_root,
            "rows": all_rows,
            "summaries": summaries,
            "combined_paths": combined_paths,
            "run_metadata": run_metadata,
        }


def absolute_path(path: str | Path) -> Path:
    """Return an absolute path without dereferencing a repository asset symlink."""
    value = Path(path).expanduser()
    return value if value.is_absolute() else Path.cwd() / value


def relative_path(path: str | Path, base: str | Path) -> str:
    try:
        return str(Path(path).relative_to(Path(base)))
    except ValueError:
        return str(path)


def portable_project_path(path: str | Path) -> str:
    try:
        return str(Path(path).relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
