"""Raw photo -> native-512 SAM3 -> independent components -> OBJ report.

No GT, LabelMe JSON or training dataset is read. Only model checkpoint identity,
source-image metadata and the existing geometry contract are used. The output
is independent of the immutable GT deliverable and keeps its 16-column format.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from PIL import Image

DEMO_ROOT = Path(__file__).resolve().parents[2]

from .geometry.asset_contract import load_mesh_asset_contract, verify_mesh_file, verify_loaded_mesh_geometry
from .geometry.camera_pose import read_dji_xmp
from .geometry.mesh_ray import MeshSurfaceIndex, build_mesh_ray_context_with_surface
from .data import CLASSES, overlay_metadata, render_overlay, sha256_file
from .prediction_geometry import quantify_predictions
from .geometry.measurement_review import REQUIRED_XMP, review_measurement
from .raw_inference import RawInferenceEngine
from .report_contract import PRIMARY_COLUMNS, DAMAGE_NAMES_KO, json_ready, write_csv, write_workbook


def _dump(path: Path, value, *, compact: bool = False) -> None:
    # Large per-component records retain every field without indentation bytes.
    path.write_text(json.dumps(json_ready(value), ensure_ascii=False,
                               indent=None if compact else 2,
                               separators=(",", ":") if compact else None,
                               allow_nan=False) + "\n", encoding="utf-8")


def _source_images(images: str | Path) -> tuple[Path, list[Path]]:
    source = Path(images).expanduser().resolve(strict=True)
    suffixes = {".jpg", ".jpeg", ".png"}
    paths = ([source] if source.is_file() else
             sorted((path for path in source.iterdir() if path.is_file()
                     and path.suffix.lower() in suffixes), key=lambda path: path.name))
    if not paths or any(path.suffix.lower() not in suffixes for path in paths):
        raise ValueError("input must be a JPG/PNG image or a folder containing JPG/PNG images")
    seen = set()
    for path in paths:
        stem = path.stem
        if (not stem or stem in {".", ".."} or any(char in stem for char in "\\\r\n\t")
                or stem.casefold() in seen):
            raise ValueError("source image stems must be unique portable filenames")
        seen.add(stem.casefold())
    return source, paths


def _camera_context(image: Path, surface):
    """Missing camera metadata does not invent XYZ or suppress pixel results."""
    try:
        xmp = read_dji_xmp(image)
        missing = [key for key in REQUIRED_XMP
                   if key not in xmp or not math.isfinite(float(xmp[key]))]
        if missing:
            return None, {"reason": "camera_metadata_unavailable", "missing_fields": missing}
        if (float(xmp["CalibratedFocalLength"]) <= 0
                or not -90 <= float(xmp["GpsLatitude"]) <= 90
                or not -180 <= float(xmp["GpsLongitude"]) <= 180):
            return None, {"reason": "camera_metadata_invalid"}
        context = build_mesh_ray_context_with_surface(
            image, surface, mrk_path=None, representative_mode="centroid",
            node_sample_count=256, ray_batch_size=256)
    except (ValueError, KeyError, TypeError) as error:
        return None, {"reason": "camera_metadata_invalid", "detail": str(error)}
    return context, None


def _prediction_row(image: Path, result: dict, damage_id: str,
                    tiles: dict[tuple[int, int], dict], context) -> tuple[dict, dict]:
    base, spatial = result["base"], result["spatial"]
    px, py = spatial.get("image_pixel_x"), spatial.get("image_pixel_y")
    origins = [tuple(origin) for origin in result["tile_origins"]]
    if not origins or len(set(origins)) != len(origins):
        raise ValueError("each predicted component needs unique actual-overlap tile links")

    def order(origin):
        tile = tiles[origin]
        match = px is not None and py is not None and (
            tile["x0"] <= px < tile["x0"] + tile["valid_width"]
            and tile["y0"] <= py < tile["y0"] + tile["valid_height"])
        return not match, tile["y0"], tile["x0"]

    origins.sort(key=order)
    linked = [tiles[origin] for origin in origins]
    for tile in linked:
        tile["damage_ids"].append(damage_id)
    review = (review_measurement(spatial, context.intrinsics.focal_length_px)
              if context is not None else
              {"physical_values_exported": False, "reasons": ["camera_metadata_unavailable"],
               "frontal_reference_m_per_px": None, "max_directional_gsd_to_frontal_ratio": None,
               "review_ratio_threshold": 10.0})
    label = base["class_name"]
    row = dict.fromkeys(PRIMARY_COLUMNS)
    row.update(image=image.name, damage_id=damage_id, damage_type=label,
               damage_name_ko=DAMAGE_NAMES_KO[label],
               pixel_nodes_json=spatial.get("nodes_image_xy_json"),
               world_center_x_m=spatial.get("world_x_m") if spatial.get("xyz_valid") else None,
               world_center_y_m=spatial.get("world_y_m") if spatial.get("xyz_valid") else None,
               world_center_z_m=spatial.get("world_z_m") if spatial.get("xyz_valid") else None,
               length_px=spatial.get("length_px", base.get("length_px")),
               width_px=spatial.get("width_px", base.get("width_px")),
               source_image_path=f"원본사진/{image.name}",
               tile_original_paths_json=json.dumps([tile["image_path"] for tile in linked],
                                                   ensure_ascii=False, separators=(",", ":")),
               tile_overlay_paths_json=json.dumps([tile["overlay_path"] for tile in linked],
                                                  ensure_ascii=False, separators=(",", ":")))
    if review["physical_values_exported"]:
        row.update({key: spatial.get(key) for key in ("length_m", "width_m", "area_m2")})
    detail = {"damage_id": damage_id, "image": image.name, "label": label,
              "prediction_component_id": base["instance_id"],
              "component_local_id": base["instance_local_id"],
              "predicted_area_px": base["area_px"],
              "representative_source_pixel_xy": [px, py],
              "tile_ids": [tile["tile_id"] for tile in linked],
              "review": review, "raw_legacy_mapping_and_measurement": spatial}
    return row, detail


def export_predictions(images: str | Path, output: str | Path,
                       model_record: str | Path, mesh: str | Path,
                       asset_manifest: str | Path | None = None, *,
                       device: str = "auto", threshold: float = 0.5,
                       ray_backend: str = "auto", warp_device: str = "cpu",
                       save_diagnostics: bool = False) -> dict:
    started = time.monotonic()
    source, image_paths = _source_images(images)
    requested = Path(output).expanduser().absolute()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"refusing to overwrite an existing result: {requested}")
    destination = requested.resolve()
    if source.is_dir() and destination.is_relative_to(source):
        raise ValueError("output must be outside the source image folder")
    mesh_path = Path(mesh).expanduser().resolve(strict=True)
    manifest_path = (Path(asset_manifest).expanduser().resolve(strict=True) if asset_manifest
                     else DEMO_ROOT / "02.시연모델/댐3D모델/assets.yaml")
    contract = load_mesh_asset_contract(manifest_path)
    mesh_verification = verify_mesh_file(mesh_path, contract)
    surface = MeshSurfaceIndex(mesh_path, ray_backend=ray_backend, warp_device=warp_device)
    geometry_verification = verify_loaded_mesh_geometry(
        vertex_count=surface.vertex_count, face_count=surface.face_count,
        bounds=surface.bounds, contract=contract)
    engine = RawInferenceEngine(model_record, device=device, threshold=threshold)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows, calculations, tile_records, image_records = [], [], [], []
    with tempfile.TemporaryDirectory(prefix=".model-results-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        for name in ("원본사진", "512원본타일", "512모델예측오버레이"):
            (staging / name).mkdir()
        if save_diagnostics:
            for label in CLASSES:
                (staging / "모델예측마스크" / label).mkdir(parents=True)
        for image_number, image in enumerate(image_paths, 1):
            image_started = time.monotonic()
            digest = sha256_file(image)
            with Image.open(image) as opened:
                rgb = np.asarray(opened.convert("RGB")).copy()
            height, width = rgb.shape[:2]
            context, metadata_issue = _camera_context(image, surface)
            print(f"원본 추론: {image_number}/{len(image_paths)} {image.name}", flush=True)
            saved_tiles: dict[tuple[int, int], dict] = {}
            total_tiles = math.ceil(width / 512) * math.ceil(height / 512)

            def save_tile(tile, tile_rgb, masks):
                x0, y0 = tile["x0"], tile["y0"]
                tile_id = f"{image.stem}__x{x0:05d}_y{y0:05d}"
                original_path = f"512원본타일/{tile_id}.png"
                overlay_path = f"512모델예측오버레이/{tile_id}.png"
                original = Image.fromarray(tile_rgb)
                original.save(staging / original_path)
                render_overlay(original, {label: Image.fromarray(masks[label].astype(np.uint8) * 255)
                                           for label in CLASSES}).save(staging / overlay_path)
                tile_record = {"tile_id": tile_id, "source_id": image.stem,
                               "image": image.name, "x0": x0, "y0": y0,
                               "valid_width": tile["valid_width"], "valid_height": tile["valid_height"],
                               "image_path": original_path, "overlay_path": overlay_path,
                               "damage_ids": [],
                               "prediction_pixels": {label: int(np.count_nonzero(masks[label])) for label in CLASSES},
                               "original_sha256": sha256_file(staging / original_path),
                               "overlay_sha256": sha256_file(staging / overlay_path)}
                if (x0, y0) in saved_tiles:
                    raise ValueError("duplicate output tile")
                saved_tiles[(x0, y0)] = tile_record
                if len(saved_tiles) % 25 == 0 or len(saved_tiles) == total_tiles:
                    print(f"  추론·이미지 저장 {len(saved_tiles)}/{total_tiles} 타일", flush=True)

            prediction = engine.predict(rgb, on_tile=save_tile)
            expected_origins = {(x, y) for y in range(0, height, 512) for x in range(0, width, 512)}
            if set(saved_tiles) != expected_origins:
                raise ValueError("prediction must cover every original-photo tile")
            class_masks = prediction["class_masks"]
            mask_paths = {}
            for index, label in enumerate(CLASSES, 1):
                mask = class_masks[index]
                if mask.dtype != np.bool_ or mask.shape != (height, width):
                    raise ValueError("prediction masks must remain in the original image grid")
                if save_diagnostics:
                    relative = f"모델예측마스크/{label}/{image.stem}.png"
                    Image.fromarray(mask.astype(np.uint8) * 255).save(staging / relative)
                    mask_paths[label] = relative
            print("  예측 손상 분리·좌표·정량 계산", flush=True)
            measured = quantify_predictions(class_masks, context)
            for result in measured["instances"]:
                if len(rows) >= 999999:
                    raise ValueError("result exceeds the existing six-digit damage ID contract")
                row, detail = _prediction_row(image, result, f"D{len(rows)+1:06d}", saved_tiles, context)
                rows.append(row)
                calculations.append(detail)
            tile_records.extend(saved_tiles.values())
            shutil.copyfile(image, staging / "원본사진" / image.name)
            if sha256_file(image) != digest or sha256_file(staging / "원본사진" / image.name) != digest:
                raise ValueError("original photo changed during processing")
            image_records.append({"image": image.name, "image_sha256": digest,
                                  "width": width, "height": height, "tiles": len(saved_tiles),
                                  "class_summary": measured["class_summary"],
                                  "component_counts": measured["component_counts"],
                                  "mask_paths": mask_paths,
                                  "mask_sha256": {label: sha256_file(staging / relative)
                                                  for label, relative in mask_paths.items()},
                                  "mapping": context.to_dict() if context is not None else None,
                                  "metadata_issue": metadata_issue,
                                  "inference_seconds": prediction["elapsed_seconds"],
                                  "elapsed_seconds": time.monotonic() - image_started})
            print(f"  완료: 예측 손상 {len(measured['instances'])}건", flush=True)
            del prediction, class_masks, measured, rgb
        for row in rows:
            for key in ("tile_original_paths_json", "tile_overlay_paths_json"):
                for relative in json.loads(row[key]):
                    if not (staging / relative).is_file():
                        raise ValueError("missing linked output image")
        summary = {"images": len(image_paths), "rows": len(rows), "tiles": len(tile_records),
                   "class_rows": dict(Counter(row["damage_type"] for row in rows)),
                   "mapped_rows": sum(row["world_center_x_m"] is not None for row in rows),
                   "physical_values_exported": sum(item["review"]["physical_values_exported"] for item in calculations),
                   "physical_values_withheld": sum(not item["review"]["physical_values_exported"] for item in calculations),
                   "review_reasons": dict(Counter(reason for item in calculations for reason in item["review"]["reasons"])),
                   "inference_seconds": sum(item["inference_seconds"] for item in image_records)}
        write_csv(staging / "damage_results.csv", rows, PRIMARY_COLUMNS)
        write_workbook(staging / "damage_results.xlsx", rows=rows)
        _dump(staging / "연결정보.json", {
            "result_source": "model_predictions_without_ground_truth", "path_base": "this_output_folder",
            "tile_size": 512, "columns": list(PRIMARY_COLUMNS), "overlay": overlay_metadata(),
            "damage_id_scope": "this_result_bundle_only_not_GT_damage_identity",
            "tiles": tile_records,
            "damages": [{key: value for key, value in item.items()
                         if key not in {"review", "raw_legacy_mapping_and_measurement"}} for item in calculations]})
        if save_diagnostics:
            _dump(staging / "추론계산기록.json", {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "result_source": "model_predictions_without_ground_truth", "ground_truth_used": False,
                "model_loaded": True, "source_resized": False, "tile_selection": "all_source_tiles",
                "model": engine.metadata, "summary": summary, "columns": list(PRIMARY_COLUMNS),
                "mesh": {"contract": contract.to_dict(), "file_verification": mesh_verification,
                         "geometry_verification": geometry_verification,
                         "ray_backend": surface.ray_backend, "warp_device": warp_device},
                "measurement": {"approximate": True, "crc_width_is_physical_aperture": False,
                                "connected_components": "8_connectivity_per_class_on_full_original_image",
                                "min_area_px": 1, "independent_overlapping_classes": True,
                                "review_ratio_threshold": 10.0},
                "images": image_records, "damages": calculations,
                "elapsed_seconds": time.monotonic() - started}, compact=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("output appeared during inference; refusing to replace it")
        staging.rename(destination)
    return {"output": str(destination), **summary, "elapsed_seconds": time.monotonic() - started}
