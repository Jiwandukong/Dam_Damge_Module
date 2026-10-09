"""Export unchanged damage detections as measured LAS subsets, PNGs and CSVs."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
from pyproj import CRS
import shapely

from runtime import CODE, file_hash, write_json
from visualize_results import collect_context, make_visualization

ROOT = CODE.parent
DEFAULT_WORK = ROOT.parent / "unmodified_multibeam" / "Multibeam_work"
FIELDS = ["dataset", "damage_id", "source_damage_id", "damage_type", "damage_name_ko",
          "world_center_x_m", "world_center_y_m", "world_center_z_m", "crs",
          "area_m2", "mean_depth_m", "max_depth_m", "median_depth_m", "volume_loss_m3",
          "point_count", "pointcloud_selection", "analysis_method", "analysis_resolution_m", "result_origin",
          "source_data_path", "pointcloud_path", "visualization_path", "boundary_xy_json"]
KEY_DTYPE = np.dtype([("X", "<i4"), ("Y", "<i4"), ("Z", "<i4")])


def in_polygon(geometry, x, y):
    west, south, east, north = geometry.bounds
    bbox = (x >= west) & (x <= east) & (y >= south) & (y <= north)
    indices = np.flatnonzero(bbox)
    if indices.size:
        indices = indices[shapely.intersects_xy(geometry, x[indices], y[indices])]
    return indices


def keys_from_xyz(xyz, header):
    scaled = np.rint((xyz - header.offsets) / header.scales).astype(np.int64)
    if (scaled < np.iinfo(np.int32).min).any() or (scaled > np.iinfo(np.int32).max).any():
        raise ValueError("Candidate coordinates exceed source LAS precision")
    reconstructed = scaled * header.scales + header.offsets
    if not np.allclose(xyz, reconstructed, rtol=0, atol=1e-8):
        raise ValueError("Candidate XYZ differs from original LAS coordinate precision")
    result = np.empty(len(xyz), dtype=KEY_DTYPE)
    for index, name in enumerate(KEY_DTYPE.names):
        result[name] = scaled[:, index]
    return result


def point_keys(points):
    result = np.empty(len(points), dtype=KEY_DTYPE)
    for name in KEY_DTYPE.names:
        result[name] = points[name]
    return result


def observe(feature, xyz):
    """Keep the closest measured height to the polygon centroid for the display Z."""
    if not len(xyz):
        return
    center = feature["geometry"].centroid
    distance = (xyz[:, 0] - center.x) ** 2 + (xyz[:, 1] - center.y) ** 2
    index = int(np.argmin(distance))
    if distance[index] < feature.get("center_distance", np.inf):
        feature["center_distance"] = float(distance[index])
        feature["center_z"] = float(xyz[index, 2])
        feature["center_z_sample_xy"] = xyz[index, :2].tolist()
    feature["point_count"] += len(xyz)


def load_features(results, dataset, method, resolution):
    stem = f"{dataset}_{method}_order1a_res{resolution:.1f}"
    paths = {"SC": results / "scour" / (stem + "_fillgap.gpkg"),
             "DP": results / "depression/depression_patches.gpkg"}
    features = []
    for kind, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Final {kind} detection is missing: {path}")
        frame = gpd.read_file(path)
        if frame.crs is None or frame.crs.to_epsg() != 5186:
            raise ValueError(f"Expected EPSG:5186: {path}")
        table = pd.read_csv(path.with_suffix(".csv")) if kind == "SC" and len(frame) else None
        for index, row in frame.iterrows():
            geometry = row.geometry
            if geometry.is_empty:
                raise ValueError(f"Empty source polygon: {path}, row {index}")
            # OpenCV contours can touch themselves at a vertex. Keep the original
            # coordinates and use the source polygon's covered region unchanged.
            shapely.prepare(geometry)
            source_id = str(table.iloc[index]["scour_id"]) if kind == "SC" else str(int(row["patch_id"]))
            span = max(geometry.bounds[2] - geometry.bounds[0], geometry.bounds[3] - geometry.bounds[1])
            margin = max(2.0, span * 0.15)
            feature = {
                "kind": kind, "source_id": source_id, "geometry": geometry,
                "area": float(row["area_m2"] if kind == "SC" else row["area"]),
                "mean_depth": float(row["mean_depth"]), "max_depth": float(row["max_depth"]),
                "median_depth": float(row["median_depth"]) if kind == "SC" else "",
                "volume": float(row["volume_loss"]) if kind == "DP" else "",
                "point_count": 0, "context": [],
                "selection": "raw_points_in_polygon" if kind == "SC" else "damage_candidates",
                "context_bounds": [geometry.bounds[0] - margin, geometry.bounds[1] - margin,
                                   geometry.bounds[2] + margin, geometry.bounds[3] + margin],
                "source_gpkg": path,
            }
            features.append(feature)
    return features


def read_existing(output):
    rows = []
    for kind in ("SC", "DP"):
        path = output / "Result" / (kind + "_result.csv")
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    if "overlay_path" in row:
                        row["visualization_path"] = row.pop("overlay_path")
                    rows.append(row)
    return rows


def assign_ids(features, rows, dataset):
    used = {row["damage_id"] for row in rows}
    existing = {(row["damage_type"], row["source_damage_id"]): row["damage_id"]
                for row in rows if row["dataset"] == dataset}
    for feature in features:
        key = (feature["kind"], feature["source_id"])
        if key in existing:
            feature["id"] = existing[key]
        else:
            numbers = [int(value.split("-")[1]) for value in used if value.startswith(feature["kind"] + "-")]
            feature["id"] = f"{feature['kind']}-{max(numbers, default=0) + 1:03d}"
            used.add(feature["id"])


def extract_scour(source_las, features, staging):
    if not features:
        return
    with laspy.open(source_las) as reader, ExitStack() as stack:
        header = deepcopy(reader.header)
        header.add_crs(CRS.from_epsg(5186))
        for feature in features:
            feature["native_point_hash"] = hashlib.sha256()
            feature["writer"] = stack.enter_context(laspy.open(staging / f"Result/SC/{feature['id']}.las", mode="w", header=header))
        for points in reader.chunk_iterator(1_000_000):
            x, y = np.asarray(points.x), np.asarray(points.y)
            for feature in features:
                indices = in_polygon(feature["geometry"], x, y)
                if not len(indices):
                    continue
                subset = points[indices]
                feature["writer"].write_points(subset)
                feature["native_point_hash"].update(subset.array.tobytes())
                observe(feature, np.column_stack([subset.x, subset.y, subset.z]))


def extract_depression(results, features, staging):
    if not features:
        return
    source = results / "slab/A_align_filled_slab_zone_open3d_patch_final_final2.las"
    candidates = pd.read_csv(results / "depression/damage_candidate_points.csv",
                             usecols=["x", "y", "z", "residual"], float_precision="round_trip")
    selected_frames = {}
    for feature in features:
        indices = in_polygon(feature["geometry"], candidates.x.to_numpy(), candidates.y.to_numpy())
        selected_frames[feature["id"]] = candidates.iloc[indices]
    without_candidates = [feature for feature in features if selected_frames[feature["id"]].empty]
    if without_candidates:
        # Grid-patch and individual-point MAD thresholds are distinct original
        # tests. A valid grid patch may have no point-level candidates. Represent
        # that patch with its actual cleaned observations, without inventing XYZ.
        pieces = {feature["id"]: [] for feature in without_candidates}
        for chunk in pd.read_csv(results / "depression/cleaned_points.csv", chunksize=250_000,
                                 usecols=["x", "y", "z", "residual"], float_precision="round_trip"):
            for feature in without_candidates:
                indices = in_polygon(feature["geometry"], chunk.x.to_numpy(), chunk.y.to_numpy())
                if len(indices): pieces[feature["id"]].append(chunk.iloc[indices])
        for feature in without_candidates:
            parts = pieces[feature["id"]]
            if parts:
                selected_frames[feature["id"]] = pd.concat(parts, ignore_index=True)
                feature["selection"] = "cleaned_patch_points"
    with laspy.open(source) as reader, ExitStack() as stack:
        native_header = reader.header
        header = deepcopy(native_header)
        header.add_crs(CRS.from_epsg(5186))
        header.add_extra_dim(laspy.ExtraBytesParams(name="depression_depth_m", type=np.float64,
                                                  description="Minus original plane residual"))
        for feature in features:
            selected = selected_frames[feature["id"]]
            keys = keys_from_xyz(selected[["x", "y", "z"]].to_numpy(), native_header)
            keys, unique_indices = np.unique(keys, return_index=True)
            feature["candidate_keys"] = keys
            feature["candidate_depth"] = -selected.residual.to_numpy()[unique_indices]
            feature["candidate_count"] = len(selected)
            feature["native_point_hash"] = hashlib.sha256()
            feature["depth_hash"] = hashlib.sha256()
            feature["writer"] = stack.enter_context(laspy.open(staging / f"Result/DP/{feature['id']}.las", mode="w", header=header))
        for points in reader.chunk_iterator(500_000):
            x, y, z = np.asarray(points.x), np.asarray(points.y), np.asarray(points.z)
            for feature in features:
                indices = in_polygon(feature["geometry"], x, y)
                keys = feature["candidate_keys"]
                if not len(indices) or not len(keys):
                    continue
                queried = point_keys(points[indices])
                positions = np.searchsorted(keys, queried)
                valid = positions < len(keys)
                valid[valid] &= keys[positions[valid]] == queried[valid]
                indices, positions = indices[valid], positions[valid]
                if not len(indices):
                    continue
                subset = points[indices]
                output = laspy.ScaleAwarePointRecord.zeros(len(subset), header=header)
                for name in native_header.point_format.dimension_names:
                    output[name] = subset[name]
                depth = feature["candidate_depth"][positions]
                output.depression_depth_m = depth
                feature["writer"].write_points(output)
                feature["native_point_hash"].update(subset.array.tobytes())
                feature["depth_hash"].update(depth.astype("<f8").tobytes())
                observe(feature, np.column_stack([subset.x, subset.y, subset.z]))
        for feature in features:
            if feature["point_count"] != feature["candidate_count"]:
                raise ValueError(f"Candidate/source LAS point mismatch: {feature['id']} "
                                 f"{feature['point_count']} != {feature['candidate_count']}")


def verify_las(feature, staging, native_header):
    path = staging / f"Result/{feature['kind']}/{feature['id']}.las"
    if not feature["point_count"]:
        raise ValueError(f"Detected polygon contains no measured points: {feature['id']}")
    native_hash, depth_hash = hashlib.sha256(), hashlib.sha256()
    with laspy.open(path) as reader:
        if reader.header.point_count != feature["point_count"] or reader.header.parse_crs().to_epsg() != 5186:
            raise ValueError(f"LAS count/CRS mismatch: {path}")
        for points in reader.chunk_iterator(500_000):
            indices = in_polygon(feature["geometry"], np.asarray(points.x), np.asarray(points.y))
            if len(indices) != len(points):
                raise ValueError(f"LAS includes points outside source polygon: {path}")
            if feature["kind"] == "SC":
                native_hash.update(points.array.tobytes())
            else:
                original = laspy.ScaleAwarePointRecord.zeros(len(points), header=native_header)
                for name in native_header.point_format.dimension_names:
                    original[name] = points[name]
                native_hash.update(original.array.tobytes())
                depth_hash.update(np.asarray(points.depression_depth_m).astype("<f8").tobytes())
    if native_hash.digest() != feature["native_point_hash"].digest():
        raise ValueError(f"Source point attributes/XYZ changed in export: {path}")
    if feature["kind"] == "DP" and depth_hash.digest() != feature["depth_hash"].digest():
        raise ValueError(f"Original residual depth changed in export: {path}")
    return {"id": feature["id"], "points": feature["point_count"],
            "source_point_bytes_identical": True, "within_original_polygon": True,
            "source_polygon_valid": bool(feature["geometry"].is_valid),
            "source_polygon_validity": shapely.is_valid_reason(feature["geometry"]),
            "pointcloud_selection": feature["selection"],
            "crs": "EPSG:5186", "sha256": file_hash(path),
            "center_z_source_distance_m": float(np.sqrt(feature["center_distance"])),
            "center_z_sample_xy": feature["center_z_sample_xy"]}


def export_results(results, source_input, las_source=None, output=None, work=None,
                   method="posterior", resolution=1.0, result_origin="existing_results"):
    results, source_input = Path(results).resolve(), Path(source_input).resolve()
    output = Path(output or ROOT / "Output").resolve()
    work = Path(work or os.environ.get("MULTIBEAM_WORK_DIR", DEFAULT_WORK)).resolve()
    las_source = Path(las_source or source_input).resolve()
    if output == work or output in work.parents or work in output.parents:
        raise ValueError("Work and final Output must use separate directories")
    features = load_features(results, source_input.stem, method, resolution)
    old_rows = read_existing(output)
    assign_ids(features, old_rows, source_input.stem)
    staging_root = work / "export_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=source_input.stem + "_", dir=staging_root) as temp:
        staging = Path(temp)
        for category in ("Result", "Visualiza"):
            for kind in ("SC", "DP"):
                (staging / category / kind).mkdir(parents=True)
        print("Extracting observed scour points from raw LAS...", flush=True)
        scour = [f for f in features if f["kind"] == "SC"]
        depression = [f for f in features if f["kind"] == "DP"]
        extract_scour(las_source, scour, staging)
        print("Matching original slab points to depression candidates...", flush=True)
        extract_depression(results, depression, staging)
        print("Sampling surrounding observations for 3D snapshots...", flush=True)
        collect_context(las_source, scour)
        collect_context(results / "slab/A_align_filled_slab_zone_open3d_patch_final_final2.las", depression)
        report, new_rows = [], []
        for feature in features:
            original_las = las_source if feature["kind"] == "SC" else results / "slab/A_align_filled_slab_zone_open3d_patch_final_final2.las"
            with laspy.open(original_las) as reader: native_header = reader.header
            report.append(verify_las(feature, staging, native_header))
            report[-1]["visualization"] = make_visualization(feature, staging)
            center = feature["geometry"].centroid
            new_rows.append(dict(zip(FIELDS, [source_input.stem, feature["id"], feature["source_id"], feature["kind"],
                "세굴" if feature["kind"] == "SC" else "슬래브 함몰", float(center.x), float(center.y), feature["center_z"],
                "EPSG:5186", feature["area"], feature["mean_depth"], feature["max_depth"], feature["median_depth"], feature["volume"],
                feature["point_count"], feature["selection"], method if feature["kind"] == "SC" else "reference_plane",
                resolution if feature["kind"] == "SC" else "", result_origin,
                os.path.relpath(source_input, output / "Result"),
                f"{feature['kind']}/{feature['id']}.las", f"../Visualiza/{feature['kind']}/{feature['id']}.png",
                json.dumps(shapely.geometry.mapping(feature["geometry"]), separators=(",", ":"))], strict=True)))
            print(f"{feature['id']}: {feature['point_count']:,} measured points; 3D PNG rendered", flush=True)
        rows = [row for row in old_rows if row["dataset"] != source_input.stem] + new_rows
        for kind in ("SC", "DP"):
            with (staging / "Result" / (kind + "_result.csv")).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(sorted((row for row in rows if row["damage_type"] == kind), key=lambda row: row["damage_id"]))
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                destination = output / path.relative_to(staging)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, destination)
        active_ids = {row["damage_id"] for row in new_rows}
        for row in old_rows:
            if row["dataset"] == source_input.stem and row["damage_id"] not in active_ids:
                for relative in (row["pointcloud_path"], row["visualization_path"]):
                    path = (output / "Result" / relative).resolve()
                    if output in path.parents and path.is_file(): path.unlink()
    verification = {"passed": True, "dataset": source_input.stem, "result_origin": result_origin,
                    "source_input_sha256": file_hash(source_input), "source_results": str(results),
                    "center_definition": "polygon centroid XY; nearest extracted observed point Z",
                    "features": report, "source_gpkg_sha256": {str(f["source_gpkg"]): file_hash(f["source_gpkg"]) for f in features}}
    write_json(work / "verification" / (source_input.stem + "_export.json"), verification)
    return new_rows


def main():
    parser = argparse.ArgumentParser(description="기존 손상 결과 → 손상별 LAS·PNG·CSV")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--las-source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Output")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()
    export_results(args.results_dir, args.input, args.las_source, args.output_dir, args.work_dir)


if __name__ == "__main__":
    main()
