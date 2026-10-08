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
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True

import geopandas as gpd
import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as PlotPath
import numpy as np
import pandas as pd
from pyproj import CRS
import rasterio
from rasterio.windows import from_bounds
import shapely

from runtime import CODE, file_hash, write_json

ROOT = CODE.parent
DEFAULT_WORK = ROOT.parent / "unmodified_multibeam" / "Multibeam_work"
FIELDS = ["dataset", "damage_id", "source_damage_id", "damage_type", "damage_name_ko",
          "world_center_x_m", "world_center_y_m", "world_center_z_m", "crs",
          "area_m2", "mean_depth_m", "max_depth_m", "median_depth_m", "volume_loss_m3",
          "point_count", "pointcloud_selection", "analysis_method", "analysis_resolution_m", "result_origin",
          "source_data_path", "pointcloud_path", "overlay_path", "boundary_xy_json"]
KEY_DTYPE = np.dtype([("X", "<i4"), ("Y", "<i4"), ("Z", "<i4")])
COLORS = {"SC": "#ed713d", "DP": "#087e87"}


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
    return features, results / "tif" / (stem + ".tif")


def read_existing(output):
    rows = []
    for kind in ("SC", "DP"):
        path = output / "Result" / (kind + "_result.csv")
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows.extend(csv.DictReader(stream))
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
                west, south, east, north = feature["context_bounds"]
                context = (x >= west) & (x <= east) & (y >= south) & (y <= north)
                if context.any():
                    feature["context"].append(np.column_stack([x[context], y[context], z[context]]))
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


def draw_boundary(ax, geometry, center, color):
    parts = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
    for part in parts:
        vertices, codes = [], []
        for ring in [part.exterior, *part.interiors]:
            xy = np.array(ring.coords)[:, :2] - center
            vertices.extend(xy.tolist())
            codes.extend([PlotPath.MOVETO] + [PlotPath.LINETO] * (len(xy) - 2) + [PlotPath.CLOSEPOLY])
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.5)
        ax.add_patch(PathPatch(PlotPath(vertices, codes), facecolor=color, alpha=0.12, edgecolor="none"))


def make_overlay(feature, raster, staging):
    center = np.array([feature["geometry"].centroid.x, feature["geometry"].centroid.y])
    west, south, east, north = feature["context_bounds"]
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f3f6f8")
    if feature["kind"] == "SC":
        with rasterio.open(raster) as src:
            window = from_bounds(west, south, east, north, src.transform).round_offsets().round_lengths()
            values = src.read(1, window=window, boundless=True, fill_value=np.nan)
            bounds = rasterio.windows.bounds(window, src.transform)
        masked = np.ma.masked_invalid(values)
        image = ax.imshow(masked, extent=[bounds[0] - center[0], bounds[2] - center[0],
                                        bounds[1] - center[1], bounds[3] - center[1]],
                          origin="upper", interpolation="nearest", cmap="Greys")
        bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
        bar.set_label("Observed surface Z (m)", color="#52687a")
    else:
        context = np.concatenate(feature["context"]) if feature["context"] else np.empty((0, 3))
        if len(context) > 60_000:
            context = context[np.linspace(0, len(context) - 1, 60_000, dtype=int)]
        ax.scatter(context[:, 0] - center[0], context[:, 1] - center[1], s=0.8, c="#aebdc7", alpha=0.7, rasterized=True)
        cloud = laspy.read(staging / f"Result/DP/{feature['id']}.las")
        take = np.linspace(0, len(cloud.points) - 1, min(60_000, len(cloud.points)), dtype=int)
        ax.scatter(np.asarray(cloud.x)[take] - center[0], np.asarray(cloud.y)[take] - center[1],
                   s=1.0, c=COLORS["DP"], alpha=0.8, rasterized=True)
    draw_boundary(ax, feature["geometry"], center, COLORS[feature["kind"]])
    ax.scatter([0], [0], marker="+", s=105, c="#152f44", linewidths=1.8, zorder=10)
    ax.set_xlim(west - center[0], east - center[0])
    ax.set_ylim(south - center[1], north - center[1])
    ax.set_aspect("equal")
    ax.set_xlabel("X offset from damage center (m)", color="#52687a")
    ax.set_ylabel("Y offset from damage center (m)", color="#52687a")
    ax.grid(alpha=0.15, color="#71889a", lw=0.5)
    ax.tick_params(labelsize=9, colors="#52687a")
    for spine in ax.spines.values(): spine.set_color("#d9e2e8")
    name = "Scour" if feature["kind"] == "SC" else "Slab depression"
    fig.suptitle(f"{feature['id']}  |  {name}", x=0.11, y=0.965, ha="left", fontsize=17, fontweight="bold", color="#17394e")
    fig.text(0.11, 0.91, f"Area {feature['area']:.2f} m²    Mean depth {feature['mean_depth']:.3f} m    Max depth {feature['max_depth']:.3f} m", fontsize=9, color="#52687a")
    legend = [Line2D([0], [0], color=COLORS[feature["kind"]], lw=2, label="Detected boundary"),
              Line2D([0], [0], color="#152f44", marker="+", linestyle="none", markersize=9, label="Damage center")]
    ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.92)
    fig.text(0.11, 0.045, f"Center: X {center[0]:.3f}  Y {center[1]:.3f}  Z {feature['center_z']:.3f} m  |  EPSG:5186", fontsize=9, color="#52687a")
    fig.subplots_adjust(left=0.11, right=0.91, top=0.88, bottom=0.13)
    fig.savefig(staging / f"Overlay/{feature['kind']}/{feature['id']}.png", facecolor="white")
    plt.close(fig)


def export_results(results, source_input, las_source=None, output=None, work=None,
                   method="posterior", resolution=1.0, result_origin="existing_results"):
    results, source_input = Path(results).resolve(), Path(source_input).resolve()
    output = Path(output or ROOT / "Output").resolve()
    work = Path(work or os.environ.get("MULTIBEAM_WORK_DIR", DEFAULT_WORK)).resolve()
    las_source = Path(las_source or source_input).resolve()
    if output == work or output in work.parents or work in output.parents:
        raise ValueError("Work and final Output must use separate directories")
    features, raster = load_features(results, source_input.stem, method, resolution)
    if not raster.exists():
        raise FileNotFoundError(raster)
    old_rows = read_existing(output)
    assign_ids(features, old_rows, source_input.stem)
    staging_root = work / "export_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=source_input.stem + "_", dir=staging_root) as temp:
        staging = Path(temp)
        for category in ("Result", "Overlay"):
            for kind in ("SC", "DP"):
                (staging / category / kind).mkdir(parents=True)
        print("Extracting observed scour points from raw LAS...", flush=True)
        scour = [f for f in features if f["kind"] == "SC"]
        depression = [f for f in features if f["kind"] == "DP"]
        extract_scour(las_source, scour, staging)
        print("Matching original slab points to depression candidates...", flush=True)
        extract_depression(results, depression, staging)
        report, new_rows = [], []
        for feature in features:
            original_las = las_source if feature["kind"] == "SC" else results / "slab/A_align_filled_slab_zone_open3d_patch_final_final2.las"
            with laspy.open(original_las) as reader: native_header = reader.header
            report.append(verify_las(feature, staging, native_header))
            make_overlay(feature, raster, staging)
            center = feature["geometry"].centroid
            new_rows.append(dict(zip(FIELDS, [source_input.stem, feature["id"], feature["source_id"], feature["kind"],
                "세굴" if feature["kind"] == "SC" else "슬래브 함몰", float(center.x), float(center.y), feature["center_z"],
                "EPSG:5186", feature["area"], feature["mean_depth"], feature["max_depth"], feature["median_depth"], feature["volume"],
                feature["point_count"], feature["selection"], method if feature["kind"] == "SC" else "reference_plane",
                resolution if feature["kind"] == "SC" else "", result_origin,
                os.path.relpath(source_input, output / "Result"),
                f"{feature['kind']}/{feature['id']}.las", f"../Overlay/{feature['kind']}/{feature['id']}.png",
                json.dumps(shapely.geometry.mapping(feature["geometry"]), separators=(",", ":"))], strict=True)))
            print(f"{feature['id']}: {feature['point_count']:,} measured points; PNG verified", flush=True)
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
                for relative in (row["pointcloud_path"], row["overlay_path"]):
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
