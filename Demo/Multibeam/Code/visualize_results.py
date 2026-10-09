"""Render measured damage LAS clouds as 3D snapshots without changing detections."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True

import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import shapely
from shapely.geometry import shape
from output_rules import REMOVED_CSV_FIELDS, apply_exclusions

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORK = ROOT.parent / "unmodified_multibeam/Multibeam_work"
BACKGROUND = "#080c14"
POINT_SIZE = 6
IMAGE_SIZE = (3840, 2160)
DPI = 300
BRIGHT_COLORS = LinearSegmentedColormap.from_list("bright_damage", ["#30dfff", "#70ffca", "#f8ff72", "#ff9860"])


def seed_for(identifier):
    return int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:8], "little")


def sample_las(path, limit=180_000):
    """Select reproducible observed points, preserving their measured XYZ and depth."""
    with laspy.open(path) as reader:
        count = reader.header.point_count
        if count == 0:
            raise ValueError(f"Empty damage LAS: {path}")
        rng = np.random.default_rng(seed_for(Path(path).stem))
        selected = np.sort(rng.choice(count, size=min(count, limit), replace=False))
        xyz, depths, offset = [], [], 0
        has_depth = "depression_depth_m" in reader.header.point_format.dimension_names
        bounds = np.array([reader.header.mins, reader.header.maxs])
        for points in reader.chunk_iterator(500_000):
            first, last = np.searchsorted(selected, [offset, offset + len(points)])
            take = selected[first:last] - offset
            if len(take):
                xyz.append(np.column_stack([points.x[take], points.y[take], points.z[take]]))
                if has_depth:
                    depths.append(np.asarray(points.depression_depth_m)[take])
            offset += len(points)
    coordinates = np.concatenate(xyz)
    if not np.isfinite(coordinates).all():
        raise ValueError(f"Non-finite measured coordinates: {path}")
    return coordinates, np.concatenate(depths) if depths else None, bounds


def collect_context(source, features, limit=30_000):
    """Sample surrounding observations with bounded memory, independently of scan order."""
    if not features or source is None:
        return
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(source)
    states = []
    for feature in features:
        states.append([feature, np.random.default_rng(seed_for(feature["id"] + "-context")),
                       np.empty((0, 3)), np.empty(0)])
    with laspy.open(source) as reader:
        for points in reader.chunk_iterator(500_000):
            x, y, z = np.asarray(points.x), np.asarray(points.y), np.asarray(points.z)
            for state in states:
                feature, rng, previous, previous_keys = state
                west, south, east, north = feature["context_bounds"]
                mask = ((x >= west) & (x <= east) & (y >= south) & (y <= north)
                        & np.isfinite(x) & np.isfinite(y) & np.isfinite(z))
                indices = np.flatnonzero(mask)
                if not len(indices):
                    continue
                # The detected region is represented by the separately sampled damage LAS.
                indices = indices[~shapely.intersects_xy(feature["geometry"], x[indices], y[indices])]
                if not len(indices):
                    continue
                candidates = np.concatenate([previous, np.column_stack([x[indices], y[indices], z[indices]])])
                keys = np.concatenate([previous_keys, rng.random(len(indices))])
                if len(keys) > limit:
                    take = np.argpartition(keys, limit - 1)[:limit]
                    candidates, keys = candidates[take], keys[take]
                state[2:] = [candidates, keys]
    for feature, _, coordinates, _ in states:
        feature["context"] = coordinates


def fit_camera(fig, ax, span, cloud, context):
    """Enlarge the orthographic view uniformly, retaining margins around all displayed points."""
    initial_zoom = 0.95
    ax.set_box_aspect(span, zoom=initial_zoom)
    fig.canvas.draw()
    visible = np.concatenate([cloud, context]) if len(context) else cloud
    x, y, _ = proj3d.proj_transform(visible[:, 0], visible[:, 1], visible[:, 2], ax.get_proj())
    pixels = ax.transData.transform(np.column_stack([x, y]))
    world_center = np.array([np.mean(ax.get_xlim()), np.mean(ax.get_ylim()), np.mean(ax.get_zlim())])
    x0, y0, _ = proj3d.proj_transform(*world_center, ax.get_proj())
    pivot = ax.transData.transform([x0, y0])
    low, high = pixels.min(axis=0), pixels.max(axis=0)
    width, height = fig.canvas.get_width_height()
    margin = np.array([width, height]) * 0.055
    space_low, space_high = pivot - margin, np.array([width, height]) - margin - pivot
    factors = np.concatenate([space_low / np.maximum(pivot - low, 1),
                              space_high / np.maximum(high - pivot, 1)])
    zoom = initial_zoom * min(float(factors.min()), 3.0)
    ax.set_box_aspect(span, zoom=zoom)
    return zoom


def make_visualization(feature, staging):
    path = Path(staging) / f"Result/{feature['kind']}/{feature['id']}.las"
    xyz, depth, bounds = sample_las(path)
    center = np.array([feature["geometry"].centroid.x, feature["geometry"].centroid.y, feature["center_z"]])
    cloud = xyz - center
    context = feature.get("context", np.empty((0, 3)))
    # Old export callers may retain context in chunks.
    if isinstance(context, list):
        context = np.concatenate(context) if context else np.empty((0, 3))
    context = context - center
    west, south, east, north = feature["geometry"].bounds
    margin = max(0.08, max(east - west, north - south) * 0.08)
    west, south, east, north = west - margin, south - margin, east + margin, north + margin
    if len(context):
        context = context[(context[:, 0] >= west - center[0]) & (context[:, 0] <= east - center[0])
                          & (context[:, 1] >= south - center[1]) & (context[:, 1] <= north - center[1])]
    limits = np.array([[west - center[0], south - center[1], bounds[0, 2] - center[2]],
                       [east - center[0], north - center[1], bounds[1, 2] - center[2]]])
    if len(context):
        lower, upper = np.percentile(context[:, 2], [0.5, 99.5])
        context = context[(context[:, 2] >= lower) & (context[:, 2] <= upper)]
        limits[0, 2] = min(limits[0, 2], lower)
        limits[1, 2] = max(limits[1, 2], upper)
    span = np.maximum(limits[1] - limits[0], 0.01)
    limits[:, 2] += np.array([-1, 1]) * max(span[2] * 0.06, 0.03)
    span = limits[1] - limits[0]
    values = depth if feature["kind"] == "DP" and depth is not None else xyz[:, 2]
    color_low, color_high = np.percentile(values, [1, 99])
    norm = Normalize(vmin=float(color_low), vmax=float(color_high) + 1e-12, clip=True)
    fig = plt.figure(figsize=(IMAGE_SIZE[0] / DPI, IMAGE_SIZE[1] / DPI), dpi=DPI, facecolor=BACKGROUND)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d", computed_zorder=False)
    ax.set_facecolor(BACKGROUND)
    if len(context):
        ax.scatter(context[:, 0], context[:, 1], context[:, 2], c="#99a9bc", s=POINT_SIZE,
                   alpha=0.65, linewidths=0, depthshade=False, clip_on=False, zorder=1)
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=values, cmap=BRIGHT_COLORS, norm=norm,
               s=POINT_SIZE, alpha=1, linewidths=0,
               depthshade=False, clip_on=False, zorder=2)
    ax.set_xlim(limits[:, 0])
    ax.set_ylim(limits[:, 1])
    ax.set_zlim(limits[:, 2])
    ax.view_init(elev=32, azim=-35)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    zoom = fit_camera(fig, ax, span, cloud, context)
    destination = Path(staging) / f"Visualiza/{feature['kind']}/{feature['id']}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, facecolor=BACKGROUND)
    plt.close(fig)
    return {"id": feature["id"], "sampled_damage_points": len(xyz),
            "sampled_context_points": len(context), "vertical_detail_factor": 1, "view_count": 1, "annotations": False,
            "true_scale": True, "image_size": list(IMAGE_SIZE), "observed_xyz_only": True,
            "damage_point_size_pt2": POINT_SIZE, "context_point_size_pt2": POINT_SIZE,
            "background": BACKGROUND, "dpi": DPI, "camera_zoom": zoom,
            "damage_alpha": 1, "context_alpha": 0.65,
            "color_display_range": [float(color_low), float(color_high)],
            "color_range_percentiles": [1, 99]}


def main():
    parser = argparse.ArgumentParser(description="기존 손상 LAS → Visualiza 3D 스냅샷; 재검출하지 않음")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Output")
    parser.add_argument("--work-dir", type=Path, default=Path(os.environ.get("MULTIBEAM_WORK_DIR", DEFAULT_WORK)))
    parser.add_argument("--slab-source", type=Path, help="DP 주변 관측점으로 사용할 원본 슬래브 LAS (선택)")
    args = parser.parse_args()
    output, work = args.output_dir.resolve(), args.work_dir.resolve()
    apply_exclusions(output)
    os.environ.setdefault("MPLCONFIGDIR", str(work / "cache/matplotlib"))
    tables, features = {}, []
    for kind in ("SC", "DP"):
        path = output / "Result" / f"{kind}_result.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = ["visualization_path" if name == "overlay_path" else name
                      for name in reader.fieldnames if name not in REMOVED_CSV_FIELDS]
            rows = list(reader)
        for row in rows:
            for name in REMOVED_CSV_FIELDS:
                row.pop(name, None)
            row.pop("overlay_path", None)
            row["visualization_path"] = f"../Visualiza/{kind}/{row['damage_id']}.png"
            geometry = shape(json.loads(row["boundary_xy_json"]))
            west, south, east, north = geometry.bounds
            margin = max(2.0, max(east - west, north - south) * 0.15)
            features.append({"kind": kind, "id": row["damage_id"], "geometry": geometry,
                             "area": float(row["area_m2"]), "mean_depth": float(row["mean_depth_m"]),
                             "center_z": float(row["world_center_z_m"]),
                             "point_count": int(row["point_count"]),
                             "context_bounds": [west - margin, south - margin, east + margin, north + margin],
                             "source": (output / "Result" / row["source_data_path"]).resolve()})
        tables[kind] = (fields, rows)
    if not features:
        raise ValueError("No damage CSV rows to visualize")
    scour_sources = {f["source"] for f in features if f["kind"] == "SC"}
    for source in scour_sources:
        group = [f for f in features if f["kind"] == "SC" and f["source"] == source]
        if source.is_file() and source.suffix.lower() in (".las", ".laz"):
            collect_context(source, group)
    if args.slab_source:
        if len({f["source"] for f in features if f["kind"] == "DP"}) > 1:
            raise ValueError("--slab-source is only supported for a single input dataset")
        collect_context(args.slab_source, [f for f in features if f["kind"] == "DP"])
    staging_root = work / "visualization_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    reports = []
    with tempfile.TemporaryDirectory(dir=staging_root) as temp:
        staging = Path(temp)
        # Existing LAS are read directly. Link them into staging; their bytes are never rewritten.
        for feature in features:
            relative = Path(f"Result/{feature['kind']}/{feature['id']}.las")
            (staging / relative).parent.mkdir(parents=True, exist_ok=True)
            (staging / relative).symlink_to(output / relative)
            reports.append(make_visualization(feature, staging))
            print(f"{feature['id']}: 3D snapshot rendered", flush=True)
        for kind, (fields, rows) in tables.items():
            (staging / "Result").mkdir(exist_ok=True)
            with (staging / "Result" / f"{kind}_result.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        for path in staging.rglob("*"):
            if path.is_file() and not path.is_symlink():
                destination = output / path.relative_to(staging)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, destination)
    (work / "verification").mkdir(parents=True, exist_ok=True)
    (work / "verification/visualization_3d.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
