"""Each stage runs in its own process so large point clouds release memory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from runtime import ALGORITHMS, CODE, SLAB_NAME, capture_plots, execute_algorithm, file_hash, write_json

sys.path.insert(0, str(ALGORITHMS))


def convert_input(source, output):
    """Adapt XYZ/E57 to the LAS format consumed by the existing notebooks."""
    source, output = Path(source), Path(output)
    if source.suffix.lower() in {".las", ".laz"}:
        return source
    import laspy
    import numpy as np

    destination = output / "input" / (source.stem + ".las")
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = destination.with_suffix(".json")
    digest = file_hash(source)
    if destination.exists() and metadata.exists():
        old = json.loads(metadata.read_text())
        if old.get("source_sha256") == digest and old.get("las_sha256") == file_hash(destination):
            return destination
    scans = []
    if source.suffix.lower() == ".e57":
        import pye57

        e57 = pye57.E57(str(source))
        arrays = []
        for index in range(e57.scan_count):
            raw = e57.read_scan_raw(index)
            header = e57.get_header(index)
            points = np.column_stack([raw[key] for key in ("cartesianX", "cartesianY", "cartesianZ")])
            keep = np.isfinite(points).all(axis=1)
            if "cartesianInvalidState" in raw:
                keep &= raw["cartesianInvalidState"] == 0
            points = points[keep]
            applied = False
            try:
                rotation, translation = header.rotation_matrix, header.translation
                if rotation is not None and translation is not None:
                    points = points @ np.asarray(rotation).T + np.asarray(translation)
                    applied = True
            except (AttributeError, KeyError, pye57.libe57.E57Exception):
                pass
            arrays.append(points)
            scans.append({"index": index, "raw_points": len(keep), "valid_points": len(points), "pose_applied": applied})
        if not arrays:
            raise ValueError("E57 contains no scans")
        xyz = np.concatenate(arrays)
        scale = 0.001  # Same E57-to-LAS precision as the original E57 execution.
    else:
        import pandas as pd

        frame = pd.read_csv(source, sep=r"\s+|,", engine="python", header=None, comment="#", usecols=[0, 1, 2])
        xyz = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(xyz).all():
            raise ValueError("XYZ must contain finite X Y Z values without a header")
        scale = 0.000001
    if len(xyz) == 0:
        raise ValueError(f"No points in {source}")
    # LAS integer coordinates must fit without silently wrapping.
    span = np.ptp(xyz, axis=0)
    if np.any(span / scale > np.iinfo(np.int32).max):
        raise ValueError("Point cloud extent exceeds LAS coordinate precision; provide LAS with suitable scales")
    header = laspy.LasHeader(point_format=0, version="1.2")
    header.scales = np.array([scale] * 3)
    header.offsets = xyz.min(axis=0)
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz.T
    las.write(destination)
    write_json(metadata, {"source": str(source), "source_sha256": digest, "las_sha256": file_hash(destination),
                          "points": len(xyz), "scales": header.scales.tolist(), "crs": None, "scans": scans})
    print(f"Converted {len(xyz):,} points: {destination}", flush=True)
    return destination


def cube(job):
    folder = Path(job["output"]) / "tif"
    folder.mkdir(parents=True, exist_ok=True)
    las = convert_input(job["input"], job["output"])
    ns = execute_algorithm("cube.py", {"las_file": str(las), "base_output_dir": str(folder),
                                       "resolutions": job["resolutions"], "methods": job["methods"]})
    records = []
    for resolution in job["resolutions"]:
        for method in job["methods"]:
            path = folder / f"{las.stem}_{method}_order1a_res{resolution:.1f}.tif"
            # Never count a stale file as a successful new calculation.
            path.unlink(missing_ok=True)
            started = time.monotonic()
            print(f"CUBE {method} / {resolution} m", flush=True)
            message = ns["process_cube"](resolution, method)
            exists = path.is_file()
            native_predicted_failure = method == "predicted" and not exists and "division by zero" in message
            record = {"method": method, "resolution": resolution, "output": str(path),
                      "success": exists, "original_predicted_failure": native_predicted_failure,
                      "message": message, "elapsed_seconds": round(time.monotonic() - started, 3)}
            records.append(record)
            write_json(Path(job["output"]) / "cube_status.json", {"records": records})
            print(message, flush=True)
    if not any(item["success"] for item in records):
        raise RuntimeError("No CUBE TIFF was produced; see cube_status.json")
    failed = [item for item in records if not item["success"] and not item["original_predicted_failure"]]
    if failed:
        raise RuntimeError(f"{len(failed)} CUBE calculations failed; see cube_status.json")


def selected_tiffs(job, folder):
    folder = Path(job["output"]) / folder
    paths = []
    for resolution in job["resolutions"]:
        for method in job["methods"]:
            suffix = "_fillgap" if folder.name == "fillgap" else ""
            path = folder / f"{Path(job['input']).stem}_{method}_order1a_res{resolution:.1f}{suffix}.tif"
            if path.exists():
                paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No selected TIFFs in {folder}; run cube/fillgap first")
    return paths


def fillgap(job):
    import whitebox

    binary_dir = CODE / "tools" / "whitebox"
    if (binary_dir / "whitebox_tools").is_file():
        os.environ["WBT_PATH"] = str(binary_dir)
    wbt = whitebox.WhiteboxTools()
    if (binary_dir / "whitebox_tools").is_file():
        wbt.set_whitebox_dir(str(binary_dir))
    output = Path(job["output"])
    (output / "fill").mkdir(exist_ok=True)
    (output / "fillgap").mkdir(exist_ok=True)
    records = []
    for path in selected_tiffs(job, "tif"):
        filled = output / "fill" / (path.stem + "_fill.tif")
        gap = output / "fillgap" / (path.stem + "_fillgap.tif")
        filled.unlink(missing_ok=True)
        gap.unlink(missing_ok=True)
        # These are exactly the two Whitebox calls in 1012.ipynb cell 3.
        rc = wbt.fill_depressions(dem=str(path), output=str(filled), fix_flats=True)
        if rc != 0 or not filled.exists():
            raise RuntimeError(f"Whitebox FillDepressions failed: {path}")
        rc = wbt.subtract(input1=str(filled), input2=str(path), output=str(gap))
        if rc != 0 or not gap.exists():
            raise RuntimeError(f"Whitebox Subtract failed: {path}")
        records.append({"cube": str(path), "fill": str(filled), "fillgap": str(gap)})
    write_json(output / "fillgap_status.json", {"records": records})


def scour(job):
    import geopandas as gpd

    output = Path(job["output"])
    folder = output / "scour"
    folder.mkdir(exist_ok=True)
    ns = execute_algorithm("scour.py")
    records = []
    for path in selected_tiffs(job, "fillgap"):
        data, transform, crs, dx, dy = ns["load_raster"](path)
        polygons, areas = ns["polygonize_raster"](data, transform, 0.5)
        polygons, areas = ns["filter_polygons"](polygons, areas, 0.5)
        depths = ns["calculate_polygon_depths"](data, polygons, transform, 0.5)
        gpkg = folder / (path.stem + ".gpkg")
        gpkg.unlink(missing_ok=True)
        ns["save_to_gpkg"](polygons, areas, depths, crs, str(gpkg))
        execute_algorithm("scour_csv.py", {"input_gpkg": str(gpkg), "tif_dir": str(output / "tif")})
        execute_algorithm("scour_xyz.py", {"input_gpkg": str(gpkg), "tif_dir": str(output / "tif")})
        ratio, valid_area, damage_area = ns["calculate_damage_ratio"](data, dx, dy, areas)
        records.append({"output": str(gpkg), "features": len(polygons), "valid_area_m2": valid_area,
                        "damage_area_m2": float(damage_area), "damage_ratio_percent": float(ratio)})
        assert len(gpd.read_file(gpkg)) == len(polygons)
    write_json(output / "scour_status.json", {"records": records})


def slab(job):
    folder = Path(job["output"]) / "slab"
    capture_plots(folder)
    las = convert_input(job["input"], job["output"])
    ns = execute_algorithm("slab.py", {"LAS_FP": str(las), "OUT_DIR": str(folder)})
    for name in ("patch_df", "merged_df", "geom_df"):
        if name in ns:
            ns[name].to_csv(folder / (name + ".csv"), index=True)
    write_json(Path(job["output"]) / "slab_status.json", {
        "full_points": len(ns["points_full"]), "working_points": len(ns["points_work"]),
        "final_groups": len(ns["final_groups"]), "filled_zone_points": len(ns["slab_points_full"]),
        "representative_plane": ns["final_slab_plane"].tolist(), "output": str(folder / SLAB_NAME)})


def depression(job):
    output = Path(job["output"])
    folder = output / "depression"
    capture_plots(folder)
    execute_algorithm("depression.py", {"INPUT_LAS": str(output / "slab" / SLAB_NAME), "OUTPUT_DIR": folder})


def local_damage(job):
    output = Path(job["output"])
    folder = output / "local_damage"
    capture_plots(folder)
    execute_algorithm("local_damage.py", {"SLAB_LAS_FP": str(output / "slab" / SLAB_NAME),
                                            "OUT_DIR": str(folder), "N_JOBS": 1})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["cube", "fillgap", "scour", "slab", "depression", "local_damage"])
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    globals()[args.stage](job)


if __name__ == "__main__":
    main()
