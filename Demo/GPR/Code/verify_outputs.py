"""Validate the GPR display package and compare it with original working results."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from export_demo import RESULT_COLUMNS, FREQUENCIES, read_candidates
from coordinate_mapping import DEFAULT_TRANSFORM, candidate_world, load_transform, mapping_status
from scan_geometry import dzt_scan_geometry
from grid_mapping import assigned_grid, load_grid_mapping

EXPECTED_COUNTS = {400: 41, 900: 42, 1600: 26, 2600: 17}


def load_results(output):
    with (output / "Result/ANM_result.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != RESULT_COLUMNS:
            raise AssertionError("Display CSV columns or order differ from the documented schema")
        return list(reader)


def validate(output, transform=DEFAULT_TRANSFORM):
    if {p.name for p in output.iterdir()} != {"Overlay", "Result"}:
        raise AssertionError("Output must contain only Overlay/ and Result/")
    rows = load_results(output)
    calibration = load_transform(transform)
    grid_mapping = load_grid_mapping(transform)
    if Counter(int(r["freq_mhz"]) for r in rows) != EXPECTED_COUNTS:
        raise AssertionError("Candidate counts differ from the original 126 candidates")
    if [r["damage_id"] for r in rows] != [f"G{i:06d}" for i in range(1, 127)]:
        raise AssertionError("Candidate IDs must be unique and sequential")
    expected_images = {
        f"Overlay/ANM/{freq}/LINE_{line:03d}_anomaly.png"
        for freq in FREQUENCIES for line in range(1, 13)
    }
    expected_files = expected_images | {"Result/ANM_result.csv"}
    actual_files = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if actual_files != expected_files:
        raise AssertionError(f"Unexpected final files: missing={expected_files-actual_files}, extra={actual_files-expected_files}")
    groups = defaultdict(list)
    scan_cache = {}
    for row in rows:
        if row["damage_type"] != "ANM" or row["damage_name_ko"] != "GPR 이상 후보":
            raise AssertionError("Incorrect candidate display type")
        if row["mapping_status"] != mapping_status(calibration):
            raise AssertionError("Candidate mapping status must match the calibration review status")
        for name in ("world_center_x_m", "world_center_y_m", "world_center_z_m", "x_m", "t_ns"):
            if not math.isfinite(float(row[name])):
                raise AssertionError(f"Non-finite {name}")
        xyz = candidate_world(row["line_no"], row["x_m"], calibration)
        if (row['member_name'],row['grid_id']) != assigned_grid(row['damage_id'],xyz,grid_mapping):
            raise AssertionError('CSV grid/member assignment differs from the reviewed model mapping')
        if any(not math.isclose(float(row["world_center_"+axis+"_m"]), value, rel_tol=0, abs_tol=1e-8)
               for axis,value in zip("xyz", xyz)):
            raise AssertionError("Candidate XYZ differs from the metre-preserving transform")
        nodes = json.loads(row["pixel_nodes_json"])
        if len(nodes) != 1 or len(nodes[0]) != 2 or not all(math.isfinite(v) for v in nodes[0]):
            raise AssertionError("GPR pixel_nodes_json must hold one marker center")
        for name in ("source_data_path", "overlay_path"):
            value = row[name]
            if "\\" in value or Path(value).is_absolute():
                raise AssertionError(f"Use portable relative paths: {name}")
            target = (output / "Result" / value).resolve()
            if not target.is_file():
                raise AssertionError(f"Missing referenced file: {target}")
            if name == "overlay_path" and not target.is_relative_to(output.resolve()):
                raise AssertionError("Overlay path escapes final Output")
            if name == "source_data_path":
                if target not in scan_cache:
                    scan_cache[target] = dzt_scan_geometry(target)
                expected_distance = nodes[0][0] * scan_cache[target]["dx_m"]
                if not math.isclose(float(row["x_m"]), expected_distance, rel_tol=0, abs_tol=1e-10):
                    raise AssertionError("Candidate distance must equal pixel trace coordinate / DZT scans-per-metre")
        groups[row["overlay_path"]].append(nodes[0])
    for line in (1, 2, 3):
        heights = {float(r['world_center_z_m']) for r in rows if int(r['line_no']) == line}
        if len(heights) > 1:
            raise AssertionError(f"Horizontal LINE_{line:03d} candidates change world Z")
    for relative in expected_images:
        path = output / relative
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.height != 512:
                raise AssertionError(f"Unexpected B-scan geometry: {path}")
            csv_path = "../" + relative
            for x, y in groups[csv_path]:
                if not (0 <= x < image.width and 0 <= y < image.height):
                    raise AssertionError("Marker center is outside its Overlay")
                if image.getpixel((round(x), round(y))) != (255, 0, 0):
                    raise AssertionError("CSV marker center does not match the red Overlay marker")
    return rows


def compare(reference, output, rows, tolerance, marker_half_size, marker_width):
    originals = read_candidates(reference)
    if len(originals) != len(rows):
        raise AssertionError("Candidate count changed")
    groups = defaultdict(list)
    for old, new in zip(originals, rows):
        pairs = [("freq_mhz", "freq_mhz"), ("line_no", "line_no"), ("t_ns", "t_ns")]
        for old_name, new_name in pairs:
            if not math.isclose(float(old[old_name]), float(new[new_name]), rel_tol=0, abs_tol=tolerance):
                raise AssertionError(f"Candidate numeric value changed: {new['damage_id']} {old_name}")
        xy = json.loads(new["pixel_nodes_json"])[0]
        for name, value in zip(("x_px", "y_px"), xy):
            if not math.isclose(float(old[name]), value, rel_tol=0, abs_tol=tolerance):
                raise AssertionError("Candidate pixel position changed")
        groups[int(old["freq_mhz"]), int(old["line_no"])].append(xy)
    for freq in FREQUENCIES:
        for line in range(1, 13):
            with Image.open(reference / "result_image" / str(freq) / f"LINE_{line:03d}.png") as original:
                base = original.convert("RGB")
            with Image.open(output / "Overlay/ANM" / str(freq) / f"LINE_{line:03d}_anomaly.png") as candidate:
                candidate.load()
                if base.size != candidate.size:
                    raise AssertionError("B-scan was resized")
                difference = ImageChops.difference(base, candidate)
                allowed = Image.new("L", base.size, 0)
                draw = ImageDraw.Draw(allowed)
                scale = max(1.0, base.width / 1000) if base.width / base.height >= 4 else 1.0
                extent = round(marker_half_size * scale) + round(marker_width * scale) + 1
                centers = groups[freq, line]
                for x, y in centers:
                    cx, cy = round(x), round(y)
                    draw.rectangle((cx-extent, cy-extent, cx+extent, cy+extent), fill=255)
                difference.paste((0, 0, 0), (0, 0), allowed)
                if difference.getbbox() is not None:
                    raise AssertionError("B-scan background changed outside marker neighborhoods")
                old_overlay = reference / "anomaly_rb" / str(freq) / f"LINE_{line:03d}_anomaly.png"
                if centers and old_overlay.is_file():
                    with Image.open(old_overlay) as old_image:
                        before = sum(p == (255, 0, 0) for p in old_image.convert("RGB").getdata())
                    after = sum(p == (255, 0, 0) for p in candidate.getdata())
                    if after <= before:
                        raise AssertionError(f"Markers did not grow: {freq} MHz LINE_{line:03d}")
    print("PASS: 126 candidate image coordinates and times preserved; measured distances validated against DZT headers; B-scan backgrounds unchanged; markers enlarged")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "Output")
    parser.add_argument("--reference", type=Path, help="Original working results containing result_image/ and anomaly_rb/")
    parser.add_argument("--transform", type=Path, default=DEFAULT_TRANSFORM)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--marker-half-size", type=int, default=10)
    parser.add_argument("--marker-width", type=int, default=3)
    args = parser.parse_args()
    rows = validate(args.output.resolve(), args.transform)
    if args.reference is not None:
        compare(args.reference.resolve(), args.output.resolve(), rows, args.tolerance,
                args.marker_half_size, args.marker_width)
    print(f"PASS: 48 Overlays, one CSV, {len(rows)} candidate rows: {args.output}")


if __name__ == "__main__":
    main()
