"""Export only GPR candidate Overlays and the 3D display CSV from working results."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

from PIL import Image

from overlay_markers import DEFAULT_MARKER_HALF_SIZE, DEFAULT_MARKER_WIDTH, render_markers
from coordinate_mapping import DEFAULT_TRANSFORM, candidate_world, load_transform, mapping_status
from scan_geometry import dzt_scan_geometry
from grid_mapping import assigned_grid, load_grid_mapping

ROOT = Path(__file__).resolve().parents[1]
FREQUENCIES = (400, 900, 1600, 2600)
FOLDER_NAMES = {400: "400MHz", 900: "900MHz", 1600: "1.6GHz", 2600: "2.6GHz"}
RESULT_COLUMNS = [
    "image", "damage_id", "damage_type", "damage_name_ko", "pixel_nodes_json",
    "world_center_x_m", "world_center_y_m", "world_center_z_m",
    "member_name", "grid_id", "freq_mhz", "line_no", "x_m", "t_ns",
    "source_data_path", "overlay_path", "mapping_status",
]
IMAGE_RX = re.compile(r"LINE_(\d{3})\.png$")


def read_candidates(source):
    with (source / "anomaly_rb/anomalies_rulebased.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    required = {"freq_mhz", "line_no", "x_px", "y_px", "x_m", "t_ns", "X", "Y", "Z"}
    if rows and not required.issubset(rows[0]):
        raise ValueError("Working candidate CSV is missing required fields")
    for row in rows:
        if not all(math.isfinite(float(row[c])) for c in required):
            raise ValueError("Candidate coordinates must be finite")
    return sorted(rows, key=lambda r: (int(r["freq_mhz"]), int(r["line_no"])))


def export_demo(source_output, output, data=ROOT / "Data/processed data", *,
                marker_half_size=DEFAULT_MARKER_HALF_SIZE, marker_width=DEFAULT_MARKER_WIDTH,
                transform=DEFAULT_TRANSFORM):
    source, output, data = map(lambda p: Path(p).expanduser().resolve(),
                               (source_output, output, data))
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Working results and final Output must be separate directories")
    if output.exists() and any(p.name not in {"Overlay", "Result"} for p in output.iterdir()):
        raise ValueError("Final Output may contain only Overlay/ and Result/; archive old outputs first")
    rows = read_candidates(source)
    calibration = load_transform(transform)
    grid_mapping = load_grid_mapping(transform)
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["freq_mhz"]), int(row["line_no"])].append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gpr-display-", dir=output.parent) as temporary:
        staged = Path(temporary) / "Output"
        (staged / "Result").mkdir(parents=True)
        image_keys = set()
        for freq in FREQUENCIES:
            images = sorted((source / "result_image" / str(freq)).glob("LINE_*.png"))
            expected_names = {f"LINE_{line:03d}.png" for line in range(1, 13)}
            if {p.name for p in images} != expected_names:
                raise ValueError(f"Expected 12 pixel-aligned B-scans for {freq} MHz")
            for image_path in images:
                line = int(IMAGE_RX.fullmatch(image_path.name).group(1))
                key = freq, line
                image_keys.add(key)
                centers = [(float(r["x_px"]), float(r["y_px"])) for r in grouped[key]]
                with Image.open(image_path) as base:
                    if any(not (0 <= x < base.width and 0 <= y < base.height) for x, y in centers):
                        raise ValueError(f"Candidate outside B-scan: {freq} MHz LINE_{line:03d}")
                    overlay = render_markers(base, centers, marker_half_size, marker_width)
                target = staged / "Overlay/ANM" / str(freq) / f"LINE_{line:03d}_anomaly.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                overlay.save(target)
        if set(grouped) - image_keys:
            raise ValueError("Candidate CSV references a missing B-scan")
        result = staged / "Result/ANM_result.csv"
        with result.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            scan_cache = {}
            for index, row in enumerate(rows, 1):
                freq, line = int(row["freq_mhz"]), int(row["line_no"])
                filename = f"{freq}_LINE_{line:03d}.DZT"
                raw = data / FOLDER_NAMES[freq] / filename
                if not raw.is_file():
                    raise FileNotFoundError(raw)
                if raw not in scan_cache:
                    scan_cache[raw] = dzt_scan_geometry(raw)
                along_m = float(row["x_px"]) * scan_cache[raw]["dx_m"]
                xyz = candidate_world(line, along_m, calibration)
                member_name, grid_id = assigned_grid(f'G{index:06d}', xyz, grid_mapping)
                writer.writerow({
                    "image": filename, "damage_id": f"G{index:06d}",
                    "damage_type": "ANM", "damage_name_ko": "GPR 이상 후보",
                    "pixel_nodes_json": json.dumps([[float(row["x_px"]), float(row["y_px"])]],
                                                   separators=(",", ":")),
                    "world_center_x_m": xyz[0], "world_center_y_m": xyz[1],
                    "world_center_z_m": xyz[2], "member_name": member_name, "grid_id": grid_id,
                    "freq_mhz": freq, "line_no": line, "x_m": along_m, "t_ns": row["t_ns"],
                    "source_data_path": os.path.relpath(raw, output / "Result").replace(os.sep, "/"),
                    "overlay_path": f"../Overlay/ANM/{freq}/LINE_{line:03d}_anomaly.png",
                    "mapping_status": mapping_status(calibration),
                })
        previous = output.parent / (output.name + ".previous_export")
        if previous.exists():
            raise FileExistsError(f"Previous export needs review: {previous}")
        had_output = output.exists()
        if had_output:
            output.rename(previous)
        try:
            staged.rename(output)
        except BaseException:
            if had_output:
                previous.rename(output)
            raise
        if had_output:
            shutil.rmtree(previous)
    print(f"Exported {len(image_keys)} Overlays and {len(rows)} candidate rows: {output}")
    return output / "Result/ANM_result.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "Output")
    parser.add_argument("--input", type=Path, default=ROOT / "Data/processed data")
    parser.add_argument("--transform", type=Path, default=DEFAULT_TRANSFORM)
    parser.add_argument("--marker-half-size", type=int, default=DEFAULT_MARKER_HALF_SIZE)
    parser.add_argument("--marker-width", type=int, default=DEFAULT_MARKER_WIDTH)
    args = parser.parse_args()
    if args.marker_half_size < 1 or args.marker_width < 1:
        parser.error("Marker size and width must be positive")
    export_demo(args.source_output, args.output, args.input,
                marker_half_size=args.marker_half_size, marker_width=args.marker_width,
                transform=args.transform)


if __name__ == "__main__":
    main()
