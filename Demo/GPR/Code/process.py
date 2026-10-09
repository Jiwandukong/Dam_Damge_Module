"""Run GPR processing and export only candidate Overlays and the 3D display CSV.

The detector settings match the delivered results. The display mapping keeps
vertical station spacing and scan lengths in metres, with an explicit
horizontal endpoint interpolation ratio when selected model anchors are used:

1. batch B-scan export and 2D line endpoints
2. apply the metre-preserving 2D-to-3D transform
3. export pixel-aligned B-scans and mapping CSV
4. rule-based anomaly detection
5. final display export to Output/Overlay and Output/Result

Intermediate images and CSVs are kept outside final Output in --work-dir.
Use --export-only to redraw markers from an existing working directory.

If the original four tie points become available, pass ``--tie-points`` with
``--x1``, ``--x2`` and ``--plane-y1`` to recompute the transform before stage 2.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from export_demo import export_demo
from overlay_markers import DEFAULT_MARKER_HALF_SIZE, DEFAULT_MARKER_WIDTH
from coordinate_mapping import load_transform


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "Code"
DEFAULT_INPUT = ROOT / "Data" / "processed data"
DEFAULT_OUTPUT = ROOT / "Output"
DEFAULT_TRANSFORM = ROOT / "Data" / "Calibration" / "transform.json"
DEFAULT_WORK = Path(os.environ.get("GPR_WORK_DIR", str(
    Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    / "dam_damage_module" / "gpr"
)))
FREQUENCY_FOLDERS = ("1.6GHz", "2.6GHz", "400MHz", "900MHz")


def run_stage(label: str, arguments: list[str]) -> None:
    print(f"\n[{label}]", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=CODE, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--export-only", action="store_true",
                        help="Redraw final Overlays and CSV from existing --work-dir results")
    parser.add_argument("--marker-half-size", type=int, default=DEFAULT_MARKER_HALF_SIZE)
    parser.add_argument("--marker-width", type=int, default=DEFAULT_MARKER_WIDTH)
    parser.add_argument("--transform", type=Path, default=DEFAULT_TRANSFORM)
    parser.add_argument("--tie-points", type=Path)
    parser.add_argument("--x1", type=float)
    parser.add_argument("--x2", type=float)
    parser.add_argument("--y0", type=float, default=0.0)
    parser.add_argument("--plane-y1", type=float,
                        help="Actual plane y coordinate of the upper calibration points; do not use the longest horizontal line")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--incoh-win", type=int, default=9)
    parser.add_argument("--tmask-top-ns", type=float, default=0.8)
    parser.add_argument("--tmask-bottom", type=float, default=0.8)
    parser.add_argument("--min-area", type=int, default=45)
    parser.add_argument("--incoh-q", type=float, default=None)
    args = parser.parse_args()
    for name in ("input", "output", "work_dir", "transform"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.tie_points is not None:
        args.tie_points = args.tie_points.expanduser().resolve()
        if args.x1 is None or args.x2 is None or args.plane_y1 is None:
            parser.error("--tie-points requires --x1, --x2 and the actual --plane-y1")
    if args.marker_half_size < 1 or args.marker_width < 1:
        parser.error("Marker size and width must be positive")
    if (args.work_dir == args.output or args.work_dir in args.output.parents
            or args.output in args.work_dir.parents):
        parser.error("--work-dir and final --output must be separate directories")
    if args.output.exists() and any(p.name not in {"Overlay", "Result"} for p in args.output.iterdir()):
        parser.error("Final Output may contain only Overlay/ and Result/; archive old outputs first")
    if args.export_only:
        export_demo(args.work_dir, args.output, args.input,
                    marker_half_size=args.marker_half_size, marker_width=args.marker_width,
                    transform=args.transform)
        return

    folders = [args.input / name for name in FREQUENCY_FOLDERS]
    if args.tie_points is None:
        load_transform(args.transform)
    missing = [str(path) for path in folders if not path.is_dir()]
    if missing:
        parser.error("Missing frequency folders: " + ", ".join(missing))

    args.work_dir.mkdir(parents=True, exist_ok=True)
    batch_out = args.work_dir / "result_batch"
    lines_out = args.work_dir / "lines3d_out"
    image_out = args.work_dir / "result_image"
    anomaly_out = args.work_dir / "anomaly_rb"

    run_stage(
        "1/5 batch preprocessing",
        [
            sys.executable,
            str(CODE / "gpr_batch_plane.py"),
            *map(str, folders),
            "--out",
            str(batch_out),
        ],
    )

    transform = args.transform
    if args.tie_points is not None:
        if args.x1 is None or args.x2 is None or args.plane_y1 is None:
            parser.error("--tie-points requires --x1, --x2 and the actual --plane-y1")
        tie_out = args.work_dir / "tie_out"
        run_stage(
            "2a/5 tie-point rectification",
            [
                sys.executable,
                str(CODE / "rectify_tiepoints.py"),
                "--plane_csv",
                str(batch_out / "line_endpoints.csv"),
                "--points_csv",
                str(args.tie_points),
                "--x1",
                str(args.x1),
                "--x2",
                str(args.x2),
                "--y0",
                str(args.y0),
                "--plane-y1",
                str(args.plane_y1),
                "--out",
                str(tie_out),
            ],
        )
        transform = tie_out / "transform.json"
    elif not transform.is_file():
        parser.error(f"Transform not found: {transform}")

    run_stage(
        "2/5 3D line mapping",
        [
            sys.executable,
            str(CODE / "apply_transform_lines.py"),
            "--plane_csv",
            str(batch_out / "line_endpoints.csv"),
            "--transform",
            str(transform),
            "--out",
            str(lines_out),
        ],
    )

    run_stage(
        "3/5 pixel-aligned B-scan export",
        [
            sys.executable,
            str(CODE / "export_result_images.py"),
            "--folders",
            *map(str, folders),
            "--lines3d",
            str(lines_out / "lines_3d.csv"),
            "--out",
            str(image_out),
        ],
    )

    detection_command = [
        sys.executable,
        str(CODE / "detect_anomalies_rulebased.py"),
        "--folders",
        *map(str, folders),
        "--lines3d",
        str(lines_out / "lines_3d.csv"),
        "--out",
        str(anomaly_out),
        "--incoh_win",
        str(args.incoh_win),
        "--tmask_top_ns",
        str(args.tmask_top_ns),
        "--tmask_bottom",
        str(args.tmask_bottom),
        "--min_area",
        str(args.min_area),
        "--marker-half-size",
        str(args.marker_half_size),
        "--marker-width",
        str(args.marker_width),
    ]
    if args.ablation:
        detection_command.append("--ablation")
    if args.incoh_q is not None:
        detection_command.extend(["--incoh_q", str(args.incoh_q)])
    run_stage("4/5 rule-based detection", detection_command)

    run_stage(
        "5/5 display Overlays and CSV",
        [
            sys.executable,
            str(CODE / "export_demo.py"),
            "--source-output",
            str(args.work_dir),
            "--output",
            str(args.output),
            "--input",
            str(args.input),
            "--transform",
            str(transform),
            "--marker-half-size",
            str(args.marker_half_size),
            "--marker-width",
            str(args.marker_width),
        ],
    )
    print(f"\nCompleted. Outputs: {args.output}")
    print(f"Intermediate results: {args.work_dir}")


if __name__ == "__main__":
    main()
