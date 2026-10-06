#!/usr/bin/env python3
"""Run a complete UAV RGB mission and create masks, coordinates and Excel."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "03_Processing" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from uav_rgb.pipeline import UAVRGBPipeline


def parse_args() -> argparse.Namespace:
    mission = PROJECT_ROOT / "01_RawData" / "missions" / "DJI_202507021616_left03"
    parser = argparse.ArgumentParser(
        description=(
            "AUG512 SAM3 detection -> EPSG:5186 OBJ ray mapping -> rough local "
            "GSD -> CSV/XLSX report"
        )
    )
    parser.add_argument(
        "--images",
        default=str(mission / "images"),
        help="One image or a directory of source-resolution UAV images.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            PROJECT_ROOT
            / "02_Model"
            / "sam3_aug512"
            / "checkpoints"
            / "checkpoint_final.pt"
        ),
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "03_Processing" / "configs" / "daechung_aug512.yaml"),
    )
    parser.add_argument(
        "--mesh",
        default=str(
            PROJECT_ROOT
            / "01_RawData"
            / "geometry"
            / "daecheong_dam_epsg5186_zup.obj"
        ),
    )
    parser.add_argument(
        "--mrk",
        default=str(
            mission
            / "navigation"
            / "DJI_202507021616_003_Timestamp.MRK"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Fresh output directory. Default: 04_Output/runs/<timestamp>.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--mesh-ray-backend",
        choices=["auto", "warp", "trimesh"],
        default="auto",
    )
    parser.add_argument("--warp-device", default="cuda:0")
    parser.add_argument(
        "--representative-mode", choices=["centroid", "median"], default="centroid"
    )
    parser.add_argument("--instance-samples", type=int, default=128)
    parser.add_argument("--node-samples", type=int, default=256)
    parser.add_argument("--ray-batch-size", type=int, default=256)
    parser.add_argument("--min-instance-area-px", type=int, default=8)
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT
        / "04_Output"
        / "runs"
        / datetime.now().strftime("left03_aug512_%Y%m%d_%H%M%S")
    )
    pipeline = UAVRGBPipeline(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        mesh_path=args.mesh,
        mrk_path=args.mrk,
        device=args.device,
        verify_checkpoint_sha256=True,
        mesh_ray_backend=args.mesh_ray_backend,
        warp_device=args.warp_device,
        representative_mode=args.representative_mode,
        instance_sample_count=args.instance_samples,
        node_sample_count=args.node_samples,
        ray_batch_size=args.ray_batch_size,
        min_instance_area_px=args.min_instance_area_px,
        profile=args.profile,
        progress=lambda message: print(message, flush=True),
    )
    result = pipeline.run(images=args.images, output_dir=output_dir)
    print(f"Output: {result['output_dir']}")
    print(f"Excel: {result['combined_paths']['excel']}")
    print(f"Damage rows: {len(result['rows'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
