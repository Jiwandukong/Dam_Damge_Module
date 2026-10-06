"""Infer raw measurement photos with the existing Release-backed demo model."""
import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from asset_paths import cached_artifact, stage_model_record

CODE = Path(__file__).resolve().parent
SENSOR = CODE.parent

def main(argv=None):
    parser = argparse.ArgumentParser(description="Rawdata photos -> Release model inference -> damage results")
    parser.add_argument("--images", type=Path, default=SENSOR / "Rawdata/images")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-record", type=Path)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--asset-manifest", type=Path, default=CODE / "assets.yaml")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--ray-backend", choices=["auto", "warp", "trimesh"], default="auto")
    parser.add_argument("--warp-device", default="cpu")
    parser.add_argument("--save-diagnostics", action="store_true")
    args = parser.parse_args(argv)
    checkpoint = cached_artifact(CODE / "model_release.json")
    mesh = args.mesh or cached_artifact(CODE / "geometry_release.json")
    if args.model_record is None and not checkpoint.is_file():
        parser.error("Download the model first: python Code/download_model.py")
    if not mesh.is_file():
        parser.error("Download the geometry first: python Code/download_geometry.py")
    model_record = args.model_record or stage_model_record(checkpoint)
    output = args.output or SENSOR / "Output/runs" / datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
    from demo512.prediction_pipeline import export_predictions
    result = export_predictions(args.images, output, model_record, mesh, args.asset_manifest,
                                device=args.device, threshold=args.threshold,
                                ray_backend=args.ray_backend, warp_device=args.warp_device,
                                save_diagnostics=args.save_diagnostics)
    print(result)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
