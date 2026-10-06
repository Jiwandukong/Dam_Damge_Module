#!/usr/bin/env python3
"""Verify byte identity and optionally strict-load the AUG512 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "03_Processing" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from uav_rgb.models import inspect_checkpoint_file, load_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(
            PROJECT_ROOT
            / "02_Model/sam3_aug512/checkpoints/checkpoint_final.pt"
        ),
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Also build pinned SAM3 and load all 1,134 tensors with strict=True.",
    )
    args = parser.parse_args()
    report = {
        key: value
        for key, value in inspect_checkpoint_file(args.checkpoint).items()
        if not key.startswith("_stat_")
    }
    if args.strict_load:
        model, strict_report = load_model(args.checkpoint, device="cpu")
        report.update(strict_report)
        del model
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report.get("checkpoint_identity_verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
