#!/usr/bin/env python3
"""Split the 10 GB checkpoint into GitHub-Release-sized verified parts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "02_Model/sam3_aug512/checkpoints/checkpoint_final.pt"
)
DEFAULT_RELEASE = PROJECT_ROOT / "02_Model/sam3_aug512/release"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_RELEASE / "dist"))
    parser.add_argument(
        "--manifest", default=str(DEFAULT_RELEASE / "release_manifest.json")
    )
    parser.add_argument("--part-size-bytes", type=int, default=1_900_000_000)
    parser.add_argument(
        "--download-base-url",
        default=(
            "https://github.com/Jiwandukong/UAV_RGB_Damage_Module/"
            "releases/download/sam3-aug512-v1"
        ),
    )
    args = parser.parse_args()

    source = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.part_size_bytes <= 0 or args.part_size_bytes >= 2_000_000_000:
        raise ValueError("part-size-bytes must be positive and below 2,000,000,000")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob(f"{source.name}.part*"))
    if existing:
        raise FileExistsError(
            f"refusing to overwrite {len(existing)} existing release parts in {output_dir}"
        )

    whole_digest = hashlib.sha256()
    part_rows: list[dict[str, object]] = []
    part_number = 0
    with source.open("rb") as input_stream:
        while True:
            first = input_stream.read(min(16 * 1024 * 1024, args.part_size_bytes))
            if not first:
                break
            part_number += 1
            name = f"{source.name}.part{part_number:03d}"
            part_path = output_dir / name
            part_digest = hashlib.sha256()
            written = 0
            with part_path.open("xb") as output_stream:
                chunk = first
                while chunk:
                    output_stream.write(chunk)
                    part_digest.update(chunk)
                    whole_digest.update(chunk)
                    written += len(chunk)
                    remaining = args.part_size_bytes - written
                    if remaining <= 0:
                        break
                    chunk = input_stream.read(min(16 * 1024 * 1024, remaining))
            part_rows.append(
                {
                    "name": name,
                    "size_bytes": written,
                    "sha256": part_digest.hexdigest(),
                }
            )
            print(f"created {name}: {written:,} bytes", flush=True)

    manifest = {
        "schema_version": "1.0",
        "artifact": source.name,
        "size_bytes": source.stat().st_size,
        "sha256": whole_digest.hexdigest(),
        "part_size_bytes": args.part_size_bytes,
        "part_count": len(part_rows),
        "release_tag": "sam3-aug512-v1",
        "download_base_url": args.download_base_url.rstrip("/"),
        "parts": part_rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
