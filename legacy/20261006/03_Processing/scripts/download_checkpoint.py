#!/usr/bin/env python3
"""Download verified release parts and reconstruct checkpoint_final.pt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE = PROJECT_ROOT / "02_Model/sam3_aug512/release"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "02_Model/sam3_aug512/checkpoints/checkpoint_final.pt"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default=str(DEFAULT_RELEASE / "release_manifest.json")
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--parts-dir",
        default=None,
        help="Use already downloaded local parts instead of HTTP.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    if output.exists():
        expected_size = int(manifest["size_bytes"])
        if output.stat().st_size == expected_size and sha256_file(output) == manifest["sha256"]:
            print(f"already verified: {output}")
            return 0
        if not args.force:
            raise FileExistsError(
                f"output exists but is not the verified artifact; pass --force: {output}"
            )

    cache = output.parent / ".release_parts"
    cache.mkdir(parents=True, exist_ok=True)
    base_url = (args.base_url or manifest.get("download_base_url") or "").rstrip("/")
    if args.parts_dir is None and "REPLACE_" in base_url:
        raise ValueError("release URL is not configured; pass --base-url")

    verified_parts: list[Path] = []
    for part in manifest["parts"]:
        name = str(part["name"])
        expected_size = int(part["size_bytes"])
        expected_sha = str(part["sha256"])
        if args.parts_dir is not None:
            part_path = Path(args.parts_dir).resolve() / name
        else:
            part_path = cache / name
            if not part_path.exists():
                url = f"{base_url}/{name}"
                print(f"downloading {url}", flush=True)
                request = Request(url, headers={"User-Agent": "UAV_RGB-checkpoint-downloader/1.0"})
                temporary = part_path.with_suffix(part_path.suffix + ".partial")
                with urlopen(request) as response, temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream, length=16 * 1024 * 1024)
                temporary.replace(part_path)
        if not part_path.is_file():
            raise FileNotFoundError(part_path)
        if part_path.stat().st_size != expected_size:
            raise ValueError(f"part size mismatch: {part_path}")
        if sha256_file(part_path) != expected_sha:
            raise ValueError(f"part SHA256 mismatch: {part_path}")
        verified_parts.append(part_path)
        print(f"verified {name}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".partial")
    if temporary_output.exists():
        temporary_output.unlink()
    with temporary_output.open("xb") as destination:
        for part_path in verified_parts:
            with part_path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    if temporary_output.stat().st_size != int(manifest["size_bytes"]):
        raise ValueError("reconstructed checkpoint size mismatch")
    if sha256_file(temporary_output) != manifest["sha256"]:
        raise ValueError("reconstructed checkpoint SHA256 mismatch")
    if output.exists():
        output.unlink()
    temporary_output.replace(output)
    print(f"reconstructed and verified: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
