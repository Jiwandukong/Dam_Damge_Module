"""Download checksum-pinned demo LAS/outputs from the public GitHub Release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from urllib.request import Request, urlopen
import zipfile

sys.dont_write_bytecode = True

from output_rules import apply_exclusions

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORK = ROOT.parent / "unmodified_multibeam/Multibeam_work"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matches(path, expected):
    return path.is_file() and path.stat().st_size == expected["bytes"] and sha256(path) == expected["sha256"]


def ensure_assets(scope="all", work=None, manifest=None):
    work = Path(work or os.environ.get("MULTIBEAM_WORK_DIR", DEFAULT_WORK)).expanduser().resolve()
    manifest = manifest or json.loads((Path(__file__).parent / "assets.json").read_text())
    cache = work / "downloads"
    for asset in manifest["assets"]:
        if scope != "all" and asset["scope"] != scope:
            continue
        if all(matches(ROOT / member["path"], member) for member in asset["members"]):
            print(f"{asset['name']}: local files verified", flush=True)
            continue
        cache.mkdir(parents=True, exist_ok=True)
        archive = cache / asset["name"]
        if not archive.is_file() or archive.stat().st_size != asset["bytes"] or sha256(archive) != asset["sha256"]:
            temporary = archive.with_suffix(archive.suffix + ".part")
            print(f"Downloading {asset['name']}...", flush=True)
            request = Request(asset["url"], headers={"User-Agent": "Multibeam-Demo/1.0"})
            with urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=4 * 1024 * 1024)
            if temporary.stat().st_size != asset["bytes"] or sha256(temporary) != asset["sha256"]:
                raise RuntimeError(f"Release checksum mismatch: {asset['name']}")
            os.replace(temporary, archive)
        with zipfile.ZipFile(archive) as packed:
            allowed = {member["path"] for member in asset["members"]}
            if set(packed.namelist()) != allowed:
                raise RuntimeError(f"Unexpected files in Release ZIP: {asset['name']}")
            for member in asset["members"]:
                destination = (ROOT / member["path"]).resolve()
                if ROOT not in destination.parents:
                    raise RuntimeError("Release ZIP path escapes project directory")
                if matches(destination, member):
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".download")
                with packed.open(member["path"]) as source, temporary.open("wb") as stream:
                    shutil.copyfileobj(source, stream, length=4 * 1024 * 1024)
                if not matches(temporary, member):
                    raise RuntimeError(f"Extracted file checksum mismatch: {member['path']}")
                os.replace(temporary, destination)
        print(f"{asset['name']}: download and extraction verified", flush=True)
    if scope in ("all", "results"):
        apply_exclusions(ROOT / "Output")


def main():
    parser = argparse.ArgumentParser(description="원본 LAS와 손상 산출물 Release 다운로드")
    parser.add_argument("--scope", choices=["all", "raw", "results"], default="all")
    parser.add_argument("--work-dir", type=Path, default=Path(os.environ.get("MULTIBEAM_WORK_DIR", DEFAULT_WORK)))
    args = parser.parse_args()
    ensure_assets(args.scope, args.work_dir)


if __name__ == "__main__":
    main()
