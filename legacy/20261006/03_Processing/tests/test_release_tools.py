from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_split_manifest_and_local_reconstruction(tmp_path):
    source = tmp_path / "checkpoint_final.pt"
    source.write_bytes(bytes(range(251)) * 41)
    parts = tmp_path / "parts"
    manifest = tmp_path / "manifest.json"
    package_script = PROJECT_ROOT / "03_Processing/scripts/package_checkpoint.py"
    download_script = PROJECT_ROOT / "03_Processing/scripts/download_checkpoint.py"
    subprocess.run(
        [
            sys.executable,
            str(package_script),
            "--checkpoint",
            str(source),
            "--output-dir",
            str(parts),
            "--manifest",
            str(manifest),
            "--part-size-bytes",
            "1024",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["part_count"] > 1
    assert sum(part["size_bytes"] for part in data["parts"]) == source.stat().st_size

    reconstructed = tmp_path / "downloaded" / "checkpoint_final.pt"
    subprocess.run(
        [
            sys.executable,
            str(download_script),
            "--manifest",
            str(manifest),
            "--parts-dir",
            str(parts),
            "--output",
            str(reconstructed),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert reconstructed.read_bytes() == source.read_bytes()
    assert hashlib.sha256(reconstructed.read_bytes()).hexdigest() == data["sha256"]
