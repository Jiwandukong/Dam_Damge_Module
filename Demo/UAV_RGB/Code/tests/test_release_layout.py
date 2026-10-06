"""Release assets restore from local parts into an isolated external cache."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SOURCE = Path(__file__).resolve().parents[1]


@pytest.fixture
def release(tmp_path):
    code = tmp_path / "standalone/Code"
    code.mkdir(parents=True)
    for name in ("asset_paths.py", "release_download.py", "download_model.py", "download_geometry.py"):
        shutil.copyfile(SOURCE / name, code / name)
    parts = tmp_path / "parts"
    parts.mkdir()
    content = b"checked model payload" * 20
    pieces = [content[:101], content[101:]]
    manifest = {
        "artifact": "demo.pt", "release_tag": "demo-fixture-v1",
        "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
        "parts": [],
    }
    for index, piece in enumerate(pieces, 1):
        name = f"demo.pt.part{index:03d}"
        (parts / name).write_bytes(piece)
        manifest["parts"].append({"name": name, "size_bytes": len(piece), "sha256": hashlib.sha256(piece).hexdigest()})
    (code / "model_release.json").write_text(json.dumps(manifest))
    (code / "training_record.json").write_text(json.dumps({"checkpoint": {
        "file": "demo.pt", "bytes": len(content), "sha256": manifest["sha256"]}}))
    outside = tmp_path / "unrelated working folder"
    outside.mkdir()
    environment = {**os.environ, "XDG_CACHE_HOME": str(tmp_path / "external-cache")}
    return code, parts, content, outside, environment


def run(release, *args):
    code, _, _, outside, environment = release
    return subprocess.run([sys.executable, "-S", str(code / "download_model.py"), *map(str, args)],
                          cwd=outside, env=environment, text=True, capture_output=True)


def test_default_cache_restore_and_cached_record_are_independent_of_cwd(release):
    code, parts, content, outside, environment = release
    result = run(release, "--parts-dir", parts)
    assert result.returncode == 0, result.stderr
    target = Path(environment["XDG_CACHE_HOME"]) / "dam_damage/demo-fixture-v1/demo.pt"
    assert target.read_bytes() == content
    record = json.loads((target.parent / "training_record.json").read_text())
    assert record["checkpoint"]["file"] == target.name
    assert not list(code.rglob("*.pt"))
    assert not list(outside.iterdir())


def test_custom_output_and_bad_existing_asset_are_preserved(release):
    _, parts, content, outside, _ = release
    result = run(release, "--parts-dir", parts, "--output", "custom/model.pt")
    assert result.returncode == 0, result.stderr
    target = outside / "custom/model.pt"
    assert target.read_bytes() == content
    record = json.loads((target.parent / "training_record.json").read_text())
    assert record["checkpoint"]["file"] == "model.pt"
    target.write_bytes(b"existing unrelated checkpoint")
    result = run(release, "--parts-dir", parts, "--output", target)
    assert result.returncode != 0
    assert target.read_bytes() == b"existing unrelated checkpoint"


def test_corrupt_part_does_not_publish_model(release):
    _, parts, _, _, environment = release
    (parts / "demo.pt.part001").write_bytes(b"corrupt")
    result = run(release, "--parts-dir", parts)
    assert result.returncode != 0
    assert not list(Path(environment["XDG_CACHE_HOME"]).rglob("*.pt"))


def test_help_without_metadata_or_site_packages(release):
    code, _, _, _, _ = release
    (code / "model_release.json").unlink()
    result = run(release, "--help")
    assert result.returncode == 0, result.stderr
    assert "--parts-dir" in result.stdout


def test_release_metadata_preserves_original_artifact_identity():
    model = json.loads((SOURCE / "model_release.json").read_text())
    geometry = json.loads((SOURCE / "geometry_release.json").read_text())
    assert model["sha256"] == "eb6489ba6e5aa9b307aef1f1d2b86037ff91d40e22f0deb98e8c704d286079d5"
    assert geometry["sha256"] == "a7566c8d8f70de91db18c70ae5f404b0f7885e679a1027fd3e1936afd2d1e470"
    for manifest in (model, geometry):
        assert "/Dam_Damge_Module/" in manifest["download_base_url"]
        assert sum(part["size_bytes"] for part in manifest["parts"]) == manifest["size_bytes"]
