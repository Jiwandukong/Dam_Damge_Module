"""The standalone 06 folder downloads an independently verified original OBJ."""

from contextlib import contextmanager
from functools import partial
import hashlib
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import pytest


DEMO_ROOT = Path(__file__).resolve().parents[2]
OBJ_NAME = "daecheong_dam_epsg5186_zup.obj"


@pytest.fixture
def geometry(tmp_path):
    demo = tmp_path / "standalone 06"
    for relative in ("code/3D모델받기.py", "code/demo512/release_download.py", "code/demo512/__init__.py"):
        target = demo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DEMO_ROOT / relative, target)
    directory = demo / "02.시연모델/댐3D모델"
    directory.mkdir(parents=True)
    parts = tmp_path / "downloaded"
    parts.mkdir()
    content = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    (parts / OBJ_NAME).write_bytes(content)
    identity = {"size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    manifest = {"artifact": OBJ_NAME, **identity, "download_base_url": "REPLACE_FOR_TEST",
                "parts": [{"name": OBJ_NAME, **identity}]}
    manifest_path = directory / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    return {"demo": demo, "wrapper": demo / "code/3D모델받기.py", "parts": parts,
            "manifest_path": manifest_path, "output": directory / OBJ_NAME,
            "content": content, "cwd": outside}


def run(geometry, *args):
    return subprocess.run(
        [sys.executable, "-S", str(geometry["wrapper"]), *map(str, args)],
        cwd=geometry["cwd"], capture_output=True, text=True, timeout=15,
    )


@contextmanager
def serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def test_default_obj_location_with_only_standalone_06_and_local_file(geometry):
    result = run(geometry, "--parts-dir", geometry["parts"])
    assert result.returncode == 0, result.stderr
    assert geometry["output"].read_bytes() == geometry["content"]
    assert not geometry["output"].is_symlink()
    assert not (geometry["demo"].parent / "01_RawData").exists()
    assert not (geometry["demo"].parent / "03_Processing").exists()


def test_obj_http_download_uses_custom_path_without_duplicate_cached_copy(geometry):
    with serve(partial(QuietHandler, directory=str(geometry["parts"]))) as base_url:
        result = run(geometry, "--base-url", base_url, "--output", "new/geometry.obj")
    assert result.returncode == 0, result.stderr
    assert (geometry["cwd"] / "new/geometry.obj").read_bytes() == geometry["content"]
    assert not (geometry["cwd"] / "new/.release_parts").exists()
    assert not list(geometry["cwd"].rglob("*.partial"))


def test_matching_existing_obj_is_verified_without_download(geometry):
    geometry["output"].write_bytes(geometry["content"])
    before = geometry["output"].stat().st_mtime_ns
    result = run(geometry)
    assert result.returncode == 0, result.stderr
    assert "already verified" in result.stdout
    assert geometry["output"].stat().st_mtime_ns == before


def test_wrong_existing_obj_is_preserved(geometry):
    geometry["output"].write_bytes(b"wrong geometry")
    result = run(geometry, "--parts-dir", geometry["parts"])
    assert result.returncode != 0 and "not the verified artifact" in result.stderr
    assert geometry["output"].read_bytes() == b"wrong geometry"


def test_corrupt_obj_download_is_not_published(geometry):
    (geometry["parts"] / OBJ_NAME).write_bytes(b"x" * len(geometry["content"]))
    with serve(partial(QuietHandler, directory=str(geometry["parts"]))) as base_url:
        result = run(geometry, "--base-url", base_url)
    assert result.returncode != 0 and "SHA256 mismatch" in result.stderr
    assert not geometry["output"].exists()
    assert not list(geometry["output"].parent.rglob("*.partial"))


def test_interrupted_obj_download_leaves_no_partial_or_final_file(geometry):
    class TruncatedHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(geometry["content"])))
            self.end_headers()
            self.wfile.write(geometry["content"][:5])
            self.close_connection = True

        def log_message(self, *args):
            pass

    with serve(TruncatedHandler) as base_url:
        result = run(geometry, "--base-url", base_url)
    assert result.returncode != 0
    assert not geometry["output"].exists()
    assert not list(geometry["output"].parent.rglob("*.partial"))


def test_help_requires_no_manifest_model_or_external_package(geometry):
    geometry["manifest_path"].unlink()
    result = run(geometry, "--help")
    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout and "--parts-dir" in result.stdout
    assert "--force" not in result.stdout


def test_published_obj_manifest_has_the_original_mesh_identity():
    path = DEMO_ROOT / "02.시연모델/댐3D모델/release_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["artifact"] == OBJ_NAME
    assert manifest["size_bytes"] == 184020785
    assert manifest["sha256"] == "a7566c8d8f70de91db18c70ae5f404b0f7885e679a1027fd3e1936afd2d1e470"
    assert manifest["parts"] == [{"name": OBJ_NAME, "size_bytes": manifest["size_bytes"],
                                  "sha256": manifest["sha256"]}]
    assert manifest["download_base_url"] == (
        "https://github.com/Jiwandukong/UAV_RGB_Damage_Module/releases/download/daecheong-dam-geometry-v1")


def test_mesh_contract_remains_relative_to_the_standalone_demo():
    # Keep the runtime downloader stdlib-only; this textual fixture does not
    # require PyYAML either. Geometry runtime tests validate parsed semantics.
    path = DEMO_ROOT / "02.시연모델/댐3D모델/assets.yaml"
    text = path.read_text(encoding="utf-8")
    assert f"repository_path: 02.시연모델/댐3D모델/{OBJ_NAME}" in text
    assert "horizontal_crs: EPSG:5186" in text and "vertical_axis: Z-up" in text
    assert "vertex_count_loaded: 1006692" in text and "face_count_loaded: 2012989" in text
    assert "01_RawData" not in text and "02_Model" not in text
