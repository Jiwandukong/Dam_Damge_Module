"""Small release parts exercise downloaders from a standalone 06-only copy."""

from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import pytest


DEMO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def release(tmp_path):
    # Relocate the actual scripts, so their default paths cannot accidentally
    # read or overwrite the real multi-GB checkpoint during the test.
    project = tmp_path / "공백 있는 저장소"
    demo = project / "06.시연용"
    wrapper = demo / "code/모델받기.py"
    for source, destination in (
        (DEMO_ROOT / "code/모델받기.py", wrapper),
        (DEMO_ROOT / "code/demo512/release_download.py", demo / "code/demo512/release_download.py"),
        (DEMO_ROOT / "code/demo512/__init__.py", demo / "code/demo512/__init__.py"),
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    parts_dir = tmp_path / "다운로드한 조각"
    parts_dir.mkdir()
    chunks = (b"demo-checkpoint-\x00\x01", b"second-part\xff\xfe")
    parts = []
    for index, chunk in enumerate(chunks, 1):
        name = f"sam3_demo512.pt.part{index:03d}"
        (parts_dir / name).write_bytes(chunk)
        parts.append({"name": name, "size_bytes": len(chunk),
                      "sha256": hashlib.sha256(chunk).hexdigest()})
    content = b"".join(chunks)
    manifest = {
        "artifact": "sam3_demo512_학습완료.pt",
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "download_base_url": "REPLACE_FOR_TEST",
        "parts": parts,
    }
    manifest_path = demo / "02.시연모델/SAM3/release_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = demo / "02.시연모델/SAM3/sam3_demo512_학습완료.pt"
    outside = tmp_path / "outside"
    outside.mkdir()
    return {"wrapper": wrapper, "parts": parts_dir, "manifest": manifest,
            "manifest_path": manifest_path, "output": output,
            "content": content, "cwd": outside}


def run_wrapper(release, *args):
    return subprocess.run(
        [sys.executable, "-S", str(release["wrapper"]), *map(str, args)],
        cwd=release["cwd"], capture_output=True, text=True, timeout=15,
    )


def test_defaults_reconstruct_from_external_working_directory(release):
    result = run_wrapper(release, "--parts-dir", release["parts"])
    assert result.returncode == 0, result.stderr
    assert release["output"].read_bytes() == release["content"]
    assert "reconstructed and verified" in result.stdout
    assert not (release["cwd"] / "02.시연모델").exists()
    assert not (release["wrapper"].parents[2] / "03_Processing").exists()


def test_custom_relative_output_and_parts_directory(release):
    result = run_wrapper(
        release, "--parts-dir", "../다운로드한 조각",
        "--output", "new folder/검증 모델.pt",
        "--manifest", release["manifest_path"],
    )
    assert result.returncode == 0, result.stderr
    assert (release["cwd"] / "new folder/검증 모델.pt").read_bytes() == release["content"]
    assert not release["output"].exists()


def test_matching_existing_checkpoint_verified_without_parts(release):
    release["output"].parent.mkdir(parents=True, exist_ok=True)
    release["output"].write_bytes(release["content"])
    before = release["output"].stat().st_mtime_ns
    result = run_wrapper(release, "--parts-dir", "nonexistent-parts")
    assert result.returncode == 0, result.stderr
    assert "already verified" in result.stdout
    assert release["output"].stat().st_mtime_ns == before


@pytest.mark.parametrize("same_size", [True, False])
def test_different_existing_checkpoint_is_not_overwritten(release, same_size):
    release["output"].parent.mkdir(parents=True, exist_ok=True)
    original = b"x" * (len(release["content"]) if same_size else 1)
    release["output"].write_bytes(original)
    result = run_wrapper(release, "--parts-dir", release["parts"])
    assert result.returncode != 0
    assert "output exists but is not the verified artifact" in result.stderr
    assert release["output"].read_bytes() == original


@pytest.mark.parametrize("same_size", [True, False])
def test_corrupt_release_part_is_rejected(release, same_size):
    part = release["parts"] / "sam3_demo512.pt.part001"
    part.write_bytes(b"x" * (part.stat().st_size if same_size else 1))
    result = run_wrapper(release, "--parts-dir", release["parts"])
    assert result.returncode != 0
    expected = "part SHA256 mismatch" if same_size else "part size mismatch"
    assert expected in result.stderr
    assert not release["output"].exists()


def test_reconstructed_checkpoint_hash_is_verified(release):
    release["manifest"]["sha256"] = "0" * 64
    release["manifest_path"].write_text(json.dumps(release["manifest"]), encoding="utf-8")
    result = run_wrapper(release, "--parts-dir", release["parts"])
    assert result.returncode != 0
    assert "reconstructed checkpoint SHA256 mismatch" in result.stderr
    assert not release["output"].exists()
    assert not list(release["output"].parent.glob("*.partial"))


def test_download_from_release_base_url(release):
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = partial(QuietHandler, directory=str(release["parts"]))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        result = run_wrapper(release, "--base-url", f"http://127.0.0.1:{server.server_port}")
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    assert result.returncode == 0, result.stderr
    assert release["output"].read_bytes() == release["content"]
    assert "downloading" in result.stdout


def test_help_does_not_require_manifest_or_installation(release):
    release["manifest_path"].unlink()
    result = run_wrapper(release, "--help")
    assert result.returncode == 0, result.stderr
    assert "--parts-dir" in result.stdout
    assert "--output" in result.stdout
    assert "--force" not in result.stdout


def test_force_overwrite_is_not_exposed(release):
    result = run_wrapper(release, "--force")
    assert result.returncode != 0
    assert "unrecognized arguments: --force" in result.stderr
    assert not release["output"].exists()


@pytest.mark.parametrize("name", ["../escape.pt", "/absolute.pt", "part\\escape.pt", ".."])
def test_unsafe_release_part_names_are_rejected(release, name):
    release["manifest"]["parts"][0]["name"] = name
    release["manifest_path"].write_text(json.dumps(release["manifest"]), encoding="utf-8")
    result = run_wrapper(release, "--parts-dir", release["parts"])
    assert result.returncode != 0
    assert "unique plain ASCII filenames" in result.stderr
    assert not release["output"].exists()


def test_output_symlink_is_rejected_without_touching_its_target(release):
    target = release["cwd"] / "existing.pt"
    target.write_bytes(release["content"])
    release["output"].parent.mkdir(parents=True, exist_ok=True)
    release["output"].symlink_to(target)
    result = run_wrapper(release, "--parts-dir", release["parts"])
    assert result.returncode != 0 and "symbolic link" in result.stderr
    assert release["output"].is_symlink()
    assert target.read_bytes() == release["content"]


def test_part_symlink_is_rejected(release):
    part = release["parts"] / "sam3_demo512.pt.part001"
    outside = release["cwd"] / "outside.part"
    outside.write_bytes(part.read_bytes())
    part.unlink()
    part.symlink_to(outside)
    result = run_wrapper(release, "--parts-dir", release["parts"])
    assert result.returncode != 0 and "non-symlink" in result.stderr
    assert not release["output"].exists()


def test_later_http_failure_preserves_verified_part_and_removes_temporary_files(release):
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    second_part = release["parts"] / "sam3_demo512.pt.part002"
    second_part.write_bytes(b"x" * second_part.stat().st_size)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(release["parts"])))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        result = run_wrapper(release, "--base-url", f"http://127.0.0.1:{server.server_port}")
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    assert result.returncode != 0 and "download SHA256 mismatch" in result.stderr
    assert not release["output"].exists()
    cache = release["output"].parent / ".release_parts"
    assert (cache / "sam3_demo512.pt.part001").read_bytes() == (
        release["parts"] / "sam3_demo512.pt.part001").read_bytes()
    assert not (cache / "sam3_demo512.pt.part002").exists()
    assert not list(release["output"].parent.rglob("*.partial"))
