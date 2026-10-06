"""Standard-library-only verified downloads for this standalone demo folder.

Final files are published only after byte-size and SHA256 verification. Atomic
hard-link creation refuses to replace a file created by another process, and
temporary files from failed downloads/reconstruction are removed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


CHUNK_SIZE = 16 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(value: dict, description: str) -> tuple[int, str]:
    size, digest = value.get("size_bytes"), value.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"{description} size_bytes must be a positive integer")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{description} SHA256 must be 64 lowercase hexadecimal characters")
    return size, digest


def _existing_output(path: Path, identity: tuple[int, str]) -> bool:
    if path.is_symlink():
        raise FileExistsError(f"output must not be a symbolic link: {path}")
    if not path.exists():
        return False
    size, digest = identity
    if path.is_file() and path.stat().st_size == size and sha256_file(path) == digest:
        print(f"already verified: {path}", flush=True)
        return True
    raise FileExistsError(f"output exists but is not the verified artifact: {path}")


def _verify(path: Path, identity: tuple[int, str], description: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{description} must be a regular non-symlink file: {path}")
    size, digest = identity
    if path.stat().st_size != size:
        raise ValueError(f"{description} size mismatch: {path}")
    if sha256_file(path) != digest:
        raise ValueError(f"{description} SHA256 mismatch: {path}")


def _temporary(parent: Path, name: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{name}.", suffix=".partial", dir=parent)
    os.close(descriptor)
    return Path(name)


def _publish(temporary: Path, destination: Path, identity: tuple[int, str]) -> None:
    # Temporary and destination are on the same filesystem. Unlike replace or
    # rename, link cannot silently overwrite a concurrently created output.
    try:
        os.link(temporary, destination)
    except FileExistsError:
        if not _existing_output(destination, identity):
            raise


def _download(url: str, destination: Path, identity: tuple[int, str]) -> None:
    expected_size, expected_digest = identity
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(destination.parent, destination.name)
    try:
        request = Request(url, headers={"User-Agent": "UAV_RGB-Demo512-downloader/1.0"})
        digest, received = hashlib.sha256(), 0
        print(f"downloading {url}", flush=True)
        with urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
            while chunk := response.read(CHUNK_SIZE):
                received += len(chunk)
                if received > expected_size:
                    raise ValueError(f"download size mismatch: {destination}")
                digest.update(chunk)
                stream.write(chunk)
        if received != expected_size:
            raise ValueError(f"download size mismatch: {destination}")
        if digest.hexdigest() != expected_digest:
            raise ValueError(f"download SHA256 mismatch: {destination}")
        _publish(temporary, destination, identity)
    finally:
        temporary.unlink(missing_ok=True)


def restore_release(manifest_path: str | Path, output: str | Path, *,
                    parts_dir: str | Path | None = None,
                    base_url: str | None = None) -> Path:
    """Download/restore a manifest's ordered parts without replacing other files.

Single-file releases (the OBJ) are downloaded directly to an atomic temporary
file. Multi-part checkpoints keep verified parts in ``.release_parts`` so an
interrupted later part does not require downloading earlier parts again.
    """
    manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("release manifest must be a JSON object")
    identity = _identity(manifest, "artifact")
    parts = manifest.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("release manifest must contain one or more parts")
    names = set()
    for part in parts:
        if not isinstance(part, dict):
            raise ValueError("release parts must be JSON objects")
        name = part.get("name")
        if (not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None
                or name in names):
            raise ValueError("release part names must be unique plain ASCII filenames")
        names.add(name)
        _identity(part, "part")
    if sum(part["size_bytes"] for part in parts) != identity[0]:
        raise ValueError("release part sizes do not sum to artifact size")
    if len(parts) == 1 and _identity(parts[0], "part") != identity:
        raise ValueError("single release part identity must match its artifact")

    # Do not resolve the final component: that would hide an existing symlink.
    destination = Path(output).expanduser().absolute()
    if _existing_output(destination, identity):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected_url = base_url or manifest.get("download_base_url") or ""
    if parts_dir is None:
        if not isinstance(selected_url, str):
            raise ValueError("release download URL must be text")
        selected_url = selected_url.rstrip("/")
        parsed = urlsplit(selected_url)
        if (parsed.scheme not in {"https", "http"} or not parsed.netloc or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("release URL must be an HTTP(S) base URL without credentials or query")
        if len(parts) == 1:
            _download(f"{selected_url}/{parts[0]['name']}", destination, identity)
            print(f"reconstructed and verified: {destination}", flush=True)
            return destination

    local_parts = Path(parts_dir).expanduser().resolve(strict=True) if parts_dir is not None else None
    cache = destination.parent / ".release_parts"
    if local_parts is None:
        if cache.is_symlink():
            raise ValueError("release part cache must not be a symbolic link")
        cache.mkdir(exist_ok=True)
    verified_parts = []
    for part in parts:
        part_path = (local_parts or cache) / part["name"]
        part_identity = _identity(part, "part")
        if local_parts is None and not part_path.exists() and not part_path.is_symlink():
            _download(f"{selected_url}/{part['name']}", part_path, part_identity)
        _verify(part_path, part_identity, "part")
        verified_parts.append(part_path)
        print(f"verified {part['name']}", flush=True)

    temporary = _temporary(destination.parent, destination.name)
    try:
        with temporary.open("wb") as stream:
            for part_path in verified_parts:
                with part_path.open("rb") as source:
                    for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                        stream.write(chunk)
        _verify(temporary, identity, "reconstructed checkpoint")
        _publish(temporary, destination, identity)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"reconstructed and verified: {destination}", flush=True)
    return destination
