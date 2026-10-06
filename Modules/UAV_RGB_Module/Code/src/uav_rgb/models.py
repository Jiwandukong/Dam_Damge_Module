"""Checkpoint inspection and strict SAM-3 model loading."""

from __future__ import annotations

import gc
import hashlib
import importlib
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

from .config import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CHECKPOINT_SIZE,
    EXPECTED_MODEL_KEYS,
    SAM3_UPSTREAM_COMMIT,
)


class CheckpointIdentityError(ValueError):
    """Raised before deserialization when checkpoint identity verification fails."""

    def __init__(self, message: str, metadata: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.metadata = dict(metadata)


def sha256_file(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Return a file's SHA-256 digest without changing the file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_checkpoint_file(
    checkpoint_path: str | Path,
    *,
    expected_size: int | None = EXPECTED_CHECKPOINT_SIZE,
    expected_sha256: str | None = EXPECTED_CHECKPOINT_SHA256,
    compute_sha256: bool = True,
    require_sha256: bool = True,
) -> dict[str, Any]:
    """Inspect checkpoint bytes without invoking ``torch.load``.

    Size is checked first.  A size mismatch is already conclusive, so hashing
    is skipped in that case.  This is intentionally the only operation used
    before deciding whether a pickle-based PyTorch checkpoint may be loaded.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    stat = path.stat()
    size_matches = (
        stat.st_size == expected_size if expected_size is not None else None
    )
    digest = None
    sha256_skipped_reason = None
    if compute_sha256 and size_matches is not False:
        digest = sha256_file(path)
    elif compute_sha256 and size_matches is False:
        sha256_skipped_reason = "checkpoint size mismatch"
    elif require_sha256 and expected_sha256 is not None:
        sha256_skipped_reason = "SHA-256 computation disabled"

    sha256_match = (
        digest == expected_sha256
        if digest is not None and expected_sha256 is not None
        else None
    )
    size_requirement_passed = size_matches is not False
    exact_identity_verified = size_requirement_passed and (
        expected_sha256 is None or sha256_match is True
    )
    load_preflight_passed = size_requirement_passed and (
        not require_sha256
        or expected_sha256 is None
        or sha256_match is True
    )

    return {
        "path": str(path),
        "size": stat.st_size,
        "expected_size": expected_size,
        "size_matches_verified_model": size_matches,
        "sha256": digest,
        "expected_sha256": expected_sha256,
        "sha256_match": sha256_match,
        "sha256_verification_required": require_sha256,
        "sha256_skipped_reason": sha256_skipped_reason,
        "checkpoint_identity_verified": bool(exact_identity_verified),
        "checkpoint_load_preflight_passed": bool(load_preflight_passed),
        # Used to reject a file that changes after hashing but before torch.load.
        "_stat_mtime_ns": stat.st_mtime_ns,
        "_stat_inode": stat.st_ino,
    }


def _identity_error(metadata: Mapping[str, Any]) -> str:
    if metadata.get("size_matches_verified_model") is False:
        return (
            "checkpoint size mismatch: "
            f"actual={metadata.get('size')}, expected={metadata.get('expected_size')}"
        )
    if metadata.get("sha256_match") is False:
        return (
            "checkpoint SHA-256 mismatch: "
            f"actual={metadata.get('sha256')}, "
            f"expected={metadata.get('expected_sha256')}"
        )
    return "checkpoint identity could not be verified before deserialization"


def _assert_checkpoint_unchanged(
    path: Path, metadata: Mapping[str, Any]
) -> None:
    current = path.stat()
    if (
        current.st_size != metadata.get("size")
        or current.st_mtime_ns != metadata.get("_stat_mtime_ns")
        or current.st_ino != metadata.get("_stat_inode")
    ):
        changed = dict(metadata)
        changed["checkpoint_identity_verified"] = False
        changed["checkpoint_load_preflight_passed"] = False
        raise CheckpointIdentityError(
            "checkpoint changed after identity verification and was not loaded",
            changed,
        )


def _git_output(checkout: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None)
        suffix = f": {detail.strip()}" if isinstance(detail, str) and detail else ""
        raise RuntimeError(
            f"cannot verify the imported SAM-3 Git checkout at {checkout}{suffix}"
        ) from exc
    return completed.stdout.strip()


def verify_sam3_source(
    sam3_module: Any | None = None,
    *,
    expected_commit: str = SAM3_UPSTREAM_COMMIT,
    imported_modules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify that the actually imported ``sam3`` package is from pinned Git HEAD."""
    module = sam3_module or importlib.import_module("sam3")
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(
            "cannot verify imported SAM-3 source: sam3.__file__ is unavailable"
        )

    source_path = Path(module_file).resolve()
    checkout_root = Path(
        _git_output(source_path.parent, "rev-parse", "--show-toplevel")
    ).resolve()
    try:
        source_path.relative_to(checkout_root)
    except ValueError as exc:
        raise RuntimeError(
            f"imported SAM-3 source {source_path} is outside Git checkout "
            f"{checkout_root}"
        ) from exc

    imported_module_paths: dict[str, str] = {}
    for name, imported_module in (imported_modules or {}).items():
        imported_file = getattr(imported_module, "__file__", None)
        if not imported_file:
            raise RuntimeError(
                f"cannot verify imported SAM-3 module {name}: __file__ is unavailable"
            )
        imported_path = Path(imported_file).resolve()
        try:
            imported_path.relative_to(checkout_root)
        except ValueError as exc:
            raise RuntimeError(
                f"imported SAM-3 module {name} at {imported_path} is outside "
                f"Git checkout {checkout_root}"
            ) from exc
        imported_module_paths[name] = str(imported_path)

    actual_commit = _git_output(checkout_root, "rev-parse", "HEAD").lower()
    expected = expected_commit.lower()
    if actual_commit != expected:
        raise RuntimeError(
            "SAM-3 upstream commit mismatch before model loading: "
            f"actual={actual_commit}, expected={expected}, source={source_path}"
        )
    dirty_entries = _git_output(
        checkout_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).splitlines()
    source_clean = not dirty_entries
    if not source_clean:
        raise RuntimeError(
            "the imported SAM-3 checkout has tracked or untracked changes; "
            "AUG512 reproducibility requires a clean pinned checkout"
        )
    return {
        "upstream_source_path": str(checkout_root),
        "upstream_package_path": str(source_path),
        "upstream_imported_module_paths": imported_module_paths,
        "upstream_repository_root": str(checkout_root),
        "actual_upstream_commit": actual_commit,
        "expected_upstream_commit": expected,
        "upstream_commit_matches": True,
        "upstream_source_clean": source_clean,
        "upstream_source_dirty_entry_count": len(dirty_entries),
        "sam3_source_verified": True,
        "upstream_source_provenance_verified": source_clean,
    }


def resolve_device(device: str) -> str:
    """Resolve ``auto`` and validate the requested torch device."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return device


def _load_checkpoint_mapping(path: Path) -> Mapping[str, Any]:
    # PyTorch checkpoints use pickle. Load only checkpoints from trusted sources.
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint root must be a mapping")
    return checkpoint


def inspect_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_size: int | None = EXPECTED_CHECKPOINT_SIZE,
    expected_sha256: str | None = EXPECTED_CHECKPOINT_SHA256,
    compute_sha256: bool = True,
) -> dict[str, Any]:
    """Inspect a checkpoint, deserializing only after byte identity passes."""
    if expected_sha256 is None or not compute_sha256:
        raise ValueError(
            "checkpoint deserialization requires an expected SHA-256 and "
            "enabled SHA-256 verification"
        )
    file_metadata = inspect_checkpoint_file(
        checkpoint_path,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        compute_sha256=compute_sha256,
        require_sha256=expected_sha256 is not None,
    )
    result = {
        key: value
        for key, value in file_metadata.items()
        if not key.startswith("_stat_")
    }
    result.update(
        {
            "checkpoint_container_inspected": False,
            "checkpoint_model_key_present": False,
            "model_state_key_count": 0,
            "checkpoint_epoch": None,
            "expected_upstream_commit": SAM3_UPSTREAM_COMMIT,
            "sam3_source_verified": False,
        }
    )
    if not file_metadata["checkpoint_load_preflight_passed"]:
        result["inspection_skipped_reason"] = _identity_error(file_metadata)
        return result

    path = Path(file_metadata["path"])
    _assert_checkpoint_unchanged(path, file_metadata)
    checkpoint = _load_checkpoint_mapping(path)
    model_state = checkpoint.get("model")
    model_key_present = isinstance(model_state, Mapping) and bool(model_state)
    result.update(
        {
            "checkpoint_container_inspected": True,
            "checkpoint_model_key_present": model_key_present,
            "model_state_key_count": len(model_state) if model_key_present else 0,
            "checkpoint_epoch": checkpoint.get("epoch"),
        }
    )
    del model_state, checkpoint
    gc.collect()
    return result


def load_model(
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
    expected_size: int | None = EXPECTED_CHECKPOINT_SIZE,
    expected_sha256: str | None = EXPECTED_CHECKPOINT_SHA256,
    verify_sha256: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build pinned upstream SAM-3 and strictly load ``checkpoint['model']``.

    The handoff checkpoint is accepted only when its byte size and, by
    default, SHA-256 match the verified artifact.  ``strict=True`` is used
    after an explicit key/shape audit.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha256 is None or not verify_sha256:
        raise ValueError(
            "checkpoint deserialization requires an expected SHA-256 and "
            "enabled SHA-256 verification"
        )

    file_metadata = inspect_checkpoint_file(
        path,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        compute_sha256=verify_sha256,
        require_sha256=expected_sha256 is not None,
    )
    if not file_metadata["checkpoint_load_preflight_passed"]:
        raise CheckpointIdentityError(
            _identity_error(file_metadata), file_metadata
        )

    resolved_device = resolve_device(device)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "the verified upstream SAM-3 commit requires CUDA during model "
            "construction, even when the requested final device is CPU"
        )

    try:
        import sam3
        from sam3.model_builder import build_sam3_image_model
    except ImportError as exc:
        raise ImportError(
            "SAM-3 is not installed. Install facebookresearch/sam3 at commit "
            f"{SAM3_UPSTREAM_COMMIT} before loading the checkpoint."
        ) from exc

    source_metadata = verify_sam3_source(
        sam3,
        imported_modules={
            "model_builder": importlib.import_module("sam3.model_builder")
        },
    )

    model = build_sam3_image_model(
        device="cpu",
        eval_mode=True,
        checkpoint_path=None,
        load_from_HF=False,
        enable_segmentation=True,
        compile=False,
    )
    _assert_checkpoint_unchanged(path, file_metadata)
    checkpoint = _load_checkpoint_mapping(path)
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("checkpoint does not contain a non-empty 'model' mapping")
    if len(state_dict) != EXPECTED_MODEL_KEYS:
        raise RuntimeError(
            "checkpoint model key count mismatch: "
            f"actual={len(state_dict)}, expected={EXPECTED_MODEL_KEYS}"
        )

    destination = model.state_dict()
    source_keys = set(state_dict)
    destination_keys = set(destination)
    missing_keys = sorted(destination_keys - source_keys)
    unexpected_keys = sorted(source_keys - destination_keys)
    shape_mismatches = [
        {
            "key": key,
            "source": list(state_dict[key].shape),
            "destination": list(destination[key].shape),
        }
        for key in sorted(source_keys & destination_keys)
        if tuple(state_dict[key].shape) != tuple(destination[key].shape)
    ]
    dtype_mismatches = [
        {
            "key": key,
            "source": str(state_dict[key].dtype),
            "destination": str(destination[key].dtype),
        }
        for key in sorted(source_keys & destination_keys)
        if state_dict[key].dtype != destination[key].dtype
    ]
    if missing_keys or unexpected_keys or shape_mismatches or dtype_mismatches:
        raise RuntimeError(
            "checkpoint/model structure mismatch: "
            f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}, "
            f"shape={len(shape_mismatches)}, dtype={len(dtype_mismatches)}"
        )

    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict load returned incompatibilities: {incompatible}")

    metadata = {
        "path": str(path),
        "filename": path.name,
        "size": file_metadata["size"],
        "expected_size": expected_size,
        "size_matches_verified_model": file_metadata[
            "size_matches_verified_model"
        ],
        "sha256": file_metadata["sha256"],
        "expected_sha256": expected_sha256,
        "sha256_match": file_metadata["sha256_match"],
        "sha256_verification_required": verify_sha256,
        "checkpoint_identity_verified": file_metadata[
            "checkpoint_identity_verified"
        ],
        "checkpoint_load_preflight_passed": file_metadata[
            "checkpoint_load_preflight_passed"
        ],
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model_state_key_count": len(state_dict),
        "strict_load_passed": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
        "dtype_mismatches": [],
        "experiment_id": checkpoint.get("experiment_id"),
        "stage": checkpoint.get("stage"),
        "successful_optimizer_steps": checkpoint.get("successful_optimizer_steps"),
        "device": resolved_device,
        # Backward-compatible key now records the observed, not assumed, HEAD.
        "upstream_commit": source_metadata["actual_upstream_commit"],
        **source_metadata,
    }
    del destination, state_dict, checkpoint
    gc.collect()
    model.eval()
    model.to(resolved_device)
    return model, metadata
