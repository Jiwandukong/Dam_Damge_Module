from __future__ import annotations

import pytest

from uav_rgb.models import load_model


def test_pickle_checkpoint_cannot_bypass_sha256(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not deserialized")
    with pytest.raises(ValueError, match="requires an expected SHA-256"):
        load_model(
            checkpoint,
            expected_size=checkpoint.stat().st_size,
            expected_sha256="0" * 64,
            verify_sha256=False,
        )


@pytest.mark.parametrize(
    ("expected_sha256", "verify_sha256"),
    [(None, True), (None, False)],
)
def test_pickle_checkpoint_requires_expected_digest(
    tmp_path,
    expected_sha256,
    verify_sha256,
):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not deserialized")
    with pytest.raises(ValueError, match="requires an expected SHA-256"):
        load_model(
            checkpoint,
            expected_size=checkpoint.stat().st_size,
            expected_sha256=expected_sha256,
            verify_sha256=verify_sha256,
        )


def test_checkpoint_inspection_cannot_deserialize_without_digest(tmp_path):
    from uav_rgb.models import inspect_checkpoint

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not deserialized")
    with pytest.raises(ValueError, match="requires an expected SHA-256"):
        inspect_checkpoint(
            checkpoint,
            expected_size=checkpoint.stat().st_size,
            expected_sha256=None,
            compute_sha256=False,
        )
