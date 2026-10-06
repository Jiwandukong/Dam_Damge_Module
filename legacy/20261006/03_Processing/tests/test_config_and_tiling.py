from __future__ import annotations

from pathlib import Path
from types import MethodType

import numpy as np
import pytest
from PIL import Image

from uav_rgb.config import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CHECKPOINT_SIZE,
    MODEL_VARIANT,
    SAM3_UPSTREAM_COMMIT,
    load_config,
)
from uav_rgb.inference import InferenceEngine, make_positions, pad_image_to_tile_grid


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_aug512_configuration_is_exact():
    config = load_config(
        PROJECT_ROOT / "03_Processing/configs/daechung_aug512.yaml"
    )
    assert config.tile_size == 512
    assert config.stride == 512
    assert config.edge_policy == "pad"
    assert config.padding_mode == "reflect"
    assert config.expected_checkpoint_size == 10_081_318_934
    assert config.expected_checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256
    assert config.palette[2] == (255, 210, 0)
    assert MODEL_VARIANT == "daechung_sam3_aug512_query_balanced_v1"


def test_sample_grid_has_88_non_overlapping_padded_tiles():
    assert make_positions(5632, 512, 512) == list(range(0, 5632, 512))
    assert make_positions(4096, 512, 512) == list(range(0, 4096, 512))
    assert len(make_positions(5632, 512, 512)) * len(
        make_positions(4096, 512, 512)
    ) == 88


def test_padding_changes_no_source_pixel_and_records_exact_sample_shape():
    source = np.arange(3 * 7 * 5, dtype=np.uint8).reshape(5, 7, 3)
    padded, metadata = pad_image_to_tile_grid(
        Image.fromarray(source), tile_size=4, padding_mode="reflect"
    )
    padded_array = np.asarray(padded)
    assert padded.size == (8, 8)
    assert np.array_equal(padded_array[:5, :7], source)
    assert metadata["source_resized"] is False
    assert metadata["pad_right_px"] == 1
    assert metadata["pad_bottom_px"] == 3

    _, sample_metadata = pad_image_to_tile_grid(
        Image.new("RGB", (5280, 3956)),
        tile_size=512,
        padding_mode="reflect",
    )
    assert sample_metadata["padded_shape_hw"] == [4096, 5632]
    assert sample_metadata["pad_right_px"] == 352
    assert sample_metadata["pad_bottom_px"] == 140


def _fake_engine(edge_policy: str = "pad") -> InferenceEngine:
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.confidence_threshold = 0.5
    engine.mask_threshold = 0.5
    engine.candidate_collection_floor = 0.05
    engine.inference_mode = "tiled"
    engine.tile_size = 512
    engine.stride = 512
    engine.processor_resolution = 1008
    engine.edge_policy = edge_policy
    engine.padding_mode = "reflect"
    engine.device = "cpu"
    engine.class_prompts = {
        "CRC": "crack",
        "DLM": "delamination",
        "SPL": "spalling",
    }
    engine.checkpoint = {
        "filename": "checkpoint_final.pt",
        "size": EXPECTED_CHECKPOINT_SIZE,
        "expected_size": EXPECTED_CHECKPOINT_SIZE,
        "size_matches_verified_model": True,
        "sha256": EXPECTED_CHECKPOINT_SHA256,
        "expected_sha256": EXPECTED_CHECKPOINT_SHA256,
        "sha256_match": True,
        "checkpoint_identity_verified": True,
        "checkpoint_load_preflight_passed": True,
        "strict_load_passed": True,
        "checkpoint_epoch": 1,
        "model_state_key_count": 1134,
        "upstream_commit": SAM3_UPSTREAM_COMMIT,
        "expected_upstream_commit": SAM3_UPSTREAM_COMMIT,
        "actual_upstream_commit": SAM3_UPSTREAM_COMMIT,
        "upstream_commit_matches": True,
        "upstream_source_path": "/tmp/sam3",
        "upstream_package_path": "/tmp/sam3/sam3/__init__.py",
        "upstream_imported_module_paths": {},
        "upstream_repository_root": "/tmp/sam3",
        "upstream_source_clean": True,
        "upstream_source_dirty_entry_count": 0,
        "sam3_source_verified": True,
        "upstream_source_provenance_verified": True,
    }

    def fake_predict(self, image):
        masks = {
            name: np.zeros((image.height, image.width), dtype=bool)
            for name in self.class_prompts
        }
        for mask in masks.values():
            mask[0, 0] = True
            mask[-1, -1] = True
        classes = {
            name: {
                "prompt": prompt,
                "candidate_count": 1,
                "accepted_candidate_count": 1,
                "candidate_scores": [0.5],
                "accepted_candidate_scores": [0.5],
                "top_score": 0.5,
                "mask_area_pixels": 2,
            }
            for name, prompt in self.class_prompts.items()
        }
        return masks, classes

    engine._predict_queries = MethodType(fake_predict, engine)
    return engine


def test_tiled_padding_is_cropped_back_to_original_grid():
    prediction = _fake_engine().predict(
        Image.new("RGB", (520, 510)), image_name="source.jpg"
    )
    assert all(mask.shape == (510, 520) for mask in prediction["masks"].values())
    assert prediction["inference"]["tile_count"] == 2
    edge = prediction["inference"]["edge_handling"]
    assert edge["source_shape_hw"] == [510, 520]
    assert edge["padded_shape_hw"] == [512, 1024]
    assert edge["source_resized"] is False
    assert edge["output_cropped_to_source_shape"] is True
    assert prediction["inference"]["official_handoff_protocol"] is False


def test_edge_error_rejects_non_divisible_source():
    with pytest.raises(ValueError, match="divisible by 512"):
        _fake_engine(edge_policy="error").predict(
            Image.new("RGB", (520, 510)), image_name="source.jpg"
        )


def test_predict_rejects_mutated_non_aug512_engine_state():
    engine = _fake_engine()
    engine.tile_size = 256
    engine.stride = 256
    with pytest.raises(RuntimeError, match="must remain 512"):
        engine.predict(Image.new("RGB", (512, 512)), image_name="source.jpg")


def test_whole_mode_rejects_non_512_source():
    with pytest.raises(ValueError, match="one 512x512 tile"):
        _fake_engine().predict(
            Image.new("RGB", (1024, 512)),
            image_name="source.jpg",
            inference_mode="whole",
        )
