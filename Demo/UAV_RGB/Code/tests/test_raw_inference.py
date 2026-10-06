import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo512 import model as model_module
from demo512 import CLASSES
from demo512.raw_inference import EXPECTED_READOUT, RawInferenceEngine


@pytest.fixture
def recorded_model(tmp_path, monkeypatch):
    checkpoint = tmp_path / "demo.pt"
    checkpoint.write_bytes(b"mock native512 checkpoint")
    record = tmp_path / "model_record.json"
    record.write_text(json.dumps({
        "status": "requested_training_complete", "input_size": [512, 512],
        "input_resized": False, "readout": EXPECTED_READOUT,
        # This referenced training manifest deliberately does not exist.
        "dataset_manifest_sha256": "0" * 64,
        "checkpoint": {"file": checkpoint.name, "bytes": checkpoint.stat().st_size,
                       "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
    }), encoding="utf-8")
    builds, calls = [], []

    def build(path, device, **kwargs):
        builds.append((path, device, kwargs))
        assert path == checkpoint
        assert device == "cpu"
        assert kwargs == {"checkpoint_kind": "demo", "expected_sha256":
                          hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        return torch.nn.Linear(1, 1)

    def forward(model, tensor, label):
        assert not model.training
        assert not torch.is_grad_enabled()
        assert not any(parameter.requires_grad for parameter in model.parameters())
        assert tensor.shape == (1, 3, 512, 512)
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.float32
        calls.append(label)
        # Each RGB channel is an independent, exactly verifiable class mask.
        channel = CLASSES.index(label)
        return tensor[:, channel:channel + 1] * 255 - 127.5

    monkeypatch.setattr(model_module, "build_demo_model", build)
    monkeypatch.setattr(model_module, "forward_class_logits", forward)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return record, builds, calls


def test_all_tiles_keep_native_pixels_stitch_original_grid_and_overlap(recorded_model):
    record, builds, calls = recorded_model
    engine = RawInferenceEngine(record, device="cpu")
    rng = np.random.default_rng(12)
    source = rng.integers(0, 256, size=(513, 1025, 3), dtype=np.uint8)
    source_before = source.copy()
    seen = []

    def observe(meta, rgb512, masks):
        seen.append(meta)
        x0, y0, w, h = (meta[key] for key in ("x0", "y0", "valid_width", "valid_height"))
        assert rgb512.shape == (512, 512, 3)
        np.testing.assert_array_equal(rgb512[:h, :w], source[y0:y0 + h, x0:x0 + w])
        assert not rgb512[h:, :].any()
        assert not rgb512[:, w:].any()
        for label in CLASSES:
            assert masks[label].dtype == np.bool_
            assert masks[label].shape == (512, 512)
            assert not masks[label][h:, :].any()
            assert not masks[label][:, w:].any()

    result = engine.predict(source, on_tile=observe)
    assert len(builds) == 1
    assert calls == list(CLASSES) * 6
    assert [(tile["x0"], tile["y0"]) for tile in result["tiles"]] == [
        (0, 0), (512, 0), (1024, 0), (0, 512), (512, 512), (1024, 512)]
    assert result["tiles"][-1] == {"id_suffix": "x01024_y00512", "x0": 1024,
                                  "y0": 512, "valid_width": 1, "valid_height": 1}
    assert seen == result["tiles"]
    for class_id in (1, 2, 3):
        mask = result["class_masks"][class_id]
        assert mask.dtype == np.bool_
        assert mask.shape == source.shape[:2]
        np.testing.assert_array_equal(mask, source[:, :, class_id - 1] >= 128)
    assert np.any(result["class_masks"][1] & result["class_masks"][2])
    np.testing.assert_array_equal(source, source_before)
    assert result["elapsed_seconds"] >= 0
    assert engine.metadata["device"] == "cpu"
    assert engine.metadata["dtype"] == "float32"
    assert engine.metadata["autocast_dtype"] is None
    assert engine.metadata["ground_truth_used_for_prediction"] is False
    assert engine.metadata["ground_truth_used_for_tile_selection"] is False
    assert engine.metadata["dataset_manifest_read"] is False
    assert engine.metadata["input_size"] == [512, 512]
    assert engine.metadata["source_resized"] is False


def test_no_labels_or_dataset_reads_and_single_load_reused(recorded_model, monkeypatch):
    record, builds, calls = recorded_model
    opened = []
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        # The mocked verified loader reads only its checkpoint; neither a GT
        # image nor a dataset.json is available or allowed to be accessed.
        assert path.name in {record.name, "demo.pt"}
        opened.append(path.name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    engine = RawInferenceEngine(record)
    first = engine.predict(np.zeros((1, 1, 3), dtype=np.uint8))
    second = engine.predict(np.full((2, 3, 3), 255, dtype=np.uint8))
    assert len(builds) == 1
    assert calls == list(CLASSES) * 2
    assert len(first["tiles"]) == len(second["tiles"]) == 1
    assert all(not mask.any() for mask in first["class_masks"].values())
    assert all(mask.all() for mask in second["class_masks"].values())
    assert set(opened) == {record.name, "demo.pt"}


def test_padding_mask_and_callback_cannot_change_stitched_prediction(recorded_model, monkeypatch):
    record, _, _ = recorded_model

    def predict_every_pixel(model, tensor, label):
        return torch.full((1, 1, 512, 512), 10.0)

    monkeypatch.setattr(model_module, "forward_class_logits", predict_every_pixel)
    engine = RawInferenceEngine(record, device="cpu")
    source = np.full((2, 3, 3), 180, dtype=np.uint8)

    def mutate(meta, rgb, masks):
        for mask in masks.values():
            assert mask.sum() == 6
            assert not mask[2:, :].any()
            assert not mask[:, 3:].any()
            mask[:] = False
        rgb[:] = 0
        meta["x0"] = 999

    result = engine.predict(source, on_tile=mutate)
    assert all(mask.all() for mask in result["class_masks"].values())
    assert result["tiles"][0]["x0"] == 0
    assert (source == 180).all()


@pytest.mark.parametrize("invalid", [
    np.zeros((0, 3, 3), dtype=np.uint8), np.zeros((3, 0, 3), dtype=np.uint8),
    np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2, 4), dtype=np.uint8),
    np.zeros((2, 2, 3), dtype=np.float32), [[[0, 0, 0]]],
])
def test_reject_invalid_source_before_prediction(recorded_model, invalid):
    record, _, calls = recorded_model
    engine = RawInferenceEngine(record, device="cpu")
    with pytest.raises(ValueError, match="HxWx3 uint8 RGB"):
        engine.predict(invalid)
    assert not calls


@pytest.mark.parametrize("threshold", [0, 1, -0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_reject_invalid_threshold_before_loading(recorded_model, threshold):
    record, builds, _ = recorded_model
    with pytest.raises(ValueError, match="threshold"):
        RawInferenceEngine(record, threshold=threshold)
    assert not builds


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "invalid"])
def test_reject_unavailable_or_invalid_device_before_loading(recorded_model, device):
    record, builds, _ = recorded_model
    with pytest.raises(ValueError, match="device|CUDA"):
        RawInferenceEngine(record, device=device)
    assert not builds


@pytest.mark.parametrize("changed", [
    {"input_size": [1008, 1008]}, {"input_resized": True}, {"readout": "instance_union"},
    {"status": "failed"},
])
def test_reject_non_demo_or_incomplete_model_record(recorded_model, changed):
    record, builds, _ = recorded_model
    metadata = json.loads(record.read_text())
    metadata.update(changed)
    record.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="completed native-512"):
        RawInferenceEngine(record)
    assert not builds


def test_checkpoint_size_checked_before_deserialization(recorded_model):
    record, builds, _ = recorded_model
    metadata = json.loads(record.read_text())
    metadata["checkpoint"]["bytes"] += 1
    record.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="byte size"):
        RawInferenceEngine(record)
    assert not builds


def test_checkpoint_path_cannot_escape_record_folder(recorded_model, tmp_path):
    record, builds, _ = recorded_model
    nested = tmp_path / "nested"
    nested.mkdir()
    metadata = json.loads(record.read_text())
    metadata["checkpoint"]["file"] = "../demo.pt"
    copied_record = nested / "record.json"
    copied_record.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="inside"):
        RawInferenceEngine(copied_record)
    assert not builds


def test_noncontiguous_rgb_input_is_not_resized(recorded_model):
    record, _, _ = recorded_model
    engine = RawInferenceEngine(record, device="cpu")
    source = np.arange(120, dtype=np.uint8).reshape(4, 10, 3)[:, ::2, :]
    assert not source.flags.c_contiguous
    result = engine.predict(source)
    for class_id in (1, 2, 3):
        np.testing.assert_array_equal(result["class_masks"][class_id],
                                      source[:, :, class_id - 1] >= 128)
