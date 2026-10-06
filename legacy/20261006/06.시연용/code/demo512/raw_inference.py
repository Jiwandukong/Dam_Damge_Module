"""Run the native-512 demo model on every tile of an unlabelled RGB photo.

Only the model record and its checkpoint are opened. No label, training dataset,
or dataset manifest is consulted. Original pixels are copied into 512px tiles;
right/bottom padding is black and is excluded from the stitched predictions.
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch

from . import CLASSES, INPUT_SIZE


TileCallback = Callable[[dict, np.ndarray, dict[str, np.ndarray]], None]
EXPECTED_READOUT = "sam3_existing_text_conditioned_semantic_head"


def _predict_masks(model, rgb, valid, threshold, device, forward, use_bfloat16) -> dict:
    """No GT, tile metadata, annotations, or dataset paths enter this function."""
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if use_bfloat16 else nullcontext()
    masks = {}
    with torch.no_grad(), context:
        for label in CLASSES:
            logits = forward(model, tensor, label)
            if tuple(logits.shape) != (1, 1, INPUT_SIZE, INPUT_SIZE):
                raise ValueError("model must return native [1,1,512,512] semantic logits")
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("non-finite model logits")
            masks[label] = (logits.float().sigmoid()[0, 0].cpu().numpy() >= threshold) & valid
    return masks


class RawInferenceEngine:
    """Load one verified demo checkpoint and reuse it for unlabelled photos.

    ``predict`` accepts an HxWx3 uint8 RGB array of any positive image size.
    It returns independent boolean masks indexed by 1=CRC, 2=DLM, and 3=SPL,
    in original-photo coordinates. An optional callback receives each tile's
    metadata, its untouched/padded RGB pixels, and independent class-name masks.
    """

    def __init__(self, model_record: str | Path, device: str = "auto",
                 threshold: float = 0.5):
        from .model import build_demo_model, forward_class_logits

        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a finite number strictly between zero and one")
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("threshold must be a finite number strictly between zero and one")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")

        record_path = Path(model_record).expanduser().resolve(strict=True)
        record_bytes = record_path.read_bytes()
        training = json.loads(record_bytes)
        if (training.get("status") != "requested_training_complete"
                or training.get("input_size") != [INPUT_SIZE, INPUT_SIZE]
                or training.get("input_resized") is not False
                or training.get("readout") != EXPECTED_READOUT):
            raise ValueError("model record must describe completed native-512 semantic training")
        saved = training.get("checkpoint")
        if not isinstance(saved, dict):
            raise ValueError("model record has no checkpoint identity")
        name = saved.get("file")
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise ValueError("checkpoint file must be relative to its model record")
        checkpoint = (record_path.parent / name).resolve(strict=True)
        if not checkpoint.is_file() or not checkpoint.is_relative_to(record_path.parent):
            raise ValueError("checkpoint must remain inside its model record folder")
        expected_size, expected_sha256 = saved.get("bytes"), saved.get("sha256")
        if (isinstance(expected_size, bool) or not isinstance(expected_size, int)
                or expected_size <= 0 or checkpoint.stat().st_size != expected_size):
            raise ValueError("checkpoint byte size does not match the model record")
        if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
                or any(character not in "0123456789abcdef" for character in expected_sha256)):
            raise ValueError("model record must contain a lowercase SHA-256 checkpoint digest")

        self.device = device
        self.threshold = float(threshold)
        self.use_bfloat16 = device == "cuda" and torch.cuda.is_bf16_supported()
        started = time.monotonic()
        # build_demo_model checks SHA-256 before deserialization and strictly
        # loads every state entry. The original 10 GB base is not needed here.
        self.model = build_demo_model(checkpoint, device, checkpoint_kind="demo",
                                      expected_sha256=expected_sha256)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._forward = forward_class_logits
        self.metadata = {
            "prediction_source": "actual_model_semantic_logits",
            "ground_truth_used_for_prediction": False,
            "ground_truth_used_for_tile_selection": False,
            "dataset_manifest_read": False,
            "model_record_name": record_path.name,
            "model_record_sha256": hashlib.sha256(record_bytes).hexdigest(),
            "checkpoint_name": checkpoint.name,
            "checkpoint_sha256": expected_sha256,
            "checkpoint_size_bytes": expected_size,
            "checkpoint_loaded_strictly": True,
            "training_status": training["status"],
            "readout": training["readout"],
            "input_size": [INPUT_SIZE, INPUT_SIZE],
            "tile_size": INPUT_SIZE,
            "input_resized": False,
            "source_resized": False,
            "native512": True,
            "tile_selection": "all_source_tiles_without_label_selection",
            "padding_policy": "black_right_bottom_then_crop_to_original_size",
            "padding_excluded_from_masks": True,
            "queried_classes_per_tile": list(CLASSES),
            "class_ids": {label: index for index, label in enumerate(CLASSES, 1)},
            "independent_class_masks": True,
            "threshold": self.threshold,
            "threshold_rule": "sigmoid(logits) >= threshold",
            "device": device,
            "dtype": "bfloat16" if self.use_bfloat16 else "float32",
            "autocast_dtype": "bfloat16" if self.use_bfloat16 else None,
            "model_load_seconds": time.monotonic() - started,
        }

    def predict(self, rgb: np.ndarray, *, on_tile: TileCallback | None = None) -> dict:
        """Predict all classes on every tile; never consult GT or select positives.

        ``on_tile(tile_metadata, rgb512, masks_by_class_name)`` runs after the
        original-coordinate masks have been copied, so callback edits cannot
        change the stitched result. Exceptions are propagated to the caller.
        """
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
                or rgb.ndim != 3 or rgb.shape[2] != 3
                or rgb.shape[0] < 1 or rgb.shape[1] < 1):
            raise ValueError("source image must be a nonempty HxWx3 uint8 RGB array")
        if on_tile is not None and not callable(on_tile):
            raise ValueError("on_tile must be callable or None")
        height, width = rgb.shape[:2]
        masks_full = {index: np.zeros((height, width), dtype=bool)
                      for index in range(1, len(CLASSES) + 1)}
        tiles = []
        started = time.monotonic()
        for y0 in range(0, height, INPUT_SIZE):
            for x0 in range(0, width, INPUT_SIZE):
                valid_width = min(INPUT_SIZE, width - x0)
                valid_height = min(INPUT_SIZE, height - y0)
                tile = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
                tile[:valid_height, :valid_width] = rgb[y0:y0 + valid_height,
                                                      x0:x0 + valid_width]
                valid = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=bool)
                valid[:valid_height, :valid_width] = True
                masks = _predict_masks(self.model, tile, valid, self.threshold,
                                       self.device, self._forward, self.use_bfloat16)
                for index, label in enumerate(CLASSES, 1):
                    masks_full[index][y0:y0 + valid_height, x0:x0 + valid_width] = (
                        masks[label][:valid_height, :valid_width])
                tile_metadata = {
                    "id_suffix": f"x{x0:05d}_y{y0:05d}",
                    "x0": x0, "y0": y0,
                    "valid_width": valid_width, "valid_height": valid_height,
                }
                tiles.append(tile_metadata)
                if on_tile is not None:
                    on_tile(dict(tile_metadata), tile, masks)
        return {"class_masks": masks_full, "tiles": tiles,
                "elapsed_seconds": time.monotonic() - started}
