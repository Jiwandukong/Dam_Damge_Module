from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


SAM3_UPSTREAM_COMMIT = "46957e47805eaa273f4aa7bbbd25a88bca9108ce"
EXPECTED_CHECKPOINT_SHA256 = (
    "a2749dba62207575afac9ed42f923d7cdfd7e2d2f0ffb2210a103e91657d985d"
)
EXPECTED_CHECKPOINT_SIZE = 10_081_318_934
EXPECTED_MODEL_KEYS = 1_134
SAM3_PROCESSOR_RESOLUTION = 1008
SAM3_TRAINING_TILE_SIZE = 512
MODEL_VARIANT = "daechung_sam3_aug512_query_balanced_v1"
SAM3_CLASS_PROMPTS: dict[str, str] = {
    "CRC": "crack",
    "DLM": "delamination",
    "SPL": "spalling",
}
SAM3_CLASS_IDS: dict[str, int] = {"CRC": 1, "DLM": 2, "SPL": 3}


@dataclass(frozen=True)
class CrackSegConfig:
    """Validated SAM-3 inference settings.

    The class masks are independent binary masks.  Consequently, pixels can
    belong to more than one damage class and must not be collapsed with
    ``argmax`` before quantitative processing.
    """

    model_family: str
    upstream_commit: str
    expected_checkpoint_size: int
    expected_checkpoint_sha256: str
    candidate_collection_floor: float
    inference_mode: str
    tile_size: int
    stride: int
    edge_policy: str
    padding_mode: str
    processor_resolution: int
    confidence_threshold: float
    mask_threshold: float
    candidate_mask_operation: str
    top_k: int | None
    class_ids: dict[str, int]
    class_prompts: dict[str, str]
    palette: dict[int, tuple[int, int, int]]
    overlay_alpha: float

    @property
    def class_names(self) -> dict[int, str]:
        names = {0: "BG"}
        names.update({class_id: name for name, class_id in self.class_ids.items()})
        return names


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _threshold(value: Any, name: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def load_config(path: str | Path) -> CrackSegConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw = _mapping(raw, "configuration root")
    model = _mapping(raw.get("model"), "model")
    inference = _mapping(raw.get("inference"), "inference")
    class_entries = _mapping(inference.get("classes"), "inference.classes")
    visualization = _mapping(raw.get("visualization"), "visualization")
    colors = _mapping(visualization.get("colors_rgb"), "visualization.colors_rgb")

    class_ids: dict[str, int] = {}
    class_prompts: dict[str, str] = {}
    palette: dict[int, tuple[int, int, int]] = {0: (0, 0, 0)}
    for class_name, expected_prompt in SAM3_CLASS_PROMPTS.items():
        entry = _mapping(class_entries.get(class_name), f"inference.classes.{class_name}")
        class_id = int(entry.get("id", SAM3_CLASS_IDS[class_name]))
        prompt = str(entry.get("prompt", ""))
        if class_id != SAM3_CLASS_IDS[class_name]:
            raise ValueError(
                f"inference.classes.{class_name}.id must be "
                f"{SAM3_CLASS_IDS[class_name]}"
            )
        if prompt != expected_prompt:
            raise ValueError(
                f"inference.classes.{class_name}.prompt must be {expected_prompt!r}"
            )
        rgb = tuple(int(channel) for channel in colors.get(class_name, ()))
        if len(rgb) != 3 or any(channel < 0 or channel > 255 for channel in rgb):
            raise ValueError(
                f"visualization.colors_rgb.{class_name} must contain three "
                "integers in [0, 255]"
            )
        class_ids[class_name] = class_id
        class_prompts[class_name] = prompt
        palette[class_id] = rgb

    unknown_classes = set(class_entries) - set(SAM3_CLASS_PROMPTS)
    if unknown_classes:
        raise ValueError(f"unsupported SAM-3 classes: {sorted(unknown_classes)}")

    model_family = str(model.get("family", ""))
    if model_family != "sam3":
        raise ValueError("model.family must be 'sam3'")
    upstream_commit = str(model.get("upstream_commit", ""))
    if upstream_commit != SAM3_UPSTREAM_COMMIT:
        raise ValueError(
            f"model.upstream_commit must be the verified commit {SAM3_UPSTREAM_COMMIT}"
        )
    expected_sha256 = str(model.get("expected_checkpoint_sha256", "")).lower()
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("model.expected_checkpoint_sha256 must be a SHA-256 hex digest")
    if expected_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "model.expected_checkpoint_sha256 does not identify the verified "
            "AUG512 query-balanced checkpoint"
        )
    expected_size = _positive_int(
        model.get("expected_checkpoint_size"),
        "model.expected_checkpoint_size",
    )
    if expected_size != EXPECTED_CHECKPOINT_SIZE:
        raise ValueError(
            "model.expected_checkpoint_size does not identify the verified "
            "AUG512 query-balanced checkpoint"
        )

    candidate_floor = _threshold(
        model.get("candidate_collection_floor"),
        "model.candidate_collection_floor",
    )
    confidence_threshold = _threshold(
        inference.get("confidence_threshold"),
        "inference.confidence_threshold",
    )
    mask_threshold = _threshold(
        inference.get("mask_threshold"),
        "inference.mask_threshold",
    )
    if candidate_floor > confidence_threshold:
        raise ValueError(
            "model.candidate_collection_floor cannot exceed "
            "inference.confidence_threshold"
        )

    mode = str(inference.get("mode", ""))
    if mode not in {"whole", "tiled"}:
        raise ValueError("inference.mode must be one of: whole, tiled")
    tile_size = _positive_int(inference.get("tile_size"), "inference.tile_size")
    stride = _positive_int(inference.get("stride"), "inference.stride")
    if stride > tile_size:
        raise ValueError("inference.stride cannot exceed inference.tile_size")
    if tile_size != SAM3_TRAINING_TILE_SIZE or stride != SAM3_TRAINING_TILE_SIZE:
        raise ValueError(
            "AUG512 inference requires tile_size=512 and stride=512"
        )
    edge_policy = str(inference.get("edge_policy", "error"))
    if edge_policy not in {"error", "pad"}:
        raise ValueError("inference.edge_policy must be one of: error, pad")
    padding_mode = str(inference.get("padding_mode", "reflect"))
    if padding_mode not in {"reflect", "edge", "constant"}:
        raise ValueError(
            "inference.padding_mode must be one of: reflect, edge, constant"
        )
    operation = str(inference.get("candidate_mask_operation", ""))
    if operation != "binary_union":
        raise ValueError("inference.candidate_mask_operation must be 'binary_union'")
    if inference.get("top_k") is not None:
        raise ValueError("inference.top_k must be null for the locked inference contract")

    processor_resolution = _positive_int(
        inference.get("processor_resolution"),
        "inference.processor_resolution",
    )
    if processor_resolution != SAM3_PROCESSOR_RESOLUTION:
        raise ValueError(
            "inference.processor_resolution must be 1008 for the pinned SAM-3 "
            "inference contract"
        )

    return CrackSegConfig(
        model_family=model_family,
        upstream_commit=upstream_commit,
        expected_checkpoint_size=expected_size,
        expected_checkpoint_sha256=expected_sha256,
        candidate_collection_floor=candidate_floor,
        inference_mode=mode,
        tile_size=tile_size,
        stride=stride,
        edge_policy=edge_policy,
        padding_mode=padding_mode,
        processor_resolution=processor_resolution,
        confidence_threshold=confidence_threshold,
        mask_threshold=mask_threshold,
        candidate_mask_operation=operation,
        top_k=None,
        class_ids=class_ids,
        class_prompts=class_prompts,
        palette=palette,
        overlay_alpha=_threshold(visualization.get("alpha"), "visualization.alpha"),
    )
