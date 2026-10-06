"""SAM-3 text-query inference in whole-image and deployment-tiled modes."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from PIL import Image

from .config import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CHECKPOINT_SIZE,
    MODEL_VARIANT,
    SAM3_CLASS_PROMPTS,
    SAM3_PROCESSOR_RESOLUTION,
    SAM3_TRAINING_TILE_SIZE,
    SAM3_UPSTREAM_COMMIT,
    CrackSegConfig,
)
from .models import load_model
from .postprocessing import normalize_mask_probabilities, union_selected_candidates


InferenceProgress = Callable[[int, int], None]


def _validate_threshold(name: str, value: float) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def make_positions(length: int, tile: int, stride: int) -> list[int]:
    """Return edge-anchored starts that completely cover one image axis."""
    if length <= 0:
        raise ValueError("length must be positive")
    if tile <= 0:
        raise ValueError("tile must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if stride > tile:
        raise ValueError("stride cannot exceed tile")
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def pad_image_to_tile_grid(
    image: Image.Image,
    *,
    tile_size: int,
    padding_mode: str,
) -> tuple[Image.Image, dict[str, Any]]:
    """Pad only the right and bottom edges; source pixels are never resized.

    The AUG512 handoff rejects non-divisible images.  Production DJI frames
    are therefore padded to a 512 grid and every prediction is cropped back to
    the exact source HxW before any ray mapping or metric calculation.
    """
    rgb = image.convert("RGB")
    target_width = ((rgb.width + tile_size - 1) // tile_size) * tile_size
    target_height = ((rgb.height + tile_size - 1) // tile_size) * tile_size
    pad_right = target_width - rgb.width
    pad_bottom = target_height - rgb.height
    metadata = {
        "policy": "pad_right_bottom_then_crop",
        "padding_mode": padding_mode,
        "source_shape_hw": [rgb.height, rgb.width],
        "padded_shape_hw": [target_height, target_width],
        "pad_left_px": 0,
        "pad_top_px": 0,
        "pad_right_px": pad_right,
        "pad_bottom_px": pad_bottom,
        "source_resized": False,
        "output_cropped_to_source_shape": True,
        "deployment_extension": bool(pad_right or pad_bottom),
    }
    if not pad_right and not pad_bottom:
        return rgb, metadata

    array = np.asarray(rgb)
    padded = np.pad(
        array,
        ((0, pad_bottom), (0, pad_right), (0, 0)),
        mode=padding_mode,
    )
    return Image.fromarray(padded.astype(np.uint8, copy=False), mode="RGB"), metadata


def stitch_tile_masks(
    destination: Mapping[str, np.ndarray],
    tile_masks: Mapping[str, np.ndarray],
    *,
    x0: int,
    y0: int,
) -> None:
    """OR a tile's independent masks into original-image pixel coordinates."""
    if set(destination) != set(tile_masks):
        raise ValueError("destination and tile mask class keys must match")
    for class_name, canvas in destination.items():
        if canvas.ndim != 2 or canvas.dtype != np.bool_:
            raise ValueError("destination masks must be two-dimensional bool arrays")
        tile = np.asarray(tile_masks[class_name], dtype=bool)
        if tile.ndim != 2:
            raise ValueError("tile masks must be two-dimensional")
        tile_height, tile_width = tile.shape
        y1 = y0 + tile_height
        x1 = x0 + tile_width
        if x0 < 0 or y0 < 0 or x1 > canvas.shape[1] or y1 > canvas.shape[0]:
            raise ValueError("tile mask lies outside the destination image")
        np.logical_or(canvas[y0:y1, x0:x1], tile, out=canvas[y0:y1, x0:x1])


class InferenceEngine:
    """Strictly loaded SAM-3 with locked whole-image and tiled inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        config: CrackSegConfig | None = None,
        device: str = "auto",
        confidence_threshold: float | None = None,
        mask_threshold: float | None = None,
        candidate_collection_floor: float | None = None,
        inference_mode: str | None = None,
        tile_size: int | None = None,
        stride: int | None = None,
        processor_resolution: int | None = None,
        edge_policy: str | None = None,
        padding_mode: str | None = None,
        expected_size: int | None = None,
        expected_sha256: str | None = None,
        verify_sha256: bool = True,
    ) -> None:
        self.confidence_threshold = _validate_threshold(
            "confidence_threshold",
            confidence_threshold
            if confidence_threshold is not None
            else (config.confidence_threshold if config else 0.5),
        )
        self.mask_threshold = _validate_threshold(
            "mask_threshold",
            mask_threshold
            if mask_threshold is not None
            else (config.mask_threshold if config else 0.5),
        )
        self.candidate_collection_floor = _validate_threshold(
            "candidate_collection_floor",
            candidate_collection_floor
            if candidate_collection_floor is not None
            else (config.candidate_collection_floor if config else 0.05),
        )
        if self.candidate_collection_floor > self.confidence_threshold:
            raise ValueError(
                "candidate_collection_floor cannot exceed confidence_threshold"
            )

        self.inference_mode = inference_mode or (
            config.inference_mode if config else "whole"
        )
        if self.inference_mode not in {"whole", "tiled"}:
            raise ValueError("inference_mode must be one of: whole, tiled")
        self.tile_size = int(
            tile_size
            if tile_size is not None
            else (config.tile_size if config else SAM3_TRAINING_TILE_SIZE)
        )
        self.stride = int(
            stride
            if stride is not None
            else (config.stride if config else SAM3_TRAINING_TILE_SIZE)
        )
        self.processor_resolution = int(
            processor_resolution
            if processor_resolution is not None
            else (
                config.processor_resolution if config else SAM3_PROCESSOR_RESOLUTION
            )
        )
        make_positions(1, self.tile_size, self.stride)
        if self.tile_size != SAM3_TRAINING_TILE_SIZE or self.stride != SAM3_TRAINING_TILE_SIZE:
            raise ValueError("AUG512 inference requires tile_size=512 and stride=512")
        self.edge_policy = edge_policy or (config.edge_policy if config else "error")
        if self.edge_policy not in {"error", "pad"}:
            raise ValueError("edge_policy must be one of: error, pad")
        self.padding_mode = padding_mode or (
            config.padding_mode if config else "reflect"
        )
        if self.padding_mode not in {"reflect", "edge", "constant"}:
            raise ValueError("padding_mode must be one of: reflect, edge, constant")
        if self.processor_resolution != SAM3_PROCESSOR_RESOLUTION:
            raise ValueError(
                "processor_resolution must be 1008 for the pinned SAM-3 "
                "inference contract"
            )

        configured_size = config.expected_checkpoint_size if config else None
        configured_sha256 = config.expected_checkpoint_sha256 if config else None
        model_expected_size = (
            expected_size
            if expected_size is not None
            else (configured_size or EXPECTED_CHECKPOINT_SIZE)
        )
        model_expected_sha256 = (
            expected_sha256
            if expected_sha256 is not None
            else (configured_sha256 or EXPECTED_CHECKPOINT_SHA256)
        )
        self.model, self.checkpoint = load_model(
            checkpoint_path,
            device=device,
            expected_size=model_expected_size,
            expected_sha256=model_expected_sha256,
            verify_sha256=verify_sha256,
        )
        self.device = str(self.checkpoint["device"])
        self.class_prompts = dict(
            config.class_prompts if config else SAM3_CLASS_PROMPTS
        )
        if self.class_prompts != SAM3_CLASS_PROMPTS:
            raise ValueError("SAM-3 class prompts must match the fixed handoff prompts")

        try:
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            raise ImportError(
                "The pinned SAM-3 image processor is unavailable; install the "
                "upstream commit recorded in configs/dachung.yaml."
            ) from exc

        self.processor = Sam3Processor(
            self.model,
            resolution=self.processor_resolution,
            device=self.device,
            confidence_threshold=self.candidate_collection_floor,
        )

    @classmethod
    def from_config(
        cls,
        checkpoint_path: str | Path,
        config: CrackSegConfig,
        **overrides: Any,
    ) -> "InferenceEngine":
        """Construct from config while allowing explicit CLI-style overrides."""
        return cls(checkpoint_path, config=config, **overrides)

    def _predict_queries(
        self, image: Image.Image
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """Run the fixed prompts with the locked whole-image mechanics."""
        rgb = image.convert("RGB")
        masks: dict[str, np.ndarray] = {}
        class_results: dict[str, dict[str, Any]] = {}

        for class_name, prompt in self.class_prompts.items():
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if self.device == "cuda"
                else nullcontext()
            )
            with torch.inference_mode(), autocast:
                # The locked evaluator intentionally recomputes this per class.
                state = self.processor.set_image(rgb)
                output = self.processor.set_text_prompt(state=state, prompt=prompt)

            probabilities = normalize_mask_probabilities(output["masks_logits"])
            scores = output["scores"].detach().float().reshape(-1)
            if tuple(probabilities.shape[-2:]) != (rgb.height, rgb.width):
                raise RuntimeError(
                    f"prediction size {tuple(probabilities.shape[-2:])} does not "
                    f"match source {(rgb.height, rgb.width)}"
                )
            available = min(int(probabilities.shape[0]), int(scores.numel()))
            mask, keep = union_selected_candidates(
                probabilities,
                scores,
                confidence_threshold=self.confidence_threshold,
                mask_threshold=self.mask_threshold,
            )
            scores_cpu = scores[:available].detach().cpu()
            keep_cpu = keep.detach().cpu()
            selected_scores = scores_cpu[keep_cpu].tolist()
            mask_np = mask.detach().cpu().numpy().astype(bool, copy=False)
            masks[class_name] = mask_np
            class_results[class_name] = {
                "prompt": prompt,
                "candidate_count": available,
                "accepted_candidate_count": int(keep_cpu.sum().item()),
                "candidate_scores": [float(value) for value in scores_cpu.tolist()],
                "accepted_candidate_scores": [
                    float(value) for value in selected_scores
                ],
                "top_score": float(scores_cpu.max().item()) if available else None,
                "mask_area_pixels": int(mask_np.sum()),
            }
            del state, output, probabilities, scores, mask, keep

        return masks, class_results

    def _predict_tiled(
        self,
        rgb: Image.Image,
        *,
        tile_size: int,
        stride: int,
        progress_callback: InferenceProgress | None,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, dict[str, Any]],
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        if self.edge_policy == "error":
            if rgb.width % tile_size or rgb.height % tile_size:
                raise ValueError(
                    "AUG512 edge_policy=error requires image width and height "
                    f"divisible by 512; received {rgb.width}x{rgb.height}"
                )
            working_rgb = rgb
            edge_metadata = {
                "policy": "error",
                "padding_mode": None,
                "source_shape_hw": [rgb.height, rgb.width],
                "padded_shape_hw": [rgb.height, rgb.width],
                "pad_left_px": 0,
                "pad_top_px": 0,
                "pad_right_px": 0,
                "pad_bottom_px": 0,
                "source_resized": False,
                "output_cropped_to_source_shape": False,
                "deployment_extension": False,
            }
        else:
            working_rgb, edge_metadata = pad_image_to_tile_grid(
                rgb,
                tile_size=tile_size,
                padding_mode=self.padding_mode,
            )

        x_positions = make_positions(working_rgb.width, tile_size, stride)
        y_positions = make_positions(working_rgb.height, tile_size, stride)
        total_tiles = len(x_positions) * len(y_positions)
        full_masks = {
            class_name: np.zeros((working_rgb.height, working_rgb.width), dtype=bool)
            for class_name in self.class_prompts
        }
        class_results: dict[str, dict[str, Any]] = {
            class_name: {
                "prompt": prompt,
                "candidate_count": 0,
                "accepted_candidate_count": 0,
                "candidate_scores": [],
                "accepted_candidate_scores": [],
                "top_score": None,
                "mask_area_pixels": 0,
            }
            for class_name, prompt in self.class_prompts.items()
        }
        tile_results: list[dict[str, Any]] = []
        tile_index = 0

        for y0 in y_positions:
            for x0 in x_positions:
                x1 = x0 + tile_size
                y1 = y0 + tile_size
                tile_masks, tile_classes = self._predict_queries(
                    working_rgb.crop((x0, y0, x1, y1))
                )
                stitch_tile_masks(full_masks, tile_masks, x0=x0, y0=y0)

                tile_index += 1
                tile_results.append(
                    {
                        "tile_index": tile_index,
                        "bbox_xyxy": [x0, y0, x1, y1],
                        "contains_padding": bool(x1 > rgb.width or y1 > rgb.height),
                        "valid_source_bbox_xyxy": [
                            x0,
                            y0,
                            min(x1, rgb.width),
                            min(y1, rgb.height),
                        ],
                        "classes": {
                            class_name: {
                                "candidate_count": result["candidate_count"],
                                "accepted_candidate_count": result[
                                    "accepted_candidate_count"
                                ],
                                "top_score": result["top_score"],
                                "mask_area_pixels_before_stitch": result[
                                    "mask_area_pixels"
                                ],
                            }
                            for class_name, result in tile_classes.items()
                        },
                    }
                )
                for class_name, result in tile_classes.items():
                    aggregate = class_results[class_name]
                    aggregate["candidate_count"] += result["candidate_count"]
                    aggregate["accepted_candidate_count"] += result[
                        "accepted_candidate_count"
                    ]
                    aggregate["candidate_scores"].extend(result["candidate_scores"])
                    aggregate["accepted_candidate_scores"].extend(
                        result["accepted_candidate_scores"]
                    )
                    top_score = result["top_score"]
                    if top_score is not None and (
                        aggregate["top_score"] is None
                        or top_score > aggregate["top_score"]
                    ):
                        aggregate["top_score"] = top_score

                if progress_callback is not None:
                    progress_callback(tile_index, total_tiles)

        cropped_masks = {
            class_name: mask[: rgb.height, : rgb.width].copy()
            for class_name, mask in full_masks.items()
        }
        for class_name, mask in cropped_masks.items():
            class_results[class_name]["mask_area_pixels"] = int(mask.sum())
        return cropped_masks, class_results, tile_results, edge_metadata

    def predict(
        self,
        image: Image.Image,
        *,
        image_name: str,
        inference_mode: str | None = None,
        progress_callback: InferenceProgress | None = None,
    ) -> dict[str, Any]:
        """Predict independent CRC/DLM/SPL masks at the original HxW.

        Tile size and stride are intentionally not per-call options.  This
        checkpoint was trained and validated with non-overlapping 512 x 512
        source tiles, so changing either value would silently break the model
        contract while retaining the same model-variant label.
        """
        rgb = image.convert("RGB")
        mode = inference_mode or self.inference_mode
        if mode not in {"whole", "tiled"}:
            raise ValueError("inference_mode must be one of: whole, tiled")
        run_tile_size = self.tile_size
        run_stride = self.stride
        if (
            run_tile_size != SAM3_TRAINING_TILE_SIZE
            or run_stride != SAM3_TRAINING_TILE_SIZE
        ):
            raise RuntimeError(
                "AUG512 engine state is invalid: tile_size and stride must remain 512"
            )
        if mode == "whole" and rgb.size != (
            SAM3_TRAINING_TILE_SIZE,
            SAM3_TRAINING_TILE_SIZE,
        ):
            raise ValueError(
                "AUG512 whole-image execution is valid only for one 512x512 tile"
            )

        if mode == "whole":
            masks, class_results = self._predict_queries(rgb)
            tile_results: list[dict[str, Any]] = []
            if progress_callback is not None:
                progress_callback(1, 1)
        else:
            masks, class_results, tile_results, edge_metadata = self._predict_tiled(
                rgb,
                tile_size=run_tile_size,
                stride=run_stride,
                progress_callback=progress_callback,
            )

        if self.device == "cuda":
            torch.cuda.empty_cache()

        verified_aug512_tile_equivalent = (
            mode == "whole"
            and rgb.size == (SAM3_TRAINING_TILE_SIZE, SAM3_TRAINING_TILE_SIZE)
            and self.candidate_collection_floor == 0.05
            and self.confidence_threshold == 0.5
            and self.mask_threshold == 0.5
            and self.processor_resolution == SAM3_PROCESSOR_RESOLUTION
            and self.class_prompts == SAM3_CLASS_PROMPTS
            and self.checkpoint.get("size") == EXPECTED_CHECKPOINT_SIZE
            and self.checkpoint.get("sha256") == EXPECTED_CHECKPOINT_SHA256
            and self.checkpoint.get("expected_sha256")
            == EXPECTED_CHECKPOINT_SHA256
            and self.checkpoint.get("sha256_match") is True
            and self.checkpoint.get("checkpoint_identity_verified") is True
            and self.checkpoint.get("checkpoint_load_preflight_passed") is True
            and self.checkpoint.get("actual_upstream_commit")
            == SAM3_UPSTREAM_COMMIT
            and self.checkpoint.get("expected_upstream_commit")
            == SAM3_UPSTREAM_COMMIT
            and self.checkpoint.get("upstream_commit_matches") is True
            and self.checkpoint.get("upstream_source_clean") is True
            and self.checkpoint.get("sam3_source_verified") is True
            and self.checkpoint.get("upstream_source_provenance_verified") is True
        )
        inference_metadata: dict[str, Any] = {
            "mode": mode,
            "candidate_collection_floor": self.candidate_collection_floor,
            "confidence_threshold": self.confidence_threshold,
            "mask_threshold": self.mask_threshold,
            "candidate_mask_operation": "binary_union",
            "class_mask_semantics": "independent_multilabel",
            "top_k": None,
            "processor_resolution": self.processor_resolution,
            "output_coordinate_space": "original_image_pixels",
            "output_shape_hw": [rgb.height, rgb.width],
            "whole_image": mode == "whole",
            "tiling": mode == "tiled",
            "model_variant": MODEL_VARIANT,
            "verified_aug512_single_tile_execution": verified_aug512_tile_equivalent,
        }
        if mode == "tiled":
            inference_metadata.update(
                {
                    "tile_size": run_tile_size,
                    "stride": run_stride,
                    "tile_count": len(tile_results),
                    "edge_handling": edge_metadata,
                    "tile_merge_operation": "logical_or_per_class",
                    "candidate_count_aggregation": (
                        "sum_across_tiles; overlapping candidates are not unique "
                        "damage instances"
                    ),
                    "deployment_setting": True,
                    "official_handoff_protocol": not bool(
                        edge_metadata["deployment_extension"]
                    ),
                    "coordinate_note": (
                        "Masks are cropped to the untouched source pixel grid; "
                        "ray coordinates use original image pixels."
                    ),
                    "tiles": tile_results,
                }
            )

        return {
            "image": image_name,
            "image_width": rgb.width,
            "image_height": rgb.height,
            "checkpoint": {
                "filename": self.checkpoint["filename"],
                "size": self.checkpoint["size"],
                "expected_size": self.checkpoint["expected_size"],
                "size_matches_verified_model": self.checkpoint[
                    "size_matches_verified_model"
                ],
                "sha256": self.checkpoint["sha256"],
                "expected_sha256": self.checkpoint["expected_sha256"],
                "sha256_match": self.checkpoint["sha256_match"],
                "checkpoint_identity_verified": self.checkpoint[
                    "checkpoint_identity_verified"
                ],
                "checkpoint_load_preflight_passed": self.checkpoint[
                    "checkpoint_load_preflight_passed"
                ],
                "strict_load_passed": self.checkpoint["strict_load_passed"],
                "checkpoint_epoch": self.checkpoint["checkpoint_epoch"],
                "model_state_key_count": self.checkpoint["model_state_key_count"],
                "upstream_commit": self.checkpoint["upstream_commit"],
                "expected_upstream_commit": self.checkpoint[
                    "expected_upstream_commit"
                ],
                "actual_upstream_commit": self.checkpoint[
                    "actual_upstream_commit"
                ],
                "upstream_commit_matches": self.checkpoint[
                    "upstream_commit_matches"
                ],
                "upstream_source_path": self.checkpoint[
                    "upstream_source_path"
                ],
                "upstream_package_path": self.checkpoint[
                    "upstream_package_path"
                ],
                "upstream_imported_module_paths": self.checkpoint[
                    "upstream_imported_module_paths"
                ],
                "upstream_repository_root": self.checkpoint[
                    "upstream_repository_root"
                ],
                "upstream_source_clean": self.checkpoint[
                    "upstream_source_clean"
                ],
                "upstream_source_dirty_entry_count": self.checkpoint[
                    "upstream_source_dirty_entry_count"
                ],
                "sam3_source_verified": self.checkpoint[
                    "sam3_source_verified"
                ],
                "upstream_source_provenance_verified": self.checkpoint[
                    "upstream_source_provenance_verified"
                ],
            },
            "inference": inference_metadata,
            "classes": class_results,
            "masks": masks,
        }
