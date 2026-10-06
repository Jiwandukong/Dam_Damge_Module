"""Measure full-photo model masks without reading labels or training data.

Each class keeps its own 8-connected components. Components are found before
matching native 512-pixel tiles, so a tile boundary never splits a CSV instance.
The production mapper retains its rotated-rectangle and local mesh-GSD metrics.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from .geometry.instances import summarize_class_mask_instances
from .geometry.mesh_ray import (
    MeshRayGeo3DContext, empty_mesh_hit, empty_surface_measurement,
    extract_contour_nodes,
)
from .geometry.metrics import ClassMasks, validate_class_masks


CLASS_NAMES = {1: "CRC", 2: "DLM", 3: "SPL"}
TILE_SIZE = 512


def _component_tiles(base: dict[str, Any], label_map: np.ndarray) -> list[tuple[int, int]]:
    """Return row-major tile origins containing pixels of this exact component."""
    x0, y0 = int(base["bbox_xmin_px"]), int(base["bbox_ymin_px"])
    x1, y1 = int(base["bbox_xmax_px"]), int(base["bbox_ymax_px"])
    local_id = int(base["instance_local_id"])
    origins = []
    for tile_y in range((y0 // TILE_SIZE) * TILE_SIZE, y1, TILE_SIZE):
        for tile_x in range((x0 // TILE_SIZE) * TILE_SIZE, x1, TILE_SIZE):
            crop = label_map[max(y0, tile_y):min(y1, tile_y + TILE_SIZE),
                             max(x0, tile_x):min(x1, tile_x + TILE_SIZE)]
            if np.any(crop == local_id):
                origins.append((tile_x, tile_y))
    return origins


def _pixel_contour_fallback(base: dict[str, Any], label_map: np.ndarray,
                            spatial: dict[str, Any]) -> None:
    """Preserve pixel-only display nodes, never fabricate world coordinates."""
    if spatial.get("nodes_image_xy_json") and spatial["nodes_image_xy_json"] != "[]":
        return
    x0, y0 = int(base["bbox_xmin_px"]), int(base["bbox_ymin_px"])
    x1, y1 = int(base["bbox_xmax_px"]), int(base["bbox_ymax_px"])
    submask = label_map[y0:y1, x0:x1] == int(base["instance_local_id"])
    nodes = extract_contour_nodes(submask, offset_x=x0, offset_y=y0, max_count=256)
    spatial.update({
        "nodes_image_xy_json": json.dumps(
            [[round(float(x), 4), round(float(y), 4)] for x, y in nodes],
            separators=(",", ":"),
        ),
        "nodes_world_xyz_json": json.dumps(
            [[None, None, None] for _ in nodes], separators=(",", ":")),
        "node_count": int(len(nodes)),
        "node_xyz_valid_count": 0,
        "node_xyz_hit_ratio": 0.0,
        "nodes_geo3d_source": "pixel_contour_only_no_world_fallback",
        "nodes_miss_reason": "native_mapper_did_not_return_contour_world_coordinates",
    })


def quantify_predictions(
    class_masks: ClassMasks, context: MeshRayGeo3DContext | None,
) -> dict[str, Any]:
    """Quantify one original-resolution photo's independent boolean class masks.

    Return scalar ``base``/``spatial`` records and component-specific
    ``tile_origins``; the caller may put the representative-point tile first.
    No arrays or full-photo label maps are retained in the returned records.
    A missing camera context leaves physical values empty but preserves every
    predicted component and its pixel geometry. No size filtering is applied.
    """
    height, width = validate_class_masks(class_masks)
    if set(class_masks) != set(CLASS_NAMES):
        raise ValueError("Expected exactly the CRC=1, DLM=2 and SPL=3 class masks")
    if height < 1 or width < 1:
        raise ValueError("Prediction image dimensions must be positive")
    if context is not None and (context.intrinsics.width, context.intrinsics.height) != (width, height):
        raise ValueError("Camera intrinsics dimensions do not match the original source image")

    base_rows, class_summary, instance_maps = summarize_class_mask_instances(
        class_masks, class_names=CLASS_NAMES, min_area_px=1)
    instances = []
    for base in base_rows:
        label_map = instance_maps[int(base["class_id"])]
        if context is None:
            reason = "camera_metadata_unavailable"
            spatial = empty_mesh_hit(reason, None, None)
            spatial.update(empty_surface_measurement(base, reason))
        else:
            spatial = dict(context.image_region_to_world3d(
                base, pred_mask=class_masks, instance_label_maps=instance_maps))
        _pixel_contour_fallback(base, label_map, spatial)
        instances.append({
            "base": base,
            "spatial": spatial,
            "tile_origins": _component_tiles(base, label_map),
        })
    return {
        "instances": instances,
        "class_summary": class_summary,
        "component_counts": {
            row["class_name"]: int(row["instance_count"]) for row in class_summary
        },
    }
