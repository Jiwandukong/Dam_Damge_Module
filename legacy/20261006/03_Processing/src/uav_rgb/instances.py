from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from skimage.measure import label, regionprops

from .metrics import ClassMasks, validate_class_masks
from .surface_metrics import measure_oriented_pixel_geometry


InstanceLabelMaps = Mapping[int, np.ndarray]


def summarize_class_mask_instances(
    class_masks: ClassMasks,
    class_names: dict[int, str],
    min_area_px: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, np.ndarray]]:
    """Measure independent class masks and return stable per-class instance maps.

    Connected components use 8-connectivity.  Each returned label map is
    relabeled from one after area filtering and has ``uint32`` dtype.  The
    scalar ``instance_local_id`` in an instance row is therefore sufficient
    to recover exactly that component without placing an array in a CSV row.
    """

    height, width = validate_class_masks(class_masks)
    if min_area_px < 1:
        raise ValueError("min_area_px must be >= 1")

    instance_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    instance_label_maps: dict[int, np.ndarray] = {}
    instance_id = 1

    ordered_class_ids = [int(class_id) for class_id in class_names if int(class_id) != 0]
    ordered_class_ids.extend(
        sorted(int(class_id) for class_id in class_masks if int(class_id) not in ordered_class_ids)
    )

    empty_mask = np.zeros((height, width), dtype=bool)
    for class_id in ordered_class_ids:
        cls_name = class_names.get(class_id, str(class_id))
        class_mask = class_masks.get(class_id, empty_mask)
        raw_labels = label(class_mask, connectivity=2)
        props = [prop for prop in regionprops(raw_labels) if int(prop.area) >= min_area_px]

        relabel_lut = np.zeros(int(raw_labels.max()) + 1, dtype=np.uint32)
        for local_id, prop in enumerate(props, start=1):
            relabel_lut[int(prop.label)] = np.uint32(local_id)
        relabeled = relabel_lut[raw_labels]
        instance_label_maps[class_id] = relabeled

        total_area = 0
        for local_id, prop in enumerate(props, start=1):
            y0, x0, y1, x1 = prop.bbox
            cy, cx = prop.centroid
            area = int(prop.area)
            oriented = measure_oriented_pixel_geometry(
                prop.image,
                offset_x=x0,
                offset_y=y0,
            )
            is_crack = int(class_id) == 1
            total_area += area
            instance_rows.append(
                {
                    "instance_id": instance_id,
                    "instance_local_id": local_id,
                    "class_id": class_id,
                    "class_name": cls_name,
                    "centroid_x_px": round(float(cx), 4),
                    "centroid_y_px": round(float(cy), 4),
                    "bbox_xmin_px": int(x0),
                    "bbox_ymin_px": int(y0),
                    "bbox_xmax_px": int(x1),
                    "bbox_ymax_px": int(y1),
                    "bbox_w_px": int(x1 - x0),
                    "bbox_h_px": int(y1 - y0),
                    "area_px": area,
                    # Match the requested report semantics: CRC receives a
                    # rotated-extent length/width; DLM and SPL use mask area.
                    "length_px": (
                        round(float(oriented.length_px), 6)
                        if is_crack and oriented is not None
                        else None
                    ),
                    "width_px": (
                        round(float(oriented.width_px), 6)
                        if is_crack and oriented is not None
                        else None
                    ),
                    "measurement_angle_deg": (
                        round(float(oriented.angle_deg), 6)
                        if is_crack and oriented is not None
                        else None
                    ),
                }
            )
            instance_id += 1

        class_rows.append(
            {
                "class_id": class_id,
                "class_name": cls_name,
                "instance_count": len(props),
                "total_area_px": int(total_area),
            }
        )

    return instance_rows, class_rows, instance_label_maps


# Descriptive alias for callers that use "multilabel" terminology.
summarize_multilabel_instances = summarize_class_mask_instances


def summarize_instances(
    pred_mask: np.ndarray,
    class_names: dict[int, str],
    min_area_px: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(pred_mask, np.ndarray) or pred_mask.ndim != 2:
        raise ValueError("pred_mask must be a two-dimensional numpy array")
    class_masks = {
        int(class_id): pred_mask == int(class_id)
        for class_id in class_names
        if int(class_id) != 0
    }
    rows, class_rows, _ = summarize_class_mask_instances(
        class_masks=class_masks,
        class_names=class_names,
        min_area_px=min_area_px,
    )
    return rows, class_rows


def validate_instance_label_maps(
    instance_label_maps: InstanceLabelMaps,
    expected_shape: tuple[int, int] | None = None,
) -> tuple[int, int]:
    if not isinstance(instance_label_maps, Mapping):
        raise TypeError("instance_label_maps must map positive class IDs to uint32 HxW arrays")
    if not instance_label_maps:
        raise ValueError("instance_label_maps must not be empty")

    shape = expected_shape
    for raw_class_id, label_map in instance_label_maps.items():
        class_id = int(raw_class_id)
        if class_id <= 0:
            raise ValueError("instance label-map class IDs must be > 0")
        if not isinstance(label_map, np.ndarray) or label_map.ndim != 2:
            raise ValueError(f"instance label map {class_id} must be a two-dimensional numpy array")
        if label_map.dtype != np.uint32:
            raise TypeError(f"instance label map {class_id} must have uint32 dtype, got {label_map.dtype}")
        current_shape = (int(label_map.shape[0]), int(label_map.shape[1]))
        if shape is None:
            shape = current_shape
        elif current_shape != shape:
            raise ValueError(
                f"all instance label maps must have shape {shape}; class {class_id} has {current_shape}"
            )
    assert shape is not None
    return shape


def instance_submask(
    instance_row: Mapping[str, Any],
    segmentation: np.ndarray | ClassMasks | None,
    *,
    instance_label_maps: InstanceLabelMaps | None = None,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
) -> np.ndarray:
    """Return only the requested component inside a global-image bounding box."""

    class_id = int(instance_row["class_id"])
    expected_shape: tuple[int, int] | None = None
    if isinstance(segmentation, Mapping):
        expected_shape = validate_class_masks(segmentation)
    elif segmentation is not None:
        if not isinstance(segmentation, np.ndarray) or segmentation.ndim != 2:
            raise ValueError("segmentation must be a class-mask mapping or a two-dimensional class-ID mask")
        expected_shape = (int(segmentation.shape[0]), int(segmentation.shape[1]))

    if instance_label_maps is not None:
        shape = validate_instance_label_maps(instance_label_maps, expected_shape=expected_shape)
        label_map = instance_label_maps.get(class_id)
        if label_map is None:
            return np.zeros((max(0, ymax - ymin), max(0, xmax - xmin)), dtype=bool)
        local_id = int(instance_row["instance_local_id"])
        return label_map[ymin:ymax, xmin:xmax] == local_id

    if segmentation is None:
        raise ValueError("segmentation or instance_label_maps is required")
    if isinstance(segmentation, Mapping):
        class_mask = segmentation.get(class_id)
        if class_mask is None:
            return np.zeros((max(0, ymax - ymin), max(0, xmax - xmin)), dtype=bool)
        binary = class_mask
    else:
        binary = segmentation == class_id

    # Backward-compatible callers may not have retained label maps.  Recover a
    # single component from the scalar row instead of using every class pixel
    # in its bbox, which could mix nested/interleaved islands.
    component = _component_for_row(binary, instance_row)
    return component[ymin:ymax, xmin:xmax]


def snapped_component_pixel(
    submask: np.ndarray,
    *,
    offset_x: int,
    offset_y: int,
    centroid_x: float,
    centroid_y: float,
) -> tuple[float, float] | None:
    """Return the component pixel nearest its geometric centroid."""

    ys, xs = np.nonzero(submask)
    if len(xs) == 0:
        return None
    global_x = xs.astype(np.float64) + float(offset_x)
    global_y = ys.astype(np.float64) + float(offset_y)
    distances_sq = (global_x - float(centroid_x)) ** 2 + (global_y - float(centroid_y)) ** 2
    nearest = int(np.argmin(distances_sq))
    return float(global_x[nearest]), float(global_y[nearest])


def _component_for_row(binary: np.ndarray, instance_row: Mapping[str, Any]) -> np.ndarray:
    labels = label(binary, connectivity=2)
    props = regionprops(labels)
    if not props:
        return np.zeros_like(binary, dtype=bool)

    target_bbox = (
        int(instance_row["bbox_ymin_px"]),
        int(instance_row["bbox_xmin_px"]),
        int(instance_row["bbox_ymax_px"]),
        int(instance_row["bbox_xmax_px"]),
    )
    exact = [prop for prop in props if tuple(int(value) for value in prop.bbox) == target_bbox]
    candidates = exact or props
    centroid_x = float(instance_row["centroid_x_px"])
    centroid_y = float(instance_row["centroid_y_px"])
    chosen = min(
        candidates,
        key=lambda prop: (float(prop.centroid[1]) - centroid_x) ** 2
        + (float(prop.centroid[0]) - centroid_y) ** 2,
    )
    return labels == int(chosen.label)
