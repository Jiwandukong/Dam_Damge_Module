from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from skimage.measure import find_contours

from .instances import InstanceLabelMaps, instance_submask, snapped_component_pixel
from .camera_pose import (
    CameraIntrinsics,
    CameraPose,
    camera_to_world_matrix,
    intrinsics_from_xmp,
    pose_from_xmp,
    read_dji_xmp,
)
from .metrics import ClassMasks, validate_class_masks
from .surface_metrics import OrientedPixelGeometry, measure_oriented_pixel_geometry


SURFACE_MEASUREMENT_METHOD = "mesh_ray_local_directional_gsd_min_area_rectangle"


class MeshSurfaceIndex:
    def __init__(
        self,
        path: str | Path,
        ray_backend: str = "trimesh",
        warp_device: str = "cuda:0",
    ):
        mesh_path = Path(path)
        if mesh_path.suffix.lower() in {".glb", ".gltf"}:
            raise ValueError(
                "GLB/glTF input is disabled for this mapping pipeline because the "
                "delivered DaecheongDam_v0829_grid.glb has no transform into the "
                "EPSG:5186 camera frame. Use the georeferenced dam - Cloud.obj."
            )
        if mesh_path.suffix.lower() != ".obj":
            raise ValueError(
                "Only the externally georeferenced EPSG:5186/Z-up OBJ is supported"
            )
        try:
            import trimesh
        except ImportError as exc:
            raise ImportError(
                "trimesh and rtree are required; install this project with pip"
            ) from exc

        ray_backend = str(ray_backend)
        if ray_backend not in {"trimesh", "warp", "auto"}:
            raise ValueError("ray_backend must be 'trimesh', 'warp', or 'auto'")

        self.path = str(mesh_path)
        self.ray_backend_requested = ray_backend
        self.warp_device = str(warp_device)
        self.ray_backend = "trimesh"
        self.ray_backend_error: str | None = None
        self._warp_raycaster = None

        loaded = trimesh.load_mesh(self.path, process=False)
        if hasattr(loaded, "geometry"):
            loaded = loaded.dump(concatenate=True)
        if loaded.faces is None or len(loaded.faces) == 0:
            raise ValueError(f"Mesh has no faces: {self.path}")

        self.mesh = loaded
        self.bounds = np.asarray(self.mesh.bounds, dtype=np.float64)
        self.vertex_count = int(len(self.mesh.vertices))
        self.face_count = int(len(self.mesh.faces))
        if ray_backend in {"warp", "auto"}:
            try:
                from .warp_ray import WarpMeshRaycaster

                self._warp_raycaster = WarpMeshRaycaster(
                    vertices=np.asarray(self.mesh.vertices),
                    faces=np.asarray(self.mesh.faces),
                    device=self.warp_device,
                    origin_offset=0.5 * (self.bounds[0] + self.bounds[1]),
                )
                self.ray_backend = "warp"
            except Exception as exc:
                self.ray_backend_error = f"{type(exc).__name__}: {exc}"
                if ray_backend == "warp":
                    raise RuntimeError(f"Failed to initialize Warp mesh ray backend: {self.ray_backend_error}") from exc

    def intersect_ray(self, origin: np.ndarray, direction: np.ndarray) -> dict[str, Any]:
        hits = self.intersect_rays(origin[None, :], direction[None, :])
        if not hits:
            return empty_mesh_hit("mesh_ray_no_hit", None, None)
        return hits[0]

    def intersect_rays(self, origins: np.ndarray, directions: np.ndarray) -> list[dict[str, Any]]:
        origins = np.asarray(origins, dtype=np.float64)
        directions = np.asarray(directions, dtype=np.float64)
        directions = directions / np.linalg.norm(directions, axis=1)[:, None]
        if self._warp_raycaster is not None:
            return self._intersect_rays_warp(origins, directions)

        locations, ray_indices, face_indices = self.mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False,
        )
        by_ray: dict[int, tuple[np.ndarray, int, float]] = {}
        for location, ray_index, face_index in zip(locations, ray_indices, face_indices):
            ray_index = int(ray_index)
            t = float(np.dot(location - origins[ray_index], directions[ray_index]))
            current = by_ray.get(ray_index)
            if current is None or t < current[2]:
                by_ray[ray_index] = (np.asarray(location, dtype=np.float64), int(face_index), t)

        rows: list[dict[str, Any]] = []
        for ray_index in range(len(origins)):
            hit = by_ray.get(ray_index)
            if hit is None:
                row = empty_mesh_hit("mesh_ray_no_hit", None, None)
                row["mesh_ray_backend"] = self.ray_backend
                rows.append(row)
                continue
            location, face_index, t = hit
            rows.append(
                {
                    "world_x_m": round(float(location[0]), 8),
                    "world_y_m": round(float(location[1]), 8),
                    "world_z_m": round(float(location[2]), 8),
                    "geo3d_source": "dji_pose_mesh_ray",
                    "xyz_valid": True,
                    "mesh_ray_hit": True,
                    "mesh_face_index": int(face_index),
                    "mesh_ray_t_m": round(float(t), 6),
                    "mesh_ray_backend": self.ray_backend,
                    "xyz_miss_reason": None,
                }
            )
        return rows

    def _intersect_rays_warp(self, origins: np.ndarray, directions: np.ndarray) -> list[dict[str, Any]]:
        if self._warp_raycaster is None:
            raise RuntimeError("Warp raycaster is not initialized.")

        hit_flags, hit_t, hit_faces, hit_points = self._warp_raycaster.intersect_rays(origins, directions)
        rows: list[dict[str, Any]] = []
        for ray_index in range(len(origins)):
            if int(hit_flags[ray_index]) == 0:
                row = empty_mesh_hit("mesh_ray_no_hit", None, None)
                row["mesh_ray_backend"] = self.ray_backend
                rows.append(row)
                continue

            location = np.asarray(hit_points[ray_index], dtype=np.float64)
            rows.append(
                {
                    "world_x_m": round(float(location[0]), 8),
                    "world_y_m": round(float(location[1]), 8),
                    "world_z_m": round(float(location[2]), 8),
                    "geo3d_source": "dji_pose_mesh_ray",
                    "xyz_valid": True,
                    "mesh_ray_hit": True,
                    "mesh_face_index": int(hit_faces[ray_index]),
                    "mesh_ray_t_m": round(float(hit_t[ray_index]), 6),
                    "mesh_ray_backend": self.ray_backend,
                    "xyz_miss_reason": None,
                }
            )
        return rows

    def to_dict(self) -> dict[str, Any]:
        data = {
            "path": self.path,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "bounds_min": tuple(float(v) for v in self.bounds[0]),
            "bounds_max": tuple(float(v) for v in self.bounds[1]),
            "coordinate_contract": (
                "This OBJ is externally known to use EPSG:5186 X/Y and meter "
                "elevation Z-up; OBJ itself does not embed a CRS."
            ),
            "ray_backend_requested": self.ray_backend_requested,
            "ray_backend": self.ray_backend,
            "warp_device": self.warp_device if self.ray_backend_requested in {"warp", "auto"} else None,
            "ray_backend_error": self.ray_backend_error,
        }
        if self._warp_raycaster is not None:
            data["warp"] = self._warp_raycaster.to_dict()
        return data


@dataclass(frozen=True)
class SurfaceMeasurementStencil:
    """Four mesh-ray samples used to estimate local directional pixel scale."""

    center_xy: tuple[float, float]
    xy: np.ndarray
    length_span_px: float
    width_span_px: float
    length_scheme: str
    width_scheme: str


@dataclass
class MeshRayGeo3DContext:
    intrinsics: CameraIntrinsics
    pose: CameraPose
    surface: MeshSurfaceIndex
    instance_sample_count: int = 128
    node_sample_count: int = 256
    ray_batch_size: int = 128
    representative_mode: str = "centroid"
    measurement_half_span_px: float = 2.0
    profile: bool = False
    timings: dict[str, float] = field(default_factory=dict)

    def image_pixel_to_world3d(self, x: float, y: float) -> dict[str, Any]:
        hit = self._intersect_pixel_batch(np.array([[float(x), float(y)]], dtype=np.float64), desc="centroid mesh rays")[0]
        hit["image_pixel_x"] = round(float(x), 4)
        hit["image_pixel_y"] = round(float(y), 4)
        return hit

    def image_regions_to_world3d(
        self,
        instance_rows: list[dict[str, Any]],
        pred_mask: np.ndarray | ClassMasks | None = None,
        *,
        class_masks: ClassMasks | None = None,
        instance_label_maps: InstanceLabelMaps | None = None,
    ) -> list[dict[str, Any]]:
        segmentation = _resolve_segmentation(pred_mask, class_masks)
        _validate_segmentation_image_shape(segmentation, self.intrinsics.height, self.intrinsics.width)
        total_start = time.perf_counter()
        prepare_start = time.perf_counter()
        prepared: list[dict[str, Any]] = []
        sample_xy_parts: list[np.ndarray] = []
        node_xy_parts: list[np.ndarray] = []
        measurement_xy_parts: list[np.ndarray] = []

        for instance_index, instance_row in enumerate(instance_rows):
            class_id = int(instance_row["class_id"])
            xmin = max(0, int(instance_row["bbox_xmin_px"]))
            ymin = max(0, int(instance_row["bbox_ymin_px"]))
            xmax = min(self.intrinsics.width, int(instance_row["bbox_xmax_px"]))
            ymax = min(self.intrinsics.height, int(instance_row["bbox_ymax_px"]))
            item: dict[str, Any] = {
                "instance_index": instance_index,
                "row": instance_row,
                "sample_start": 0,
                "sample_count": 0,
                "node_start": 0,
                "node_count": 0,
                "measurement_start": 0,
                "measurement_count": 0,
                "measurement_geometry": None,
                "measurement_stencil": None,
                "fallback_reason": None,
            }
            if xmin >= xmax or ymin >= ymax:
                item["fallback_reason"] = "empty_instance_bbox"
                prepared.append(item)
                continue

            submask = instance_submask(
                instance_row,
                segmentation,
                instance_label_maps=instance_label_maps,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
            ys, xs = np.nonzero(submask)
            if len(xs) == 0:
                item["fallback_reason"] = "empty_instance_mask"
                prepared.append(item)
                continue

            sample_xy = self._representative_pixels(instance_row, xs, ys, xmin, ymin)
            node_xy = extract_contour_nodes(submask, offset_x=xmin, offset_y=ymin, max_count=self.node_sample_count)
            geometry = measure_oriented_pixel_geometry(
                submask,
                offset_x=xmin,
                offset_y=ymin,
            )
            measurement_center = component_measurement_center(
                submask,
                geometry,
                offset_x=xmin,
                offset_y=ymin,
            )
            stencil = (
                build_surface_measurement_stencil(
                    geometry,
                    center_xy=np.asarray(measurement_center, dtype=np.float64),
                    image_width=self.intrinsics.width,
                    image_height=self.intrinsics.height,
                    half_span_px=self.measurement_half_span_px,
                )
                if geometry is not None and measurement_center is not None
                else None
            )

            item["sample_start"] = sum(len(part) for part in sample_xy_parts)
            item["sample_count"] = len(sample_xy)
            item["node_start"] = sum(len(part) for part in node_xy_parts)
            item["node_count"] = len(node_xy)
            item["measurement_start"] = sum(len(part) for part in measurement_xy_parts)
            item["measurement_count"] = len(stencil.xy) if stencil is not None else 0
            item["measurement_geometry"] = geometry
            item["measurement_stencil"] = stencil
            sample_xy_parts.append(sample_xy)
            if len(node_xy) > 0:
                node_xy_parts.append(node_xy)
            if stencil is not None:
                measurement_xy_parts.append(stencil.xy)
            prepared.append(item)
        self._add_timing("mesh_prepare_regions_sec", time.perf_counter() - prepare_start)

        sample_xy_all = np.vstack(sample_xy_parts) if sample_xy_parts else np.empty((0, 2), dtype=np.float64)
        node_xy_all = np.vstack(node_xy_parts) if node_xy_parts else np.empty((0, 2), dtype=np.float64)
        measurement_xy_all = (
            np.vstack(measurement_xy_parts)
            if measurement_xy_parts
            else np.empty((0, 2), dtype=np.float64)
        )
        self.timings["mesh_instance_ray_count"] = float(len(sample_xy_all))
        self.timings["mesh_node_ray_count"] = float(len(node_xy_all))
        self.timings["mesh_measurement_ray_count"] = float(len(measurement_xy_all))
        self.timings["mesh_total_ray_count"] = float(
            len(sample_xy_all) + len(node_xy_all) + len(measurement_xy_all)
        )

        sample_start = time.perf_counter()
        sample_hits = self._intersect_pixel_batch(sample_xy_all, desc="instance mesh rays")
        self._add_timing("mesh_instance_intersection_sec", time.perf_counter() - sample_start)

        node_start = time.perf_counter()
        node_hits = self._intersect_pixel_batch(node_xy_all, desc="node mesh rays")
        self._add_timing("mesh_node_intersection_sec", time.perf_counter() - node_start)

        measurement_start = time.perf_counter()
        measurement_hits = self._intersect_pixel_batch(
            measurement_xy_all,
            desc="measurement mesh rays",
        )
        self._add_timing(
            "mesh_measurement_intersection_sec",
            time.perf_counter() - measurement_start,
        )

        assemble_start = time.perf_counter()
        results: list[dict[str, Any]] = []
        for item in prepared:
            row = item["row"]
            if item["fallback_reason"] is not None:
                result = self._centroid_fallback(row, str(item["fallback_reason"]))
                result.update(
                    empty_surface_measurement(
                        row,
                        str(item["fallback_reason"]),
                    )
                )
                results.append(result)
                continue

            sample_start = int(item["sample_start"])
            sample_count = int(item["sample_count"])
            node_start = int(item["node_start"])
            node_count = int(item["node_count"])
            measurement_start = int(item["measurement_start"])
            measurement_count = int(item["measurement_count"])
            sample_slice = sample_hits[sample_start : sample_start + sample_count]
            node_xy = node_xy_all[node_start : node_start + node_count]
            node_slice = node_hits[node_start : node_start + node_count]
            sample_xy = sample_xy_all[sample_start : sample_start + sample_count]
            measurement_slice = measurement_hits[
                measurement_start : measurement_start + measurement_count
            ]
            result = self._result_from_hits(
                row,
                sample_xy,
                sample_slice,
                node_xy,
                node_slice,
            )
            result.update(
                surface_measurement_from_hits(
                    row,
                    item["measurement_geometry"],
                    item["measurement_stencil"],
                    measurement_slice,
                )
            )
            results.append(result)

        self._add_timing("mesh_assemble_results_sec", time.perf_counter() - assemble_start)
        self._add_timing("mesh_total_sec", time.perf_counter() - total_start)
        return results

    def image_region_to_world3d(
        self,
        instance_row: dict[str, Any],
        pred_mask: np.ndarray | ClassMasks | None = None,
        *,
        class_masks: ClassMasks | None = None,
        instance_label_maps: InstanceLabelMaps | None = None,
    ) -> dict[str, Any]:
        segmentation = _resolve_segmentation(pred_mask, class_masks)
        _validate_segmentation_image_shape(segmentation, self.intrinsics.height, self.intrinsics.width)
        xmin = max(0, int(instance_row["bbox_xmin_px"]))
        ymin = max(0, int(instance_row["bbox_ymin_px"]))
        xmax = min(self.intrinsics.width, int(instance_row["bbox_xmax_px"]))
        ymax = min(self.intrinsics.height, int(instance_row["bbox_ymax_px"]))
        if xmin >= xmax or ymin >= ymax:
            result = self._centroid_fallback(instance_row, "empty_instance_bbox")
            result.update(empty_surface_measurement(instance_row, "empty_instance_bbox"))
            return result

        submask = instance_submask(
            instance_row,
            segmentation,
            instance_label_maps=instance_label_maps,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )
        ys, xs = np.nonzero(submask)
        if len(xs) == 0:
            result = self._centroid_fallback(instance_row, "empty_instance_mask")
            result.update(empty_surface_measurement(instance_row, "empty_instance_mask"))
            return result

        sample_xy = self._representative_pixels(instance_row, xs, ys, xmin, ymin)
        node_xy = extract_contour_nodes(submask, offset_x=xmin, offset_y=ymin, max_count=self.node_sample_count)
        geometry = measure_oriented_pixel_geometry(
            submask,
            offset_x=xmin,
            offset_y=ymin,
        )
        measurement_center = component_measurement_center(
            submask,
            geometry,
            offset_x=xmin,
            offset_y=ymin,
        )
        stencil = (
            build_surface_measurement_stencil(
                geometry,
                center_xy=np.asarray(measurement_center, dtype=np.float64),
                image_width=self.intrinsics.width,
                image_height=self.intrinsics.height,
                half_span_px=self.measurement_half_span_px,
            )
            if geometry is not None and measurement_center is not None
            else None
        )
        hits = self._intersect_pixel_batch(sample_xy, desc="instance mesh rays")
        node_hits = self._intersect_pixel_batch(node_xy, desc="node mesh rays")
        measurement_hits = self._intersect_pixel_batch(
            stencil.xy if stencil is not None else np.empty((0, 2), dtype=np.float64),
            desc="measurement mesh rays",
        )
        result = self._result_from_hits(instance_row, sample_xy, hits, node_xy, node_hits)
        result.update(
            surface_measurement_from_hits(
                instance_row,
                geometry,
                stencil,
                measurement_hits,
            )
        )
        return result

    def _representative_pixels(
        self,
        instance_row: dict[str, Any],
        xs: np.ndarray,
        ys: np.ndarray,
        xmin: int,
        ymin: int,
    ) -> np.ndarray:
        if self.representative_mode == "centroid":
            submask = np.zeros(
                (
                    max(1, int(np.max(ys)) + 1),
                    max(1, int(np.max(xs)) + 1),
                ),
                dtype=bool,
            )
            submask[ys, xs] = True
            snapped = snapped_component_pixel(
                submask,
                offset_x=xmin,
                offset_y=ymin,
                centroid_x=float(instance_row["centroid_x_px"]),
                centroid_y=float(instance_row["centroid_y_px"]),
            )
            if snapped is None:
                return np.empty((0, 2), dtype=np.float64)
            return np.asarray([snapped], dtype=np.float64)
        if self.representative_mode != "median":
            raise ValueError("representative_mode must be 'centroid' or 'median'")

        xs = xs.astype(np.float64) + float(xmin)
        ys = ys.astype(np.float64) + float(ymin)
        selected = sample_indices(len(xs), self.instance_sample_count)
        return np.column_stack((xs[selected], ys[selected])).astype(np.float64)

    def _result_from_hits(
        self,
        instance_row: dict[str, Any],
        sample_xy: np.ndarray,
        hits: list[dict[str, Any]],
        node_xy: np.ndarray,
        node_hits: list[dict[str, Any]],
    ) -> dict[str, Any]:
        valid_hits = [hit for hit in hits if hit.get("mesh_ray_hit")]
        if not valid_hits:
            fallback_xy = sample_xy[0] if len(sample_xy) else None
            result = self._centroid_fallback(
                instance_row,
                "mesh_ray_no_hit_in_instance_samples",
                representative_xy=fallback_xy,
            )
            result.update(
                {
                    "instance_mesh_sample_count": int(len(hits)),
                    "instance_mesh_hit_count": 0,
                    "instance_mesh_hit_ratio": 0.0,
                }
            )
            return result

        wx = np.array([float(hit["world_x_m"]) for hit in valid_hits], dtype=np.float64)
        wy = np.array([float(hit["world_y_m"]) for hit in valid_hits], dtype=np.float64)
        wz = np.array([float(hit["world_z_m"]) for hit in valid_hits], dtype=np.float64)
        t = np.array([float(hit["mesh_ray_t_m"]) for hit in valid_hits], dtype=np.float64)
        face_indices = [int(hit["mesh_face_index"]) for hit in valid_hits]
        if self.representative_mode == "centroid":
            representative_hit = valid_hits[0]
            representative_xy = sample_xy[0]
            world_x = float(representative_hit["world_x_m"])
            world_y = float(representative_hit["world_y_m"])
            world_z = float(representative_hit["world_z_m"])
            ray_t = float(representative_hit["mesh_ray_t_m"])
            mesh_face_index = int(representative_hit["mesh_face_index"])
            geo3d_source = "dji_pose_mesh_ray_centroid"
        else:
            representative_xy = np.median(sample_xy, axis=0)
            world_x = float(np.median(wx))
            world_y = float(np.median(wy))
            world_z = float(np.median(wz))
            ray_t = float(np.median(t))
            mesh_face_index = None
            geo3d_source = "dji_pose_mesh_ray_instance_median"
        result = {
            "world_x_m": round(world_x, 8),
            "world_y_m": round(world_y, 8),
            "world_z_m": round(world_z, 8),
            "geo3d_source": geo3d_source,
            "xyz_valid": True,
            "mesh_ray_hit": True,
            "mesh_face_index": mesh_face_index,
            "mesh_ray_t_m": round(ray_t, 6),
            "mesh_ray_backend": self.surface.ray_backend,
            "xyz_miss_reason": None,
            "image_pixel_x": round(float(representative_xy[0]), 4),
            "image_pixel_y": round(float(representative_xy[1]), 4),
            "centroid_snapped_to_component": bool(
                not np.allclose(
                    representative_xy,
                    [float(instance_row["centroid_x_px"]), float(instance_row["centroid_y_px"])],
                )
            ),
            "instance_mesh_sample_count": int(len(hits)),
            "instance_mesh_hit_count": int(len(valid_hits)),
            "instance_mesh_hit_ratio": round(float(len(valid_hits) / len(hits)), 6),
            "instance_mesh_unique_face_count": int(len(set(face_indices))),
            "instance_mesh_ray_t_min_m": round(float(np.min(t)), 6),
            "instance_mesh_ray_t_max_m": round(float(np.max(t)), 6),
        }
        result.update(self._nodes_to_world3d(node_xy, node_hits))
        return result

    def _nodes_to_world3d(self, node_xy: np.ndarray, hits: list[dict[str, Any]]) -> dict[str, Any]:
        if len(node_xy) == 0:
            return empty_mesh_nodes("no_instance_contour_nodes")

        image_nodes: list[list[float]] = []
        world_nodes: list[list[float | None]] = []
        valid_count = 0
        for (x, y), hit in zip(node_xy, hits):
            image_nodes.append([round(float(x), 4), round(float(y), 4)])
            if hit.get("mesh_ray_hit"):
                valid_count += 1
                world_nodes.append(
                    [
                        round(float(hit["world_x_m"]), 8),
                        round(float(hit["world_y_m"]), 8),
                        round(float(hit["world_z_m"]), 8),
                    ]
                )
            else:
                world_nodes.append([None, None, None])

        return {
            "node_count": int(len(node_xy)),
            "node_xyz_valid_count": int(valid_count),
            "node_xyz_hit_ratio": round(float(valid_count / len(node_xy)), 6),
            "nodes_image_xy_json": json.dumps(image_nodes, separators=(",", ":")),
            "nodes_world_xyz_json": json.dumps(world_nodes, separators=(",", ":")),
            "nodes_geo3d_source": "dji_pose_mesh_ray_contour_nodes",
            "nodes_miss_reason": None if valid_count > 0 else "mesh_ray_no_hit_for_nodes",
        }

    def _intersect_pixel_batch(self, xy: np.ndarray, desc: str = "mesh rays") -> list[dict[str, Any]]:
        if len(xy) == 0:
            return []
        xy = np.asarray(xy, dtype=np.float64)
        ray_start = time.perf_counter()
        directions = pixels_to_world_rays(self.intrinsics, self.pose, xy)
        self._add_timing("mesh_ray_direction_sec", time.perf_counter() - ray_start)
        origin = np.asarray(self.pose.center_xyz, dtype=np.float64)
        results: list[dict[str, Any]] = []
        batch_size = max(1, int(self.ray_batch_size))
        ranges = range(0, len(xy), batch_size)
        if self.profile:
            from tqdm import tqdm

            ranges = tqdm(ranges, desc=desc, unit="chunk", leave=False)
        for start in ranges:
            end = min(start + batch_size, len(xy))
            origins = np.repeat(origin[None, :], end - start, axis=0)
            results.extend(self.surface.intersect_rays(origins, directions[start:end]))
        return results

    def _add_timing(self, key: str, elapsed_sec: float) -> None:
        self.timings[key] = self.timings.get(key, 0.0) + float(elapsed_sec)

    def _centroid_fallback(
        self,
        instance_row: dict[str, Any],
        reason: str,
        representative_xy: np.ndarray | tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        if representative_xy is None:
            x = float(instance_row["centroid_x_px"])
            y = float(instance_row["centroid_y_px"])
        else:
            x = float(representative_xy[0])
            y = float(representative_xy[1])
        result = self.image_pixel_to_world3d(x, y)
        result["centroid_snapped_to_component"] = bool(
            not np.allclose([x, y], [float(instance_row["centroid_x_px"]), float(instance_row["centroid_y_px"])])
        )
        if not result.get("mesh_ray_hit"):
            result["xyz_miss_reason"] = reason
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "dji_pose_mesh_ray",
            "intrinsics": self.intrinsics.to_dict(),
            "pose": self.pose.to_dict(),
            "mesh": self.surface.to_dict(),
            "instance_sample_count": self.instance_sample_count,
            "node_sample_count": self.node_sample_count,
            "ray_batch_size": self.ray_batch_size,
            "representative_mode": self.representative_mode,
            "surface_measurement": {
                "enabled": True,
                "method": SURFACE_MEASUREMENT_METHOD,
                "gsd_baseline_half_span_px": self.measurement_half_span_px,
                "length_width_definition": "minimum_area_rotated_rectangle_extent",
                "area_definition": "mask_area_px_times_local_surface_jacobian_area",
                "class_reporting": {
                    "CRC": ["length_px", "length_m", "width_px", "width_m"],
                    "DLM": ["area_px", "area_m2"],
                    "SPL": ["area_px", "area_m2"],
                },
                "approximate": True,
                "lens_distortion_applied": False,
            },
            "profile_timings_sec": self.timings if self.timings else None,
            "accuracy_note": (
                "Damage instance coordinates are estimated by intersecting DJI camera rays with a triangulated mesh. "
                "By default, the reported instance coordinate is the mesh hit at the 2D instance centroid, snapped "
                "to the nearest pixel in that exact connected component when the centroid falls outside the mask. "
                "representative_mode=median uses the median of sampled mesh hits inside the damage mask. "
                "nodes_world_xyz_json stores sampled contour node coordinates as [[x,y,z], ...]. "
                "Local GSD starts at the component pixel nearest the minimum-area-rectangle center. "
                "Physical length, width, and area are rough local estimates from directional mesh-ray GSD; "
                "they are not surveyed dimensions or mesh geodesic measurements."
            ),
        }


def component_measurement_center(
    submask: np.ndarray,
    geometry: OrientedPixelGeometry | None,
    *,
    offset_x: int,
    offset_y: int,
) -> tuple[float, float] | None:
    """Choose a GSD origin that is guaranteed to be in this component.

    A minimum-area rectangle center can lie in the empty middle of a concave
    component.  Snapping that center to the closest component pixel keeps the
    local mesh stencil tied to the damage observation being quantified.
    """
    if geometry is None:
        return None
    return snapped_component_pixel(
        submask,
        offset_x=offset_x,
        offset_y=offset_y,
        centroid_x=geometry.center_x_px,
        centroid_y=geometry.center_y_px,
    )


def build_surface_measurement_stencil(
    geometry: OrientedPixelGeometry,
    *,
    center_xy: np.ndarray,
    image_width: int,
    image_height: int,
    half_span_px: float = 2.0,
) -> SurfaceMeasurementStencil | None:
    """Build bounded samples along the instance's rotated major/minor axes.

    A symmetric four-pixel baseline is preferred for mesh stability.  The
    resulting 3D chord is divided by its exact image-pixel span, so the output
    remains a one-pixel GSD.  Near an image edge, a one-sided baseline is used.
    """

    center = np.asarray(center_xy, dtype=np.float64)
    if center.shape != (2,):
        raise ValueError("center_xy must contain exactly x and y")
    if image_width < 1 or image_height < 1:
        raise ValueError("image dimensions must be positive")
    if not np.isfinite(center).all():
        raise ValueError("center_xy must be finite")
    if not 0.0 <= float(center[0]) <= float(image_width - 1):
        return None
    if not 0.0 <= float(center[1]) <= float(image_height - 1):
        return None
    half_span = float(half_span_px)
    if not np.isfinite(half_span) or half_span <= 0.0:
        raise ValueError("half_span_px must be a positive finite value")

    major = np.array([geometry.major_dx, geometry.major_dy], dtype=np.float64)
    minor = np.array([geometry.minor_dx, geometry.minor_dy], dtype=np.float64)
    length_segment = _bounded_axis_segment(
        center,
        major,
        image_width=image_width,
        image_height=image_height,
        half_span_px=half_span,
    )
    width_segment = _bounded_axis_segment(
        center,
        minor,
        image_width=image_width,
        image_height=image_height,
        half_span_px=half_span,
    )
    if length_segment is None or width_segment is None:
        return None

    length_start, length_end, length_span, length_scheme = length_segment
    width_start, width_end, width_span, width_scheme = width_segment
    return SurfaceMeasurementStencil(
        center_xy=(float(center[0]), float(center[1])),
        xy=np.vstack((length_start, length_end, width_start, width_end)),
        length_span_px=float(length_span),
        width_span_px=float(width_span),
        length_scheme=length_scheme,
        width_scheme=width_scheme,
    )


def surface_measurement_from_hits(
    instance_row: Mapping[str, Any],
    geometry: OrientedPixelGeometry | None,
    stencil: SurfaceMeasurementStencil | None,
    hits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert four local mesh hits into approximate instance dimensions."""

    if geometry is None:
        return empty_surface_measurement(instance_row, "oriented_geometry_unavailable")
    if stencil is None:
        return empty_surface_measurement(
            instance_row,
            "measurement_stencil_outside_image",
            geometry=geometry,
        )
    if len(hits) != 4:
        return empty_surface_measurement(
            instance_row,
            "measurement_ray_count_mismatch",
            geometry=geometry,
            ray_count=len(hits),
            hit_count=sum(bool(hit.get("mesh_ray_hit")) for hit in hits),
        )

    hit_count = sum(bool(hit.get("mesh_ray_hit")) for hit in hits)
    points = [_mesh_hit_point(hit) for hit in hits]
    length_vector = None
    width_vector = None
    if points[0] is not None and points[1] is not None:
        length_vector = (points[1] - points[0]) / stencil.length_span_px
    if points[2] is not None and points[3] is not None:
        width_vector = (points[3] - points[2]) / stencil.width_span_px

    gsd_length = _positive_norm_or_none(length_vector)
    gsd_width = _positive_norm_or_none(width_vector)
    gsd_area = None
    if length_vector is not None and width_vector is not None:
        candidate_area = float(np.linalg.norm(np.cross(length_vector, width_vector)))
        if np.isfinite(candidate_area) and candidate_area > 0.0:
            gsd_area = candidate_area

    is_crack = int(instance_row.get("class_id", -1)) == 1
    valid = (
        gsd_length is not None and gsd_width is not None
        if is_crack
        else gsd_area is not None
    )
    missing: list[str] = []
    if gsd_length is None:
        missing.append("length_axis_mesh_hits_invalid")
    if gsd_width is None:
        missing.append("width_axis_mesh_hits_invalid")
    if not is_crack and gsd_area is None and not missing:
        missing.append("surface_jacobian_area_invalid")

    length_px = float(geometry.length_px) if is_crack else None
    width_px = float(geometry.width_px) if is_crack else None
    length_m = length_px * gsd_length if length_px is not None and gsd_length is not None else None
    width_m = width_px * gsd_width if width_px is not None and gsd_width is not None else None
    area_m2 = (
        float(instance_row["area_px"]) * gsd_area
        if not is_crack and gsd_area is not None
        else None
    )
    schemes = {stencil.length_scheme, stencil.width_scheme}
    quality = (
        "local_central_difference"
        if valid and schemes == {"central"}
        else "local_edge_adjusted"
        if valid
        else "invalid"
    )
    return {
        "length_px": round(length_px, 6) if length_px is not None else None,
        "length_m": round(length_m, 8) if length_m is not None else None,
        "width_px": round(width_px, 6) if width_px is not None else None,
        "width_m": round(width_m, 8) if width_m is not None else None,
        "area_m2": round(area_m2, 10) if area_m2 is not None else None,
        "area_m2_source": (
            "mesh_ray_local_surface_jacobian" if area_m2 is not None else None
        ),
        "gsd_length_m_per_px": (
            round(gsd_length, 10) if gsd_length is not None else None
        ),
        "gsd_width_m_per_px": (
            round(gsd_width, 10) if gsd_width is not None else None
        ),
        "gsd_area_m2_per_px": round(gsd_area, 12) if gsd_area is not None else None,
        "measurement_angle_deg": round(float(geometry.angle_deg), 6) if is_crack else None,
        "measurement_center_x_px": round(float(stencil.center_xy[0]), 4),
        "measurement_center_y_px": round(float(stencil.center_xy[1]), 4),
        "measurement_box_xy_json": json.dumps(
            [[round(x, 4), round(y, 4)] for x, y in geometry.box_xy],
            separators=(",", ":"),
        ),
        "measurement_method": SURFACE_MEASUREMENT_METHOD,
        "measurement_quality": quality,
        "measurement_valid": bool(valid),
        "measurement_is_approximate": True,
        "measurement_ray_count": 4,
        "measurement_hit_count": int(hit_count),
        "measurement_hit_ratio": round(float(hit_count / 4.0), 6),
        "measurement_length_baseline_px": round(float(stencil.length_span_px), 6),
        "measurement_width_baseline_px": round(float(stencil.width_span_px), 6),
        "measurement_length_stencil": stencil.length_scheme,
        "measurement_width_stencil": stencil.width_scheme,
        "measurement_miss_reason": None if valid else ";".join(missing),
    }


def empty_surface_measurement(
    instance_row: Mapping[str, Any],
    reason: str,
    *,
    geometry: OrientedPixelGeometry | None = None,
    ray_count: int = 0,
    hit_count: int = 0,
) -> dict[str, Any]:
    is_crack = int(instance_row.get("class_id", -1)) == 1
    length_px = (
        float(geometry.length_px)
        if is_crack and geometry is not None
        else instance_row.get("length_px")
        if is_crack
        else None
    )
    width_px = (
        float(geometry.width_px)
        if is_crack and geometry is not None
        else instance_row.get("width_px")
        if is_crack
        else None
    )
    return {
        "length_px": round(float(length_px), 6) if length_px is not None else None,
        "length_m": None,
        "width_px": round(float(width_px), 6) if width_px is not None else None,
        "width_m": None,
        "area_m2": None,
        "area_m2_source": None,
        "gsd_length_m_per_px": None,
        "gsd_width_m_per_px": None,
        "gsd_area_m2_per_px": None,
        "measurement_angle_deg": (
            round(float(geometry.angle_deg), 6)
            if is_crack and geometry is not None
            else instance_row.get("measurement_angle_deg")
            if is_crack
            else None
        ),
        "measurement_center_x_px": None,
        "measurement_center_y_px": None,
        "measurement_box_xy_json": (
            json.dumps(
                [[round(x, 4), round(y, 4)] for x, y in geometry.box_xy],
                separators=(",", ":"),
            )
            if geometry is not None
            else "[]"
        ),
        "measurement_method": SURFACE_MEASUREMENT_METHOD,
        "measurement_quality": "invalid",
        "measurement_valid": False,
        "measurement_is_approximate": True,
        "measurement_ray_count": int(ray_count),
        "measurement_hit_count": int(hit_count),
        "measurement_hit_ratio": (
            round(float(hit_count / ray_count), 6) if ray_count > 0 else 0.0
        ),
        "measurement_length_baseline_px": None,
        "measurement_width_baseline_px": None,
        "measurement_length_stencil": None,
        "measurement_width_stencil": None,
        "measurement_miss_reason": str(reason),
    }


def _bounded_axis_segment(
    center: np.ndarray,
    direction: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    half_span_px: float,
) -> tuple[np.ndarray, np.ndarray, float, str] | None:
    axis = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    axis /= norm
    positive_limit = _image_travel_limit(
        center,
        axis,
        image_width=image_width,
        image_height=image_height,
    )
    negative_limit = _image_travel_limit(
        center,
        -axis,
        image_width=image_width,
        image_height=image_height,
    )
    half_span = float(half_span_px)
    if positive_limit >= half_span and negative_limit >= half_span:
        t_start, t_end, scheme = -half_span, half_span, "central"
    elif positive_limit >= 2.0 * half_span:
        t_start, t_end, scheme = 0.0, 2.0 * half_span, "forward"
    elif negative_limit >= 2.0 * half_span:
        t_start, t_end, scheme = -2.0 * half_span, 0.0, "backward"
    else:
        t_start = -min(negative_limit, half_span)
        t_end = min(positive_limit, half_span)
        scheme = "clipped"
    span = float(t_end - t_start)
    if not np.isfinite(span) or span <= 1e-9:
        return None
    lower = np.array([0.0, 0.0], dtype=np.float64)
    upper = np.array([float(image_width - 1), float(image_height - 1)], dtype=np.float64)
    start = np.clip(center + t_start * axis, lower, upper)
    end = np.clip(center + t_end * axis, lower, upper)
    actual_span = float(np.dot(end - start, axis))
    if not np.isfinite(actual_span) or actual_span <= 1e-9:
        return None
    return start, end, actual_span, scheme


def _image_travel_limit(
    center: np.ndarray,
    direction: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> float:
    limits: list[float] = []
    for coordinate, delta, upper in (
        (float(center[0]), float(direction[0]), float(image_width - 1)),
        (float(center[1]), float(direction[1]), float(image_height - 1)),
    ):
        if delta > 1e-12:
            limits.append((upper - coordinate) / delta)
        elif delta < -1e-12:
            limits.append(coordinate / -delta)
    return max(0.0, min(limits)) if limits else float("inf")


def _mesh_hit_point(hit: Mapping[str, Any]) -> np.ndarray | None:
    if not hit.get("mesh_ray_hit"):
        return None
    try:
        point = np.array(
            [hit["world_x_m"], hit["world_y_m"], hit["world_z_m"]],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None
    return point if np.isfinite(point).all() else None


def _positive_norm_or_none(vector: np.ndarray | None) -> float | None:
    if vector is None:
        return None
    value = float(np.linalg.norm(vector))
    return value if np.isfinite(value) and value > 0.0 else None


def build_mesh_ray_context(
    image_path: str | Path,
    mesh_path: str | Path,
    mrk_path: str | Path | None = None,
    instance_sample_count: int = 128,
    node_sample_count: int = 256,
    ray_batch_size: int = 128,
    representative_mode: str = "centroid",
    ray_backend: str = "trimesh",
    warp_device: str = "cuda:0",
    profile: bool = False,
) -> MeshRayGeo3DContext:
    surface = MeshSurfaceIndex(mesh_path, ray_backend=ray_backend, warp_device=warp_device)
    return build_mesh_ray_context_with_surface(
        image_path=image_path,
        surface=surface,
        mrk_path=mrk_path,
        instance_sample_count=instance_sample_count,
        node_sample_count=node_sample_count,
        ray_batch_size=ray_batch_size,
        representative_mode=representative_mode,
        profile=profile,
    )


def _resolve_segmentation(
    pred_mask: np.ndarray | ClassMasks | None,
    class_masks: ClassMasks | None,
) -> np.ndarray | ClassMasks:
    if pred_mask is not None and class_masks is not None:
        raise ValueError("pass either pred_mask or class_masks, not both")
    segmentation = class_masks if class_masks is not None else pred_mask
    if segmentation is None:
        raise ValueError("pred_mask or class_masks is required for instance ray mapping")
    return segmentation


def _validate_segmentation_image_shape(
    segmentation: np.ndarray | ClassMasks,
    expected_height: int,
    expected_width: int,
) -> None:
    if isinstance(segmentation, Mapping):
        shape = validate_class_masks(segmentation)
    elif isinstance(segmentation, np.ndarray) and segmentation.ndim == 2:
        shape = (int(segmentation.shape[0]), int(segmentation.shape[1]))
    else:
        raise ValueError("segmentation must be a class-mask mapping or a two-dimensional class-ID mask")
    expected = (int(expected_height), int(expected_width))
    if shape != expected:
        raise ValueError(f"segmentation shape {shape} does not match camera image shape {expected}")


def build_mesh_ray_context_with_surface(
    image_path: str | Path,
    surface: MeshSurfaceIndex,
    mrk_path: str | Path | None = None,
    instance_sample_count: int = 128,
    node_sample_count: int = 256,
    ray_batch_size: int = 128,
    representative_mode: str = "centroid",
    profile: bool = False,
) -> MeshRayGeo3DContext:
    xmp = read_dji_xmp(image_path)
    intrinsics = intrinsics_from_xmp(image_path, xmp)
    pose = pose_from_xmp(image_path, xmp, mrk_path=mrk_path)
    return MeshRayGeo3DContext(
        intrinsics=intrinsics,
        pose=pose,
        surface=surface,
        instance_sample_count=int(instance_sample_count),
        node_sample_count=int(node_sample_count),
        ray_batch_size=int(ray_batch_size),
        representative_mode=str(representative_mode),
        profile=bool(profile),
    )


def pixels_to_world_rays(intrinsics: CameraIntrinsics, pose: CameraPose, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    camera_rays = np.empty((len(xy), 3), dtype=np.float64)
    camera_rays[:, 0] = (xy[:, 0] - intrinsics.cx) / intrinsics.focal_length_px
    camera_rays[:, 1] = (xy[:, 1] - intrinsics.cy) / intrinsics.focal_length_px
    camera_rays[:, 2] = 1.0
    camera_rays /= np.linalg.norm(camera_rays, axis=1)[:, None]
    rotation = camera_to_world_matrix(pose.yaw_deg, pose.pitch_deg, pose.roll_deg)
    world_rays = camera_rays @ rotation.T
    return world_rays / np.linalg.norm(world_rays, axis=1)[:, None]


def sample_indices(length: int, max_count: int) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=np.int64)
    if max_count <= 0 or length <= max_count:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, int(max_count), dtype=np.int64))


def extract_contour_nodes(submask: np.ndarray, offset_x: int, offset_y: int, max_count: int) -> np.ndarray:
    if submask.shape[0] < 2 or submask.shape[1] < 2:
        contours = []
    else:
        contours = find_contours(submask.astype(np.uint8), 0.5)
    if contours:
        contour = max(contours, key=len)
        nodes = np.column_stack((contour[:, 1] + float(offset_x), contour[:, 0] + float(offset_y)))
    else:
        ys, xs = np.nonzero(submask)
        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.float64)
        nodes = np.column_stack((xs.astype(np.float64) + float(offset_x), ys.astype(np.float64) + float(offset_y)))

    selected = sample_indices(len(nodes), max_count)
    return nodes[selected].astype(np.float64)


def empty_mesh_nodes(reason: str) -> dict[str, Any]:
    return {
        "node_count": 0,
        "node_xyz_valid_count": 0,
        "node_xyz_hit_ratio": 0.0,
        "nodes_image_xy_json": "[]",
        "nodes_world_xyz_json": "[]",
        "nodes_geo3d_source": "dji_pose_mesh_ray_contour_nodes",
        "nodes_miss_reason": reason,
    }


def empty_mesh_hit(reason: str, x: float | None, y: float | None) -> dict[str, Any]:
    result = {
        "world_x_m": None,
        "world_y_m": None,
        "world_z_m": None,
        "geo3d_source": "dji_pose_mesh_ray",
        "xyz_valid": False,
        "mesh_ray_hit": False,
        "mesh_face_index": None,
        "mesh_ray_t_m": None,
        "xyz_miss_reason": reason,
    }
    if x is not None and y is not None:
        result["image_pixel_x"] = round(float(x), 4)
        result["image_pixel_y"] = round(float(y), 4)
    return result
