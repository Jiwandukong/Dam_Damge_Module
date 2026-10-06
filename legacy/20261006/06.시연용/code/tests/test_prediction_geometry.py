"""Full-photo prediction geometry tests without GT files, checkpoints or an OBJ."""
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo512.prediction_geometry import quantify_predictions
from demo512.geometry.camera_pose import CameraIntrinsics, CameraPose
from demo512.geometry.instances import summarize_class_mask_instances
from demo512.geometry.mesh_ray import MeshRayGeo3DContext, empty_mesh_hit, empty_surface_measurement


def masks_for(height=64, width=64):
    return {class_id: np.zeros((height, width), dtype=bool) for class_id in (1, 2, 3)}


class FakeContext:
    def __init__(self, width, height, *, miss=False):
        self.intrinsics = SimpleNamespace(width=width, height=height)
        self.miss = miss
        self.calls = []

    def image_region_to_world3d(self, base, pred_mask, *, instance_label_maps):
        self.calls.append((base, pred_mask, instance_label_maps))
        if self.miss:
            spatial = empty_mesh_hit("test_no_hit", None, None)
            spatial.update(empty_surface_measurement(base, "test_no_hit"))
            spatial["nodes_image_xy_json"] = "[]"
            return spatial
        return {"xyz_valid": True, "measurement_valid": True,
                "world_x_m": 243000.0, "world_y_m": 431000.0, "world_z_m": 10.0,
                "nodes_image_xy_json": "[[1,2]]", "nodes_world_xyz_json": "[[3,4,5]]",
                "native_mapper_result": "unchanged"}


def test_crossing_four_tiles_remains_one_original_coordinate_component():
    masks = masks_for(520, 1030)
    masks[2][505:518, 508:519] = True
    context = FakeContext(1030, 520)
    result = quantify_predictions(masks, context)
    assert result["component_counts"] == {"CRC": 0, "DLM": 1, "SPL": 0}
    assert len(result["instances"]) == 1
    record = result["instances"][0]
    assert record["tile_origins"] == [(0, 0), (512, 0), (0, 512), (512, 512)]
    assert record["base"]["area_px"] == 143
    assert record["base"]["bbox_xmin_px"] == 508
    assert record["base"]["bbox_ymin_px"] == 505
    assert record["base"]["centroid_x_px"] == 513.0
    assert record["base"]["centroid_y_px"] == 511.0
    assert context.calls[0][1] is masks
    assert record["spatial"]["nodes_world_xyz_json"] == "[[3,4,5]]"
    assert record["spatial"]["native_mapper_result"] == "unchanged"


def test_overlapping_classes_remain_independent_and_use_exact_component_maps():
    masks = masks_for()
    masks[1][10:15, 10:15] = True
    masks[2][10:15, 10:15] = True
    context = FakeContext(64, 64)
    result = quantify_predictions(masks, context)
    assert [row["base"]["class_name"] for row in result["instances"]] == ["CRC", "DLM"]
    assert [row["base"]["area_px"] for row in result["instances"]] == [25, 25]
    expected, summary, _ = summarize_class_mask_instances(masks, {1: "CRC", 2: "DLM", 3: "SPL"}, 1)
    assert [row["base"] for row in result["instances"]] == expected
    assert result["class_summary"] == summary
    assert len({id(call[2]) for call in context.calls}) == 1
    for base, prediction, maps in context.calls:
        assert prediction is masks
        assert set(maps) == {1, 2, 3}
        assert all(label_map.dtype == np.uint32 for label_map in maps.values())
        exact = maps[base["class_id"]] == base["instance_local_id"]
        assert np.array_equal(exact, masks[base["class_id"]])


def test_disjoint_components_are_separate_but_diagonal_pixels_are_connected():
    masks = masks_for()
    masks[3][10, 10] = masks[3][11, 11] = masks[3][40, 40] = True
    result = quantify_predictions(masks, FakeContext(64, 64))
    assert result["component_counts"] == {"CRC": 0, "DLM": 0, "SPL": 2}
    assert [row["base"]["area_px"] for row in result["instances"]] == [2, 1]
    assert [row["base"]["instance_local_id"] for row in result["instances"]] == [1, 2]


def test_nested_island_cannot_add_an_empty_bbox_tile_to_surrounding_component():
    masks = masks_for(1026, 1026)
    ring = masks[2]
    ring[1, 1:1025] = ring[1024, 1:1025] = True
    ring[1:1025, 1] = ring[1:1025, 1024] = True
    ring[700, 700] = True
    result = quantify_predictions(masks, FakeContext(1026, 1026))
    outer, inner = result["instances"]
    assert outer["base"]["area_px"] == 4092
    assert len(outer["tile_origins"]) == 8
    assert (512, 512) not in outer["tile_origins"]
    assert inner["base"]["area_px"] == 1
    assert inner["tile_origins"] == [(512, 512)]


def test_single_pixel_crc_is_retained_as_one_by_one_pixel_rectangle():
    masks = masks_for()
    masks[1][15, 22] = True
    result = quantify_predictions(masks, FakeContext(64, 64))
    base = result["instances"][0]["base"]
    assert base["area_px"] == 1
    assert base["length_px"] == pytest.approx(1.0)
    assert base["width_px"] == pytest.approx(1.0)


def test_bent_crc_keeps_rotated_rectangle_not_centerline_length_or_aperture():
    masks = masks_for()
    masks[1][10, 10:31] = True
    masks[1][10:26, 30] = True
    base = quantify_predictions(masks, FakeContext(64, 64))["instances"][0]["base"]
    assert base["area_px"] == 36
    assert base["width_px"] > 1
    assert base["length_px"] != 35.0


def test_all_empty_masks_return_zero_counts_and_never_map_geometry():
    masks = masks_for()
    context = FakeContext(64, 64)
    result = quantify_predictions(masks, context)
    assert result["instances"] == []
    assert result["component_counts"] == {"CRC": 0, "DLM": 0, "SPL": 0}
    assert [row["total_area_px"] for row in result["class_summary"]] == [0, 0, 0]
    assert context.calls == []


@pytest.mark.parametrize("missing_context", [False, True])
def test_no_camera_or_no_hit_keeps_contours_but_no_fabricated_world_quantities(missing_context):
    masks = masks_for()
    masks[1][12:25, 20:31] = True
    context = None if missing_context else FakeContext(64, 64, miss=True)
    record = quantify_predictions(masks, context)["instances"][0]
    spatial = record["spatial"]
    assert spatial["xyz_valid"] is False and spatial["measurement_valid"] is False
    for key in ("world_x_m", "world_y_m", "world_z_m", "length_m", "width_m", "area_m2",
                "gsd_length_m_per_px", "gsd_width_m_per_px", "gsd_area_m2_per_px"):
        assert spatial[key] is None
    assert spatial["length_px"] == record["base"]["length_px"]
    assert spatial["width_px"] == record["base"]["width_px"]
    nodes = json.loads(spatial["nodes_image_xy_json"])
    assert 0 < len(nodes) <= 256
    assert all(20 <= x <= 30 and 12 <= y <= 24 for x, y in nodes)
    assert json.loads(spatial["nodes_world_xyz_json"]) == [[None, None, None] for _ in nodes]
    assert spatial["node_xyz_valid_count"] == 0
    assert spatial["nodes_geo3d_source"] == "pixel_contour_only_no_world_fallback"
    reason = "camera_metadata_unavailable" if missing_context else "test_no_hit"
    assert spatial["xyz_miss_reason"] == spatial["measurement_miss_reason"] == reason


def test_intrinsics_mismatch_is_rejected_before_mapping():
    context = FakeContext(512, 512)
    with pytest.raises(ValueError, match="intrinsics dimensions"):
        quantify_predictions(masks_for(), context)
    assert context.calls == []


@pytest.mark.parametrize("case", ["uint8", "different_shapes", "extra_class", "missing_class", "zero_size"])
def test_invalid_masks_are_rejected(case):
    masks = masks_for()
    if case == "uint8":
        masks[1] = masks[1].astype(np.uint8)
    elif case == "different_shapes":
        masks[2] = np.zeros((63, 64), dtype=bool)
    elif case == "extra_class":
        masks[4] = np.zeros((64, 64), dtype=bool)
    elif case == "missing_class":
        del masks[3]
    else:
        masks = masks_for(0, 64)
    with pytest.raises((ValueError, TypeError)):
        quantify_predictions(masks, None)


def test_returned_records_do_not_retain_full_photo_instance_maps():
    refs = []

    class NonRetainingContext(FakeContext):
        def image_region_to_world3d(self, base, pred_mask, *, instance_label_maps):
            refs.extend(weakref.ref(array) for array in instance_label_maps.values())
            return {"nodes_image_xy_json": "[[1,2]]"}

    masks = masks_for()
    masks[1][10, 10] = True
    before = {key: value.copy() for key, value in masks.items()}
    result = quantify_predictions(masks, NonRetainingContext(64, 64))
    gc.collect()
    assert refs and all(ref() is None for ref in refs)
    for key in masks:
        assert np.array_equal(masks[key], before[key])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("miss", [False, True])
def test_real_production_mapper_retains_planar_gsd_and_missing_world_semantics(miss):
    class AnalyticPlane:
        ray_backend = "analytic_test_plane"

        def intersect_rays(self, origins, directions):
            hits = []
            for origin, direction in zip(origins, directions):
                if miss or direction[2] >= 0:
                    hits.append(empty_mesh_hit("test_no_hit", None, None))
                    continue
                distance = -origin[2] / direction[2]
                point = origin + distance * direction
                hits.append({"world_x_m": float(point[0]), "world_y_m": float(point[1]),
                             "world_z_m": float(point[2]), "mesh_ray_t_m": float(distance),
                             "mesh_face_index": 0, "mesh_ray_hit": True, "xyz_valid": True})
            return hits

    masks = masks_for(520, 1030)
    masks[1][50:65, 900] = True
    masks[2][505:518, 508:519] = True
    masks[3][505:518, 508:519] = True
    context = MeshRayGeo3DContext(
        intrinsics=CameraIntrinsics(1030, 520, 1000.0, 515.0, 260.0),
        pose=CameraPose("source.JPG", (243000.0, 431000.0, 10.0),
                        0.0, -90.0, 0.0, "test", "test"),
        surface=AnalyticPlane(), node_sample_count=256,
    )
    result = quantify_predictions(masks, context)
    assert result["component_counts"] == {"CRC": 1, "DLM": 1, "SPL": 1}
    for record in result["instances"]:
        base, spatial = record["base"], record["spatial"]
        assert spatial["xyz_valid"] is not miss
        assert spatial["measurement_valid"] is not miss
        if miss:
            assert spatial["world_x_m"] is None
            assert spatial["length_m"] is spatial["width_m"] is spatial["area_m2"] is None
            assert spatial["node_xyz_valid_count"] == 0
            assert len(json.loads(spatial["nodes_image_xy_json"])) > 0
        else:
            assert spatial["world_z_m"] == pytest.approx(0.0)
            assert spatial["gsd_length_m_per_px"] == pytest.approx(0.01)
            assert spatial["gsd_width_m_per_px"] == pytest.approx(0.01)
            if base["class_id"] == 1:
                assert spatial["length_m"] == pytest.approx(base["length_px"] * 0.01)
                assert spatial["width_m"] == pytest.approx(base["width_px"] * 0.01)
                assert spatial["area_m2"] is None
            else:
                assert spatial["area_m2"] == pytest.approx(143 * 0.0001)
                assert spatial["length_m"] is spatial["width_m"] is None
