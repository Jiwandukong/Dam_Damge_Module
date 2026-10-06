import numpy as np
import json
import pytest

from uav_rgb.camera_pose import CameraIntrinsics, CameraPose
from uav_rgb.instances import summarize_class_mask_instances
from uav_rgb.mesh_ray import (
    MeshRayGeo3DContext,
    MeshSurfaceIndex,
    build_surface_measurement_stencil,
    component_measurement_center,
    extract_contour_nodes,
    pixels_to_world_rays,
    sample_indices,
    surface_measurement_from_hits,
)
from uav_rgb.surface_metrics import measure_oriented_pixel_geometry


def test_unreferenced_glb_is_rejected_before_loading(tmp_path):
    glb = tmp_path / "DaecheongDam_v0829_grid.glb"
    glb.write_bytes(b"not needed because suffix is rejected first")
    with pytest.raises(ValueError, match="no transform into the EPSG:5186"):
        MeshSurfaceIndex(glb)


def test_sample_indices_limits_deterministically():
    assert sample_indices(3, 10).tolist() == [0, 1, 2]
    assert sample_indices(10, 3).tolist() == [0, 4, 9]


def test_mesh_ray_intersects_triangle(tmp_path):
    obj = tmp_path / "surface.obj"
    obj.write_text(
        "\n".join(
            [
                "v -2 10 -2",
                "v 2 10 -2",
                "v 0 10 2",
                "f 1 2 3",
            ]
        ),
        encoding="utf-8",
    )

    surface = MeshSurfaceIndex(obj)
    intrinsics = CameraIntrinsics(width=11, height=11, focal_length_px=10.0, cx=5.0, cy=5.0)
    pose = CameraPose(
        source_path="image.jpg",
        center_xyz=(0.0, 0.0, 0.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        position_source="test",
        orientation_source="test",
    )
    context = MeshRayGeo3DContext(intrinsics=intrinsics, pose=pose, surface=surface, instance_sample_count=2)

    hit = context.image_pixel_to_world3d(5, 5)
    assert hit["mesh_ray_hit"] is True
    assert np.isclose(hit["world_x_m"], 0.0)
    assert np.isclose(hit["world_y_m"], 10.0)
    assert np.isclose(hit["world_z_m"], 0.0)

    pred_mask = np.zeros((11, 11), dtype=np.uint8)
    pred_mask[5, 5] = 1
    row = {
        "class_id": 1,
        "centroid_x_px": 5.0,
        "centroid_y_px": 5.0,
        "bbox_xmin_px": 5,
        "bbox_ymin_px": 5,
        "bbox_xmax_px": 6,
        "bbox_ymax_px": 6,
    }
    region = context.image_region_to_world3d(row, pred_mask)
    assert region["xyz_valid"] is True
    assert region["geo3d_source"] == "dji_pose_mesh_ray_centroid"
    assert region["mesh_ray_backend"] == "trimesh"
    assert region["instance_mesh_sample_count"] == 1
    assert region["instance_mesh_hit_count"] == 1
    assert region["node_count"] >= 1
    assert region["node_xyz_valid_count"] >= 1
    assert json.loads(region["nodes_world_xyz_json"])[0][1] == 10.0

    batch_region = context.image_regions_to_world3d([row], pred_mask)[0]
    assert batch_region["xyz_valid"] is True
    assert batch_region["world_y_m"] == region["world_y_m"]
    assert batch_region["node_xyz_valid_count"] == region["node_xyz_valid_count"]

    median_context = MeshRayGeo3DContext(
        intrinsics=intrinsics,
        pose=pose,
        surface=surface,
        instance_sample_count=2,
        representative_mode="median",
    )
    median_region = median_context.image_region_to_world3d(row, pred_mask)
    assert median_region["geo3d_source"] == "dji_pose_mesh_ray_instance_median"


def test_warp_mesh_ray_backend_intersects_triangle(tmp_path):
    pytest.importorskip("warp")

    obj = tmp_path / "surface.obj"
    obj.write_text(
        "\n".join(
            [
                "v 100000 -2 200000",
                "v 100004 -2 200000",
                "v 100002 2 200010",
                "f 1 2 3",
            ]
        ),
        encoding="utf-8",
    )

    surface = MeshSurfaceIndex(obj, ray_backend="warp", warp_device="cpu")
    hits = surface.intersect_rays(
        origins=np.array([[100002.0, 0.0, 199990.0]], dtype=np.float64),
        directions=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
    )
    assert surface.ray_backend == "warp"
    assert hits[0]["mesh_ray_backend"] == "warp"
    assert hits[0]["mesh_ray_hit"] is True
    assert np.isclose(hits[0]["world_x_m"], 100002.0, atol=1e-3)
    assert np.isclose(hits[0]["world_z_m"], 200005.0, atol=1e-3)


def test_extract_contour_nodes_returns_limited_xy_nodes():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    nodes = extract_contour_nodes(mask, offset_x=10, offset_y=20, max_count=5)
    assert nodes.shape == (5, 2)
    assert np.all(nodes[:, 0] >= 10)
    assert np.all(nodes[:, 1] >= 20)


def test_pixels_to_world_rays_vectorizes_camera_model():
    intrinsics = CameraIntrinsics(width=11, height=11, focal_length_px=10.0, cx=5.0, cy=5.0)
    pose = CameraPose(
        source_path="image.jpg",
        center_xyz=(0.0, 0.0, 0.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
        position_source="test",
        orientation_source="test",
    )
    rays = pixels_to_world_rays(intrinsics, pose, np.array([[5.0, 5.0], [6.0, 5.0]]))
    assert np.allclose(rays[0], [0.0, 1.0, 0.0])
    assert rays.shape == (2, 3)


def test_mesh_ray_estimates_crc_dimensions_and_damage_area_on_frontal_plane(tmp_path):
    obj = tmp_path / "wide-surface.obj"
    obj.write_text(
        "\n".join(
            [
                "v -50 10 -50",
                "v 50 10 -50",
                "v 50 10 50",
                "v -50 10 50",
                "f 1 2 3",
                "f 1 3 4",
            ]
        ),
        encoding="utf-8",
    )
    surface = MeshSurfaceIndex(obj)
    context = MeshRayGeo3DContext(
        intrinsics=CameraIntrinsics(
            width=21,
            height=21,
            focal_length_px=10.0,
            cx=10.0,
            cy=10.0,
        ),
        pose=CameraPose(
            source_path="image.jpg",
            center_xyz=(0.0, 0.0, 0.0),
            yaw_deg=0.0,
            pitch_deg=0.0,
            roll_deg=0.0,
            position_source="test",
            orientation_source="test",
        ),
        surface=surface,
        measurement_half_span_px=2.0,
    )

    rectangle = np.zeros((21, 21), dtype=bool)
    rectangle[8:13, 5:16] = True  # 11 x 5 complete pixel footprints
    class_masks = {
        1: rectangle,
        2: rectangle.copy(),
        3: np.zeros_like(rectangle),
    }
    rows, _, label_maps = summarize_class_mask_instances(
        class_masks,
        {0: "BG", 1: "CRC", 2: "DLM", 3: "SPL"},
    )
    results = context.image_regions_to_world3d(
        rows,
        class_masks=class_masks,
        instance_label_maps=label_maps,
    )
    crc = results[0]
    dlm = results[1]

    assert crc["measurement_valid"] is True
    assert crc["length_px"] == pytest.approx(11.0)
    assert crc["width_px"] == pytest.approx(5.0)
    assert crc["gsd_length_m_per_px"] == pytest.approx(1.0)
    assert crc["gsd_width_m_per_px"] == pytest.approx(1.0)
    assert crc["length_m"] == pytest.approx(11.0)
    assert crc["width_m"] == pytest.approx(5.0)
    assert crc["area_m2"] is None
    assert crc["measurement_hit_count"] == 4

    assert dlm["measurement_valid"] is True
    assert dlm["length_m"] is None
    assert dlm["width_m"] is None
    assert dlm["gsd_area_m2_per_px"] == pytest.approx(1.0)
    assert dlm["area_m2"] == pytest.approx(55.0)
    assert dlm["area_m2_source"] == "mesh_ray_local_surface_jacobian"


def test_surface_measurement_uses_cross_product_and_reports_missing_axis():
    geometry = measure_oriented_pixel_geometry(np.ones((2, 6), dtype=bool))
    assert geometry is not None
    stencil = build_surface_measurement_stencil(
        geometry,
        center_xy=np.array([10.0, 10.0]),
        image_width=21,
        image_height=21,
        half_span_px=2.0,
    )
    assert stencil is not None

    def hit(x, y, z):
        return {
            "mesh_ray_hit": True,
            "world_x_m": x,
            "world_y_m": y,
            "world_z_m": z,
        }

    # Four-pixel image spans produce local vectors [2,0,0] and [0,1,1].
    hits = [hit(0, 0, 0), hit(8, 0, 0), hit(0, 0, 0), hit(0, 4, 4)]
    measured = surface_measurement_from_hits(
        {"class_id": 2, "class_name": "DLM", "area_px": 10},
        geometry,
        stencil,
        hits,
    )
    assert measured["gsd_length_m_per_px"] == pytest.approx(2.0)
    assert measured["gsd_width_m_per_px"] == pytest.approx(np.sqrt(2.0))
    assert measured["gsd_area_m2_per_px"] == pytest.approx(2.0 * np.sqrt(2.0))
    assert measured["area_m2"] == pytest.approx(20.0 * np.sqrt(2.0))

    missing = surface_measurement_from_hits(
        {"class_id": 1, "class_name": "CRC", "area_px": 10},
        geometry,
        stencil,
        [hits[0], hits[1], {"mesh_ray_hit": False}, hits[3]],
    )
    assert missing["measurement_valid"] is False
    assert missing["length_m"] is not None
    assert missing["width_m"] is None
    assert missing["measurement_hit_ratio"] == pytest.approx(0.75)
    assert "width_axis" in missing["measurement_miss_reason"]


def test_surface_measurement_stencil_uses_one_sided_edge_baseline():
    geometry = measure_oriented_pixel_geometry(np.ones((1, 3), dtype=bool))
    assert geometry is not None
    stencil = build_surface_measurement_stencil(
        geometry,
        center_xy=np.array([0.0, 0.0]),
        image_width=10,
        image_height=10,
        half_span_px=2.0,
    )
    assert stencil is not None
    assert stencil.length_scheme == "forward"
    assert stencil.width_scheme == "forward"
    assert np.all(stencil.xy >= 0.0)
    assert np.all(stencil.xy[:, 0] <= 9.0)
    assert np.all(stencil.xy[:, 1] <= 9.0)


def test_concave_component_measurement_center_is_snapped_inside_component():
    mask = np.zeros((9, 9), dtype=bool)
    mask[1:8, 1:3] = True
    mask[6:8, 1:8] = True
    mask[1:8, 6:8] = True
    geometry = measure_oriented_pixel_geometry(mask, offset_x=20, offset_y=30)
    assert geometry is not None
    center = component_measurement_center(
        mask,
        geometry,
        offset_x=20,
        offset_y=30,
    )
    assert center is not None
    x, y = center
    assert mask[int(y - 30), int(x - 20)]


def test_corner_diagonal_component_has_feasible_geometry_center_stencil():
    mask = np.eye(2, dtype=bool)
    geometry = measure_oriented_pixel_geometry(mask)
    assert geometry is not None
    stencil = build_surface_measurement_stencil(
        geometry,
        center_xy=np.array([geometry.center_x_px, geometry.center_y_px]),
        image_width=10,
        image_height=10,
        half_span_px=2.0,
    )
    assert stencil is not None
    assert stencil.length_span_px > 0.0
    assert stencil.width_span_px > 0.0
    assert np.all(stencil.xy >= 0.0)
