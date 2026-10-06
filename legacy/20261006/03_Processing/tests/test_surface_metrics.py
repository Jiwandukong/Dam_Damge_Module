import cv2
import numpy as np
import pytest

from uav_rgb.surface_metrics import measure_oriented_pixel_geometry


def test_empty_component_returns_none():
    assert measure_oriented_pixel_geometry(np.zeros((4, 7), dtype=bool)) is None


def test_measurement_requires_two_dimensional_mask():
    with pytest.raises(ValueError, match="two-dimensional"):
        measure_oriented_pixel_geometry(np.zeros((2, 3, 1), dtype=bool))


def test_single_pixel_has_full_one_by_one_footprint():
    geometry = measure_oriented_pixel_geometry(np.ones((1, 1), dtype=bool))

    assert geometry is not None
    assert geometry.center_x_px == pytest.approx(0.0)
    assert geometry.center_y_px == pytest.approx(0.0)
    assert geometry.length_px == pytest.approx(1.0)
    assert geometry.width_px == pytest.approx(1.0)
    assert (geometry.major_dx, geometry.major_dy) == pytest.approx((1.0, 0.0))
    assert (geometry.minor_dx, geometry.minor_dy) == pytest.approx((0.0, 1.0))
    assert geometry.angle_deg == pytest.approx(0.0)
    assert np.min(np.asarray(geometry.box_xy), axis=0) == pytest.approx((-0.5, -0.5))
    assert np.max(np.asarray(geometry.box_xy), axis=0) == pytest.approx((0.5, 0.5))


def test_global_offsets_are_applied_to_center_and_box_only():
    mask = np.ones((3, 7), dtype=np.uint8)
    local = measure_oriented_pixel_geometry(mask)
    shifted = measure_oriented_pixel_geometry(mask, offset_x=100, offset_y=250)

    assert local is not None and shifted is not None
    assert shifted.center_x_px == pytest.approx(local.center_x_px + 100.0)
    assert shifted.center_y_px == pytest.approx(local.center_y_px + 250.0)
    assert shifted.length_px == pytest.approx(local.length_px)
    assert shifted.width_px == pytest.approx(local.width_px)
    assert shifted.major_dx == pytest.approx(local.major_dx)
    assert shifted.major_dy == pytest.approx(local.major_dy)
    assert np.asarray(shifted.box_xy) == pytest.approx(
        np.asarray(local.box_xy) + np.array([100.0, 250.0])
    )


def test_rotated_rectangle_has_deterministic_major_and_minor_axes():
    mask = np.zeros((80, 80), dtype=np.uint8)
    source_box = cv2.boxPoints(((40.0, 40.0), (42.0, 12.0), 30.0))
    cv2.fillConvexPoly(mask, np.rint(source_box).astype(np.int32), 1)

    geometry = measure_oriented_pixel_geometry(mask)

    assert geometry is not None
    assert geometry.length_px > geometry.width_px
    assert geometry.angle_deg == pytest.approx(30.0, abs=2.0)
    assert geometry.major_dx > 0.0
    assert geometry.major_dy > 0.0
    assert np.hypot(geometry.major_dx, geometry.major_dy) == pytest.approx(1.0)
    assert np.hypot(geometry.minor_dx, geometry.minor_dy) == pytest.approx(1.0)
    assert (
        geometry.major_dx * geometry.minor_dx
        + geometry.major_dy * geometry.minor_dy
    ) == pytest.approx(0.0, abs=1e-6)

    repeated = measure_oriented_pixel_geometry(mask.copy())
    assert repeated == geometry


def test_square_chooses_positive_x_axis_deterministically():
    geometry = measure_oriented_pixel_geometry(np.ones((5, 5), dtype=bool))

    assert geometry is not None
    assert geometry.length_px == pytest.approx(5.0)
    assert geometry.width_px == pytest.approx(5.0)
    assert (geometry.major_dx, geometry.major_dy) == pytest.approx((1.0, 0.0))
    assert (geometry.minor_dx, geometry.minor_dy) == pytest.approx((0.0, 1.0))
    assert geometry.angle_deg == pytest.approx(0.0)
