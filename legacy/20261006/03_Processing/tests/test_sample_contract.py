from __future__ import annotations

from pathlib import Path

from uav_rgb.camera_pose import (
    image_index_from_name,
    pose_from_xmp,
    read_dji_xmp,
    read_mrk_positions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_all_sample_images_match_mrk_and_epsg5186_extent():
    mission = PROJECT_ROOT / "01_RawData/missions/DJI_202507021616_left03"
    images = sorted((mission / "images").glob("*.JPG"))
    mrk = mission / "navigation/DJI_202507021616_003_Timestamp.MRK"
    positions = read_mrk_positions(mrk)
    assert len(images) == 7
    assert len(positions) == 34
    for image in images:
        index = image_index_from_name(image.name)
        assert index in positions
        pose = pose_from_xmp(image, read_dji_xmp(image), mrk)
        x, y, z = pose.center_xyz
        assert 243030 < x < 243050
        assert 431240 < y < 431280
        assert 84.7 < z < 84.9
        assert pose.position_source == "timestamp_mrk"
