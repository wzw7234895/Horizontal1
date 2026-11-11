import cv2
import numpy as np
import pytest

from pixel_distance.geometry import load_json
from pixel_distance.group_height import compare_group_points
from pixel_distance.pose_state import decoded_hash


def _scenario(trolley_shift=0.0):
    image = np.zeros((720, 1280, 3), np.uint8)
    images = {name: image for name in ("D01", "D03", "D02", "D04")}
    K = np.array([[800, 0, 639.5], [0, 800, 359.5], [0, 0, 1]], float)
    centers = {"D01": [0, 0, 0], "D03": [2, 0, 0],
               "D02": [trolley_shift, 0, 0], "D04": [trolley_shift + 2, 0, 0]}
    state = {"method": "surveyed_independent_group_poses", "scale_status": "synthetic",
             "cameras": {name: {"K": K.tolist(), "D": [0]*5,
                                "R": np.eye(3).tolist(), "t_m": (-np.asarray(center)).tolist(),
                                "image_size": [1280, 720], "decoded_image_sha256": decoded_hash(image)}
                         for name, center in centers.items()}}
    points = {"A": np.array([1, 0.2, 8]), "B": np.array([1+trolley_shift, 0.3, 7])}
    observations = {label: [{"camera_id": name, "pixel": cv2.projectPoints(
        point.reshape(1, 3), np.zeros(3), -np.asarray(centers[name], float), K, None)[0]
        .reshape(2).tolist()} for name in names]
        for label, point, names in (("A", points["A"], ("D01", "D03")),
                                    ("B", points["B"], ("D02", "D04")))}
    return observations, images, state


def test_cross_group_height_ignores_horizontal_trolley_translation():
    config = load_json("pixel_distance/camera_presets.json")
    alignment = {"status": "validated", "z_offsets_m": {"gantry": 0, "trolley": 0},
                 "point_height_error_bound_m": {"A": 0.04, "B": 0.05},
                 "alignment_error_bound_m": 0.02}
    for shift in (0, 4):
        observations, images, state = _scenario(shift)
        result = compare_group_points(observations, images, state, config, alignment)
        assert result["distance_m"] is None
        assert result["height_difference_m"] == pytest.approx(-1, abs=1e-6)
        assert result["height_difference_interval_m"] == pytest.approx([-1.11, -0.89])


def test_cross_group_height_rejects_missing_alignment_and_cross_pair():
    observations, images, state = _scenario()
    config = load_json("pixel_distance/camera_presets.json")
    with pytest.raises(ValueError, match="竖直高度基准"):
        compare_group_points(observations, images, state, config)
    observations["A"][1]["camera_id"] = "D04"
    with pytest.raises(ValueError, match="同一安装组"):
        compare_group_points(observations, images, state, config)


def test_height_state_reuses_offline_group_geometry_on_new_images():
    from pixel_distance.pose_state import state_from_surveyed_group_poses
    config = load_json("pixel_distance/camera_presets.json")
    source = np.zeros((720, 1280, 3), np.uint8)
    current = np.full_like(source, 80)
    K = [[800, 0, 639.5], [0, 800, 359.5], [0, 0, 1]]
    views = [{"camera_id": name, "image": current,
              "pose": {"image_size": [1280, 720],
                       "intrinsics": {"model": "pinhole", "K": K, "D": [0]*5,
                                      "image_size": [1280, 720]},
                       "rvec": [0, 0, 0], "tvec_m": [-i, 0, 0]}}
             for i, name in enumerate(("D01", "D03", "D02", "D04"))]
    state = state_from_surveyed_group_poses(views, config)
    assert state["method"] == "surveyed_independent_group_poses"
    assert state["cameras"]["D01"]["decoded_image_sha256"] == decoded_hash(current)
    assert state["cameras"]["D01"]["decoded_image_sha256"] != decoded_hash(source)
