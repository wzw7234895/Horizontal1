import copy
import math

import numpy as np
import pytest

from pixel_distance.manual_nominal import estimate_manual_nominal
from pixel_distance.cli import format_result


SIZE = [1784, 1003]
CONFIG = {"cameras": {"D03": {"horizontal_fov_deg": 60, "down_tilt_deg": 45},
                      "D04": {"horizontal_fov_deg": 80, "down_tilt_deg": 85}}}


def project(points, center, fov, tilt, yaw):
    a, t = math.radians(yaw), math.radians(tilt)
    right = np.array([math.cos(a), math.sin(a), 0])
    forward = np.array([-math.sin(a), math.cos(a), 0])
    optical = math.cos(t) * forward + np.array([0, 0, -math.sin(t)])
    down = -math.sin(t) * forward + np.array([0, 0, -math.cos(t)])
    relative = np.asarray(points) - center
    focal = SIZE[0] / (2 * math.tan(math.radians(fov) / 2))
    depth = relative @ optical
    return np.column_stack((relative @ right / depth, relative @ down / depth)) * focal + [
        (SIZE[0] - 1) / 2, (SIZE[1] - 1) / 2]


def test_manual_correspondence_recovers_two_arbitrary_3d_points():
    world = np.array([[2.5, 1.4, 1.3], [2.5, -2.0, 1.1]])
    cameras = [("D04", np.array([0.0, 0.0, 10.0]), 80, 85, 0),
               ("D03", np.array([8.0, 1.3, 10.0]), 60, 45, 90)]
    views = [{"camera_id": name, "image_size": SIZE, "center_m": center.tolist(),
              "azimuth_deg": yaw,
              "pixels": project(world, center, fov, tilt, yaw).tolist()}
             for name, center, fov, tilt, yaw in cameras]
    result = estimate_manual_nominal(views, CONFIG, max_ray_gap_m=0.01)
    assert result["status"] == "nominal_3d_candidate_not_calibrated"
    assert result["distance_m"] == pytest.approx(np.linalg.norm(world[1] - world[0]), abs=1e-6)
    np.testing.assert_allclose(result["point_xyz_m"], world, atol=1e-6)
    assert max(result["ray_gap_m"]) < 1e-6
    assert result["actual_distance_determined"] is False


def test_nominal_result_changes_with_unmeasured_cross_rig_position():
    views = [
        {"camera_id": "D04", "image_size": SIZE, "center_m": [0, 0, 10],
         "azimuth_deg": 0,
         "pixels": [[1205.4743, 412.8686], [1200.3771, 858.8686]]},
        {"camera_id": "D03", "image_size": SIZE, "center_m": [8, 1.28, 10],
         "azimuth_deg": 90,
         "pixels": [[893.75, 866.90], [402.85, 871.87]]},
    ]
    first = estimate_manual_nominal(views, CONFIG)
    alternate = copy.deepcopy(views)
    alternate[1]["center_m"] = [4, 0.64, 10]
    second = estimate_manual_nominal(alternate, CONFIG)
    assert first["distance_m"] == pytest.approx(3.46, abs=0.01)
    assert second["distance_m"] == pytest.approx(1.73, abs=0.01)
    assert first["uncertainty_status"] == "not_bounded_without_measured_relative_pose_and_error_limits"
    assert first["synchronization"]["status"] == "unknown"
    assert "真实距离尚无法确定" in format_result("manual-nominal-3d", first)


def test_half_metre_default_accepts_moderate_ray_gap_but_old_limit_rejects():
    views = [
        {"camera_id": "D04", "image_size": SIZE, "center_m": [0, 0, 10],
         "azimuth_deg": 0,
         "pixels": [[1205.4743, 412.8686], [1200.3771, 858.8686]]},
        {"camera_id": "D03", "image_size": SIZE, "center_m": [8, 1.28, 10],
         "azimuth_deg": 90,
         "pixels": [[883.75, 866.90], [402.85, 871.87]]},
    ]
    accepted = estimate_manual_nominal(views, CONFIG)
    rejected = estimate_manual_nominal(views, CONFIG, max_ray_gap_m=0.25)
    assert 0.25 < accepted["ray_gap_m"][0] < 0.5
    assert accepted["max_ray_gap_m"] == 0.5
    assert accepted["status"] == "nominal_3d_candidate_not_calibrated"
    assert rejected["status"] == "unknown"


def test_pairwise_distance_is_invariant_to_rigid_motion_of_both_cameras():
    views = [
        {"camera_id": "D04", "image_size": SIZE, "center_m": [0, 0, 10],
         "azimuth_deg": 0,
         "pixels": [[1205.4743, 412.8686], [1200.3771, 858.8686]]},
        {"camera_id": "D03", "image_size": SIZE, "center_m": [8, 1.28, 10],
         "azimuth_deg": 90,
         "pixels": [[893.75, 866.90], [402.85, 871.87]]},
    ]
    original = estimate_manual_nominal(views, CONFIG)
    moved = copy.deepcopy(views)
    angle = math.radians(25)
    for view in moved:
        x, y, z = view["center_m"]
        view["center_m"] = [math.cos(angle) * x - math.sin(angle) * y + 100,
                            math.sin(angle) * x + math.cos(angle) * y + 50, z + 3]
        view["azimuth_deg"] += 25
    measured_again = estimate_manual_nominal(moved, CONFIG)
    assert measured_again["distance_m"] == pytest.approx(original["distance_m"], abs=1e-9)
    np.testing.assert_allclose(measured_again["ray_gap_m"], original["ray_gap_m"], atol=1e-9)


def test_known_length_range_is_an_independent_consistency_check():
    views = [
        {"camera_id": "D04", "image_size": SIZE, "center_m": [0, 0, 10],
         "azimuth_deg": 0, "pixels": [[1205.4743, 412.8686], [1200.3771, 858.8686]]},
        {"camera_id": "D03", "image_size": SIZE, "center_m": [8, 1.28, 10],
         "azimuth_deg": 90, "pixels": [[893.75, 866.90], [402.85, 871.87]]},
    ]
    result = estimate_manual_nominal(views, CONFIG, known_length_range_m=(4, 5))
    assert result["reference_length_check"]["status"] == "outside_range"
    assert "范围不符" in format_result("manual-nominal-3d", result)
    with pytest.raises(ValueError, match="Known length range"):
        estimate_manual_nominal(views, CONFIG, known_length_range_m=(5, 4))


def test_rejects_noncorresponding_rays_and_mismatched_intrinsic_size():
    views = [
        {"camera_id": "D04", "image_size": SIZE, "center_m": [0, 0, 10],
         "azimuth_deg": 0, "pixels": [[1205, 413], [1200, 859]]},
        {"camera_id": "D03", "image_size": SIZE, "center_m": [8, 1.28, 10],
         "azimuth_deg": 90, "pixels": [[403, 872], [894, 867]]},
    ]
    result = estimate_manual_nominal(views, CONFIG)
    assert result["status"] == "unknown"
    views[0]["intrinsics"] = {"model": "pinhole", "image_size": [1920, 1080],
                              "K": np.eye(3).tolist(), "D": [0, 0, 0, 0, 0]}
    with pytest.raises(ValueError, match="image size"):
        estimate_manual_nominal(views, CONFIG)


def test_rejects_exposure_offset_when_real_timestamps_are_available():
    views = [
        {"camera_id": "D04", "image_size": SIZE, "center_m": [0, 0, 10],
         "azimuth_deg": 0, "exposure_time_ms": 1000,
         "pixels": [[1205, 413], [1200, 859]]},
        {"camera_id": "D03", "image_size": SIZE, "center_m": [8, 1.28, 10],
         "azimuth_deg": 90, "exposure_time_ms": 1010,
         "pixels": [[894, 867], [403, 872]]},
    ]
    result = estimate_manual_nominal(views, CONFIG, max_time_offset_ms=5)
    assert result["status"] == "unknown"
    assert result["synchronization"]["offset_ms"] == 10


def test_nonzero_roll_is_supported_for_nominal_views():
    world = np.array([[2.5, 1.4, 1.3], [2.5, -2.0, 1.1]])
    cameras = [("D04", np.array([0.0, 0.0, 10.0]), 80, 85, 0),
               ("D03", np.array([8.0, 1.3, 10.0]), 60, 45, 90)]
    views = [{"camera_id": name, "image_size": SIZE, "center_m": center.tolist(),
              "azimuth_deg": yaw,
              "pixels": project(world, center, fov, tilt, yaw).tolist()}
             for name, center, fov, tilt, yaw in cameras]
    tilted = copy.deepcopy(CONFIG)
    tilted["cameras"]["D04"]["roll_deg"] = 2.0
    result = estimate_manual_nominal(views, tilted, max_ray_gap_m=1)
    assert result["status"] == "nominal_3d_candidate_not_calibrated"
    assert result["distance_m"] != pytest.approx(np.linalg.norm(world[1] - world[0]), abs=1e-4)
