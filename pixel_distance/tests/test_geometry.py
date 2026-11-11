import math
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from pixel_distance.geometry import fit_plane, fit_pose, measure_3d, measure_ground


SIZE = (1920, 1080)
K = np.array([[1050.0, 0, 960.0], [0, 1060.0, 540.0], [0, 0, 1]])
D = np.array([0.03, -0.01, 0.0005, -0.0004, 0.0])
INTRINSICS = {"model": "pinhole", "image_size": list(SIZE), "K": K.tolist(), "D": D.tolist()}


def camera(center):
    # A downward-looking camera; changing center creates a stereo baseline.
    rotation = np.diag([1.0, -1.0, -1.0])
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = -rotation @ np.asarray(center, dtype=float)
    return rvec.reshape(3), tvec


def pixels(world, center):
    rvec, tvec = camera(center)
    projected, _ = cv2.projectPoints(np.asarray(world, float).reshape(-1, 3),
                                     rvec, tvec, K, D)
    return projected.reshape(-1, 2)


def surveyed_ground():
    ground = np.array([(x, y, 0) for y in (-2.0, 0.0, 2.0)
                       for x in (-3.0, -1.0, 1.0, 3.0)])
    uv = pixels(ground, (0, 0, 10))
    anchors = []
    for i, (xyz, pixel) in enumerate(zip(ground, uv)):
        anchors.append({"id": str(i), "role": "check" if i in (1, 10) else "control",
                        "pixel": pixel.tolist(), "xy_m": xyz[:2].tolist()})
    return anchors


def test_ground_distance_with_distortion_and_interleaved_check_points():
    calibration = fit_plane(SIZE, surveyed_ground(), INTRINSICS, 0.01, 0.01)
    a, b = np.array([[-2, -1, 0], [2, 1, 0]])
    uv = pixels([a, b], (0, 0, 10))
    result = measure_ground(calibration, SIZE, uv[0], uv[1])
    assert result["distance_m"] == pytest.approx(math.sqrt(20), abs=1e-4)
    assert calibration["diagnostics"]["check_count"] == 2


def test_ground_rejects_elevated_interpretation_and_extrapolation():
    calibration = fit_plane(SIZE, surveyed_ground(), INTRINSICS, 0.01, 0.01)
    far = pixels([[8, 0, 0]], (0, 0, 10))[0]
    inside = pixels([[0, 0, 0]], (0, 0, 10))[0]
    with pytest.raises(ValueError, match="outside the ground control area"):
        measure_ground(calibration, SIZE, inside, far)
    with pytest.raises(ValueError, match="dimensions"):
        measure_ground(calibration, (1280, 720), inside, inside)


def test_bad_independent_check_is_rejected():
    anchors = surveyed_ground()
    anchors[1]["xy_m"][0] += 0.5
    with pytest.raises(ValueError, match="Independent check error"):
        fit_plane(SIZE, anchors, INTRINSICS, 0.01, 0.05)


def pose_anchors(center):
    xyz = np.array([(x, y, z) for z in (0, 1.0, 2.0)
                    for y in (-1.5, 1.5) for x in (-2.5, 2.5)])
    uv = pixels(xyz, center)
    return [{"id": str(i), "role": "check" if i in (2, 9) else "control",
             "xyz_m": point.tolist(), "pixel": pixel.tolist()}
            for i, (point, pixel) in enumerate(zip(xyz, uv))]


def test_3d_distance_from_two_current_poses():
    poses = [fit_pose(SIZE, pose_anchors(center), INTRINSICS, 0.1)
             for center in ((-2, 0, 10), (2, 0, 10))]
    world = np.array([[-0.5, -0.5, 2], [1.0, 1.0, 3.0]])
    views = []
    for i, center in enumerate(((-2, 0, 10), (2, 0, 10))):
        views.append({"timestamp_ms": 1000 + i, "image_size": list(SIZE),
                      "pose": poses[i], "pixels": pixels(world, center).tolist()})
    result = measure_3d(views, 2, 0.2)
    assert result["distance_m"] == pytest.approx(np.linalg.norm(world[1] - world[0]), abs=1e-4)
    assert np.asarray(result["world_xyz_m"]) == pytest.approx(world, abs=1e-4)
    with pytest.raises(ValueError, match="time offset"):
        measure_3d(views, 0.5, 0.2)
    views[1]["pixels"][0][1] += 20
    with pytest.raises(ValueError, match="reprojection error"):
        measure_3d(views, 2, 0.2)


def test_rigid_stereo_distance_survives_common_camera_translation():
    centers = ((-2.0, 0.0, 10.0), (2.0, 0.0, 10.0))
    original_poses = [fit_pose(SIZE, pose_anchors(center), INTRINSICS, 0.1)
                      for center in centers]
    moved = np.array([3.0, 1.0, 0.0])
    for world in (np.array([[-0.5, -0.5, 2.0], [1.0, 1.0, 3.0]]),
                  np.array([[0.5, 0.2, 1.0], [2.0, -0.4, 4.0]])):
        views = [{"timestamp_ms": 1000, "image_size": list(SIZE),
                  "pose": pose, "pixels": pixels(world, np.asarray(center) + moved).tolist()}
                 for pose, center in zip(original_poses, centers)]
        result = measure_3d(views, 1, 0.2)
        assert result["distance_m"] == pytest.approx(np.linalg.norm(world[1] - world[0]), abs=1e-4)
        np.testing.assert_allclose(result["world_xyz_m"], world - moved, atol=1e-4)


def test_cli_ground_workflow(tmp_path):
    image = tmp_path / "frame.png"
    cv2.imwrite(str(image), np.zeros((SIZE[1], SIZE[0], 3), np.uint8))
    points = tmp_path / "points.json"
    points.write_text(json.dumps({"points": surveyed_ground()}), encoding="utf-8")
    intrinsics = tmp_path / "intrinsics.json"
    intrinsics.write_text(json.dumps(INTRINSICS), encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    command = [sys.executable, "-m", "pixel_distance"]
    fit = subprocess.run(command + ["calibrate-plane", "--image", str(image),
                          "--points", str(points), "--intrinsics", str(intrinsics),
                          "--max-check-error-m", "0.01", "--out", str(calibration)],
                         cwd=root, capture_output=True, text=True)
    assert fit.returncode == 0, fit.stderr
    uv = pixels([[-2, -1, 0], [2, 1, 0]], (0, 0, 10))
    measure = subprocess.run(command + ["measure", "--image", str(image),
                              "--calibration", str(calibration), "--surface", "ground",
                              "--point-a", f"{uv[0,0]},{uv[0,1]}",
                              "--point-b", f"{uv[1,0]},{uv[1,1]}",
                              "--annotated", str(tmp_path / "result.png"), "--json"],
                             cwd=root, capture_output=True, text=True)
    assert measure.returncode == 0, measure.stderr
    assert json.loads(measure.stdout)["distance_m"] == pytest.approx(math.sqrt(20), abs=1e-4)
    assert (tmp_path / "result.png").exists()
