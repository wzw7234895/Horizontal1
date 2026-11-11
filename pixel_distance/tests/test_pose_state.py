import cv2
import numpy as np
import pytest

from pixel_distance.pose_state import (decoded_hash, state_from_surveyed_poses,
                                       track_camera_pose, triangulate_with_state,
                                       validate_pose_frame)


def test_shared_pose_triangulates_arbitrary_heights_and_rejects_stale_frame():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    K = np.array([[800, 0, 639.5], [0, 800, 359.5], [0, 0, 1]], dtype=float)
    intrinsic = {"model": "pinhole", "image_size": [1280, 720],
                 "K": K.tolist(), "D": [0, 0, 0, 0, 0]}
    world = np.array([[0.3, -0.4, 5.0], [1.1, 0.7, 7.0]])
    views = []
    pixels = []
    for name, t in (("D04", np.zeros(3)), ("D03", np.array([-2.0, 0, 0]))):
        pose = {"image_size": [1280, 720], "intrinsics": intrinsic,
                "rvec": [0, 0, 0], "tvec_m": t.tolist()}
        views.append({"camera_id": name, "image": image.copy(), "pose": pose})
        projected = cv2.projectPoints(world, np.zeros(3), t, K, None)[0].reshape(-1, 2)
        pixels.append(projected)
    state = state_from_surveyed_poses(views, ground_z_m=0)
    result = triangulate_with_state(pixels[0], pixels[1],
                                    state["cameras"]["D04"], state["cameras"]["D03"])
    assert result["distance_m"] == pytest.approx(np.linalg.norm(world[1] - world[0]),
                                                 abs=1e-6)
    assert max(result["reprojection_error_px"]) < 1e-6
    validate_pose_frame(image, state["cameras"]["D04"])
    image[10, 10] = 255
    with pytest.raises(ValueError, match="does not match pose state"):
        validate_pose_frame(image, state["cameras"]["D04"])


def test_temporal_ground_tracking_recovers_small_camera_motion():
    rng = np.random.default_rng(17)
    size = (720, 1280)
    K = np.array([[850, 0, 639.5], [0, 850, 359.5], [0, 0, 1]], dtype=float)
    R = np.diag([1.0, -1.0, -1.0])
    old_center = np.array([0.0, 0.0, 10.0])
    new_center = np.array([0.12, -0.06, 10.0])
    world = np.column_stack((rng.uniform(-5, 5, 600),
                             rng.uniform(-3, 3, 600), np.zeros(600)))
    images = []
    for center in (old_center, new_center):
        frame = np.full((size[0], size[1], 3), 80, dtype=np.uint8)
        pixels = cv2.projectPoints(world, cv2.Rodrigues(R)[0], -R @ center,
                                   K, None)[0].reshape(-1, 2)
        for point in pixels:
            cv2.circle(frame, tuple(np.round(point).astype(int)), 3, (220, 220, 220), -1)
        images.append(frame)
    entry = {"image_size": [1280, 720], "decoded_image_sha256": decoded_hash(images[0]),
             "K": K.tolist(), "D": [0, 0, 0, 0, 0], "R": R.tolist(),
             "t_m": (-R @ old_center).tolist(), "tracking_steps": 0}
    updated = track_camera_pose(images[0], images[1], entry, ground_z_m=0)
    new_R = np.asarray(updated["R"])
    new_t = np.asarray(updated["t_m"])
    actual_center = -new_R.T @ new_t
    np.testing.assert_allclose(actual_center, new_center, atol=0.1)
    assert updated["tracking_steps"] == 1


def test_pose_hold_has_a_bounded_lifetime():
    image = np.full((720, 1280, 3), 100, dtype=np.uint8)
    for y in range(120, 650, 25):
        for x in range(80, 1200, 25):
            cv2.circle(image, (x, y), 3, (220, 220, 220), -1)
    entry = {"image_size": [1280, 720], "decoded_image_sha256": decoded_hash(image),
             "K": [[800, 0, 639.5], [0, 800, 359.5], [0, 0, 1]],
             "D": [0, 0, 0, 0, 0], "R": np.diag([1.0, -1.0, -1.0]).tolist(),
             "t_m": [0, 0, 10], "tracking_steps": 25}
    with pytest.raises(ValueError, match="too many frames"):
        track_camera_pose(image, image, entry, ground_z_m=0)
