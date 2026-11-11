import cv2
import numpy as np
import pytest

from pixel_distance.ground_segmentation import _point_decision, save_ground_overlay


def test_point_decision_requires_local_support_and_rejects_image_edge():
    floor = np.zeros((100, 120), dtype=np.float32)
    floor[25:75, 25:95] = 0.95
    labels = np.zeros((100, 120), dtype=np.int32)
    labels[25:75, 25:95] = 3
    names = {0: "wall", 3: "floor"}
    assert _point_decision(floor, labels, names, np.array([50, 50]), 0.75)[
        "status"] == "ground_candidate_unverified"
    assert _point_decision(floor, labels, names, np.array([25, 50]), 0.75)[
        "status"] == "unknown"
    assert _point_decision(floor, labels, names, np.array([0, 50]), 0.75)[
        "status"] == "unknown"
    with pytest.raises(ValueError, match="超出"):
        _point_decision(floor, labels, names, np.array([120, 50]), 0.75)


def test_overlay_is_written_at_original_resolution(tmp_path):
    image = np.full((50, 80, 3), 100, dtype=np.uint8)
    mask = np.zeros((50, 80), dtype=np.uint8)
    mask[10:30, 20:50] = 255
    path = tmp_path / "ground.png"
    save_ground_overlay(image, mask, path, [[25, 20]])
    result = cv2.imread(str(path))
    assert result.shape == image.shape
    assert result[20, 30, 1] > result[20, 30, 0]
