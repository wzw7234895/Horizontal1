import cv2
import numpy as np
import pytest

from pixel_distance.auto3d import auto_measure
from pixel_distance.pose_state import decoded_hash
from pixel_distance.stereo_patch import match_rectified_patches


def _synthetic_pair():
    rng = np.random.default_rng(88)
    reference = rng.integers(20, 235, size=(480, 640), dtype=np.uint8)
    target = cv2.warpAffine(reference, np.array([[1, 0, -40], [0, 1, 0]], dtype=float),
                            (640, 480))
    reference = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
    k = [[800, 0, 319.5], [0, 800, 239.5], [0, 0, 1]]
    entries = {}
    for name, frame, translation in (("D04", reference, [0, 0, 0]),
                                     ("D03", target, [-0.2, 0, 0])):
        entries[name] = {"K": k, "D": [0, 0, 0, 0, 0],
                         "R": np.eye(3).tolist(), "t_m": translation,
                         "image_size": [640, 480],
                         "decoded_image_sha256": decoded_hash(frame)}
    return reference, target, entries


def test_rectified_patch_recovers_arbitrary_textured_points_and_metric_distance():
    reference, target, entries = _synthetic_pair()
    selected = np.array([[250, 200], [400, 300]], dtype=float)
    matched = match_rectified_patches(reference, target, selected,
                                      entries["D04"], entries["D03"])
    np.testing.assert_allclose(matched["pixel_points"], selected - [40, 0], atol=1)
    config = {"rig_origins_m": {"gantry": [0, 0, 0], "trolley": [0, 0, 0]},
              "cameras": {"D04": {"rig": "trolley"}, "D03": {"rig": "gantry"}}}
    state = {"method": "surveyed_common_world_poses", "scale_status": "synthetic",
             "cameras": entries}
    result = auto_measure({"D04": reference, "D03": target}, "D04", selected,
                          config, matcher="patch", pose_state=state)
    assert result["distance_m"] == pytest.approx(
        np.linalg.norm(selected[1] - selected[0]) * 4 / 800, abs=0.02)


def test_rectified_patch_rejects_textureless_selected_point():
    reference, target, entries = _synthetic_pair()
    reference[350:420, 200:280] = 128
    selected = np.array([[250, 200], [240, 380]], dtype=float)
    matched = match_rectified_patches(reference, target, selected,
                                      entries["D04"], entries["D03"],
                                      allow_partial=True)
    assert matched["status"] == "partial_coverage"
    assert matched["pixel_points"][0] is not None
    assert matched["pixel_points"][1] is None
    assert "纹理不足" in matched["point_checks"][1]["reason"]


def test_rectified_patch_recovers_points_at_different_depths():
    reference, _, entries = _synthetic_pair()
    target = np.zeros_like(reference)
    target[:240] = cv2.warpAffine(reference[:240],
                                   np.array([[1, 0, -40], [0, 1, 0]], dtype=float),
                                   (640, 240))
    target[240:] = cv2.warpAffine(reference[240:],
                                   np.array([[1, 0, -20], [0, 1, 0]], dtype=float),
                                   (640, 240))
    selected = np.array([[250, 160], [400, 320]], dtype=float)
    matched = match_rectified_patches(reference, target, selected,
                                      entries["D04"], entries["D03"])
    np.testing.assert_allclose(matched["pixel_points"], [[210, 160], [380, 320]], atol=1)
    k = np.asarray(entries["D04"]["K"])
    world = np.column_stack(([(point[0] - k[0, 2]) * depth / k[0, 0]
                              for point, depth in zip(selected, (4, 8))],
                             [(point[1] - k[1, 2]) * depth / k[1, 1]
                              for point, depth in zip(selected, (4, 8))],
                             [4, 8]))
    config = {"rig_origins_m": {"gantry": [0, 0, 0], "trolley": [0, 0, 0]},
              "cameras": {"D04": {"rig": "trolley"}, "D03": {"rig": "gantry"}}}
    entries["D03"]["decoded_image_sha256"] = decoded_hash(target)
    state = {"method": "surveyed_common_world_poses", "scale_status": "synthetic",
             "cameras": entries}
    result = auto_measure({"D04": reference, "D03": target}, "D04", selected,
                          config, matcher="patch", pose_state=state)
    assert result["distance_m"] == pytest.approx(np.linalg.norm(world[1] - world[0]),
                                                  abs=0.02)


def test_rectified_patch_requires_pose_state():
    reference, target, _ = _synthetic_pair()
    config = {"cameras": {"D04": {}, "D03": {}}}
    with pytest.raises(ValueError, match="共享位姿"):
        auto_measure({"D04": reference, "D03": target}, "D04",
                     [[250, 200], [400, 300]], config, matcher="patch")


def test_rectified_patch_reports_each_failed_point():
    reference, target, entries = _synthetic_pair()
    reference[180:230, 230:280] = 128
    reference[280:330, 380:430] = 128
    result = match_rectified_patches(reference, target,
                                     [[250, 200], [400, 300]],
                                     entries["D04"], entries["D03"],
                                     allow_partial=True)
    assert result["status"] == "unknown"
    assert result["pixel_points"] == [None, None]
    assert all("纹理不足" in item["reason"] for item in result["point_checks"])
