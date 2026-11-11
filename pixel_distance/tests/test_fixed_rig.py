import cv2
import numpy as np
import pytest

from pixel_distance.auto3d import auto_measure
from pixel_distance.geometry import load_json
from pixel_distance.pose_state import (state_from_nominal_fixed_rig,
                                       triangulate_with_state)


def test_fixed_four_camera_geometry_reuses_relative_pose_across_frames(monkeypatch):
    config = load_json("pixel_distance/camera_presets.json")
    config["all_cameras_rigid"] = True  # Explicit legacy diagnostic assumption.
    first = {name: np.zeros((1003, 1784, 3), dtype=np.uint8)
             for name in ("D04", "D03")}
    second = {name: np.full((1003, 1784, 3), 80, dtype=np.uint8)
              for name in first}
    state1 = state_from_nominal_fixed_rig(first, config)
    state2 = state_from_nominal_fixed_rig(second, config)
    assert state1["method"] == "nominal_fixed_four_camera_rig"
    assert state1["cameras"]["D04"]["decoded_image_sha256"] != state2["cameras"]["D04"]["decoded_image_sha256"]
    for name in first:
        for field in ("K", "D", "R", "t_m"):
            np.testing.assert_allclose(state1["cameras"][name][field],
                                       state2["cameras"][name][field])

    world = np.array([[2.5, 1.4, 1.3], [2.5, -2.0, 1.1]], dtype=float)
    projected = {}
    for name, entry in state1["cameras"].items():
        R = np.asarray(entry["R"])
        projected[name] = cv2.projectPoints(
            world, cv2.Rodrigues(R)[0], np.asarray(entry["t_m"]),
            np.asarray(entry["K"]), np.asarray(entry["D"]))[0].reshape(-1, 2)
        assert np.all(projected[name] >= 0)
        assert np.all(projected[name][:, 0] < 1784)
        assert np.all(projected[name][:, 1] < 1003)

    def known_correspondences(*_args, **_kwargs):
        assert _kwargs["candidate_epipolar_px"] == 20.0
        return {"status": "matched_candidate", "matcher": "synthetic",
                "pixel_points": projected["D03"].tolist(),
                "point_checks": [{"status": "matched_candidate"} for _ in range(2)]}

    monkeypatch.setattr("pixel_distance.auto3d.match_view_with_pose", known_correspondences)
    for images in (first, second):
        result = auto_measure(images, "D04", projected["D04"], config)
        assert result["distance_m"] == pytest.approx(
            np.linalg.norm(world[1] - world[0]), abs=1e-6)
        assert result["pose_state_method"] == "nominal_fixed_four_camera_rig"
        assert result["views"]["D03"]["status"] == "matched_candidate"
    direct = triangulate_with_state(projected["D04"], projected["D03"],
                                    state1["cameras"]["D04"], state1["cameras"]["D03"])
    assert direct["distance_m"] == pytest.approx(np.linalg.norm(world[1] - world[0]), abs=1e-6)


def test_fixed_geometry_requires_explicit_rigid_assumption():
    config = load_json("pixel_distance/camera_presets.json")
    config["all_cameras_rigid"] = False
    with pytest.raises(ValueError, match="尚未"):
        state_from_nominal_fixed_rig({"D04": np.zeros((100, 100, 3), dtype=np.uint8)}, config)


def test_default_group_geometry_keeps_cross_group_visual_candidates(monkeypatch):
    config = load_json("pixel_distance/camera_presets.json")
    assert config["all_cameras_rigid"] is False
    image = np.zeros((1003, 1784, 3), dtype=np.uint8)
    from pixel_distance.pose_state import state_from_nominal_group
    state = state_from_nominal_group({"D04": image, "D02": image}, config)
    world = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 2.0]])
    pixels = {}
    for name, entry in state["cameras"].items():
        pixels[name] = cv2.projectPoints(world, cv2.Rodrigues(np.asarray(entry["R"]))[0],
                                         np.asarray(entry["t_m"]), np.asarray(entry["K"]),
                                         np.asarray(entry["D"]))[0].reshape(-1, 2)
    monkeypatch.setattr("pixel_distance.auto3d.match_view_with_pose",
                        lambda *_a, **_k: {"status": "matched_candidate", "matcher": "synthetic",
                                           "pixel_points": pixels["D02"].tolist(),
                                           "point_checks": [{"status": "matched_candidate"}] * 2})
    visual_calls = []
    def cross_visual(*_a, **_k):
        visual_calls.append(True)
        return {"status": "partial_coverage", "matcher": "synthetic",
                "pixel_points": [None, None],
                "point_checks": [{"status": "unknown"}] * 2}
    monkeypatch.setattr("pixel_distance.auto3d.match_view", cross_visual)
    result = auto_measure({"D04": image, "D02": image, "D03": image},
                          "D04", pixels["D04"], config)
    assert result["distance_m"] == pytest.approx(np.linalg.norm(world[1]-world[0]), abs=1e-6)
    assert result["views"]["D03"]["role"] == "cross_group_visual_candidate"
    assert len(visual_calls) == 1
    assert result["metric_group"] == "trolley"
    missing = auto_measure({"D04": image, "D03": image}, "D04", pixels["D04"], config)
    assert missing["status"] == "unknown"
    assert missing["distance_m"] is None
    assert len(visual_calls) == 2


def test_cross_group_visual_transfer_can_measure_inside_other_group(monkeypatch):
    from pixel_distance.pose_state import state_from_nominal_group
    config = load_json("pixel_distance/camera_presets.json")
    image = np.zeros((1003, 1784, 3), dtype=np.uint8)
    images = {name: image for name in ("D04", "D01", "D03")}
    state = state_from_nominal_group({name: image for name in ("D01", "D03")}, config)
    world = np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 2.0]])
    projected = {name: cv2.projectPoints(
        world, cv2.Rodrigues(np.asarray(entry["R"]))[0],
        np.asarray(entry["t_m"]), np.asarray(entry["K"]),
        np.asarray(entry["D"]))[0].reshape(-1, 2)
        for name, entry in state["cameras"].items()}
    selected = np.array([[900., 400.], [940., 450.]])
    def visual(_ref, target, *_a, **_k):
        name = "D01" if target is images["D01"] else "D03"
        return {"status": "matched_candidate", "matcher": "synthetic",
                "pixel_points": projected[name].tolist(),
                "point_checks": [{"status": "matched_candidate"}] * 2}
    # Distinct image objects let the mock identify the target view.
    images["D01"] = image.copy()
    images["D03"] = image.copy()
    monkeypatch.setattr("pixel_distance.auto3d.match_view", visual)
    result = auto_measure(images, "D04", selected, config)
    assert result["metric_group"] == "gantry"
    assert result["distance_m"] == pytest.approx(np.linalg.norm(world[1] - world[0]), abs=1e-6)
    assert result["views"]["D01"]["role"] == "cross_group_visual_candidate"
    assert result["views"]["D03"]["role"] == "cross_group_visual_candidate"


def test_split_group_points_only_yield_height_with_validated_alignment(monkeypatch):
    from pixel_distance.pose_state import state_from_nominal_group
    config = load_json("pixel_distance/camera_presets.json")
    images = {name: np.zeros((1003, 1784, 3), dtype=np.uint8)
              for name in ("D04", "D02", "D01", "D03")}
    states = [state_from_nominal_group({name: images[name] for name in names}, config)
              for names in (("D04", "D02"), ("D01", "D03"))]
    entries = {name: entry for state in states for name, entry in state["cameras"].items()}
    state = {"method": "surveyed_independent_group_poses",
             "scale_status": "surveyed_but_requires_independent_accuracy_checks",
             "cameras": entries}
    def project(name, point):
        entry = entries[name]
        return cv2.projectPoints(
            np.asarray([point], dtype=float),
            cv2.Rodrigues(np.asarray(entry["R"]))[0],
            np.asarray(entry["t_m"]), np.asarray(entry["K"]),
            np.asarray(entry["D"]))[0].reshape(2).tolist()
    a = [0.0, 0.0, 1.0]
    b = [0.0, 0.0, 2.0]
    selected = [project("D04", a), project("D04", b)]
    monkeypatch.setattr("pixel_distance.auto3d.match_view_with_pose",
                        lambda *_a, **_k: {"status": "partial_coverage",
                                           "pixel_points": [project("D02", a), None],
                                           "point_checks": [{"status": "matched_candidate"},
                                                            {"status": "unknown"}]})
    def visual(_reference, target, *_a, **_k):
        name = next(name for name in ("D01", "D03") if images[name] is target)
        return {"status": "partial_coverage",
                "pixel_points": [None, project(name, b)],
                "point_checks": [{"status": "unknown"},
                                 {"status": "matched_candidate"}]}
    monkeypatch.setattr("pixel_distance.auto3d.match_view", visual)
    alignment = {"status": "validated", "z_offsets_m": {"trolley": 0, "gantry": 0},
                 "point_height_error_bound_m": {"A": 0.1, "B": 0.1},
                 "alignment_error_bound_m": 0.05}
    result = auto_measure(images, "D04", selected, config,
                          pose_state=state, height_alignment=alignment)
    assert result["measurement_mode"] == "cross_group_height_only"
    assert result["distance_m"] is None
    assert result["height_difference_m"] == pytest.approx(1.0, abs=1e-6)
    assert result["height_difference_interval_m"] == pytest.approx([0.75, 1.25], abs=1e-6)
    missing = auto_measure(images, "D04", selected, config, pose_state=state)
    assert missing["status"] == "unknown"
    assert missing["distance_m"] is None
