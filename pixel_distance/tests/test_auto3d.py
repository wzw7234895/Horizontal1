import math

import cv2
import numpy as np
import pytest

from pixel_distance.auto3d import (_contextual_transfer, _nominal_baseline, _triangulate_pair,
                                   _valid_feature_mask, _xfeat_matches,
                                   auto_measure, match_view, match_view_with_pose)
from pixel_distance.cli import format_result


def test_feature_mask_keeps_visible_top_and_bottom_regions():
    gray = np.full((1080, 1920), 100, dtype=np.uint8)
    mask = _valid_feature_mask(gray)
    assert mask[100, 684] == 255
    assert mask[1000, 963] == 255
    assert mask[1031, 434] == 255
    assert mask[0, 684] == 0
    assert mask[1079, 434] == 0


def test_cross_group_auto_matcher_fills_missing_selected_point(monkeypatch):
    calls = []
    def candidate(_reference, _target, _points, _reference_camera,
                  _target_camera, method, *_args, **_kwargs):
        calls.append(method)
        pixels = ([[10, 10], None] if method == "sift" else [None, [20, 20]])
        return {"matcher": method, "pixel_points": pixels,
                "matched_point_count": 1,
                "point_checks": [{"status": "matched_candidate" if point else "unknown"}
                                 for point in pixels]}
    monkeypatch.setattr("pixel_distance.auto3d._match_view_with_method", candidate)
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    result = match_view(image, image, np.array([[1, 1], [2, 2]]), {}, {},
                        matcher="auto", allow_partial=True)
    assert calls == ["sift", "loftr"]
    assert result["status"] == "matched_candidate"
    assert result["pixel_points"] == [[10, 10], [20, 20]]


def test_contextual_transfer_uses_nearby_scene_support():
    source = np.array([[x, y] for x in range(40, 161, 20)
                       for y in range(20, 121, 20)], dtype=float)
    target = source + [120, 35]
    point, check = _contextual_transfer(np.array([90.0, 70.0]), source, target)
    np.testing.assert_allclose(point, [210, 105], atol=0.01)
    assert check["local_inliers"] >= 8
    with pytest.raises(ValueError, match="覆盖范围"):
        _contextual_transfer(np.array([205.0, 70.0]), source, target)


def test_context_candidate_cannot_override_bad_nominal_geometry(monkeypatch):
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    source = np.array([[x, y] for x in range(500, 661, 20)
                       for y in range(300, 461, 20)], dtype=float)
    target_pixels = source + [-100, 30]
    monkeypatch.setattr("pixel_distance.auto3d._symmetric_sift_matches",
                        lambda *_args, **_kwargs: (source.copy(), target_pixels.copy()))
    K = [[800, 0, 639.5], [0, 800, 359.5], [0, 0, 1]]
    reference = {"K": K, "D": [0] * 5, "R": np.eye(3).tolist(), "t_m": [0, 0, 0]}
    target = {**reference, "t_m": [-2, 0, 0]}
    result = match_view_with_pose(image, image, [[570, 370], [610, 410]],
                                  reference, target, matcher="sift", allow_partial=True)
    assert result["status"] == "unknown"
    assert result["pixel_points"] == [None, None]
    np.testing.assert_allclose(result["point_checks"][0]["visual_candidate_pixel"],
                               [470, 400], atol=0.1)
    assert result["point_checks"][0]["nominal_epipolar_error_px"] > 20


def test_baseline_uses_current_trolley_origin():
    gantry = {"rig": "gantry", "center_in_rig_m": [0, 0, 10]}
    trolley = {"rig": "trolley", "center_in_rig_m": [1, 0, 10]}
    origins = {"gantry": [0, 0, 0], "trolley": [2, 0, 0]}
    assert _nominal_baseline(gantry, trolley, origins) == pytest.approx(3)
    origins["trolley"] = [4, 0, 0]
    assert _nominal_baseline(gantry, trolley, origins) == pytest.approx(5)
    peer = {"rig": "trolley", "center_in_rig_m": [-1, 0, 10]}
    assert _nominal_baseline(trolley, peer, origins) == pytest.approx(2)


def test_triangulation_recovers_synthetic_distance_from_features():
    rng = np.random.default_rng(5)
    points = rng.uniform([-2, -1, 5], [2, 1, 9], size=(200, 3))
    baseline = 2.0
    k = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], dtype=float)
    left = points[:, :2] / points[:, 2, None]
    right = np.column_stack(((points[:, 0] - baseline) / points[:, 2],
                             points[:, 1] / points[:, 2]))
    pixels_left = left * 800 + [640, 360]
    pixels_right = right * 800 + [640, 360]
    selected = pixels_left[:2]
    matched = {"_source_inliers": pixels_left,
               "_target_inliers": pixels_right,
               "_reference_K": k, "_target_K": k,
               "_reference_scale": np.array([1, 1]),
               "_target_scale": np.array([1, 1]),
               "pixel_points": pixels_right[:2].tolist()}
    result = _triangulate_pair(selected, matched, baseline)
    assert result["distance_m"] == pytest.approx(np.linalg.norm(points[1] - points[0]), rel=1e-3)
    assert min(result["ray_angles_deg"]) > 1


def test_auto_measure_returns_unknown_when_views_lack_texture():
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    config = {"rig_origins_m": {"gantry": [0, 0, 0], "trolley": [0, 0, 0]},
              "cameras": {"D01": {"rig": "gantry", "center_in_rig_m": [0, 0, 10],
                                   "horizontal_fov_deg": 60},
                          "D03": {"rig": "gantry", "center_in_rig_m": [2, 0, 10],
                                   "horizontal_fov_deg": 60}}}
    result = auto_measure({"D01": blank, "D03": blank}, "D01",
                          np.array([[200, 200], [300, 300]]), config, matcher="sift")
    assert result["status"] == "unknown"
    assert result["views"]["D03"]["status"] == "unknown"


def test_cross_rig_reports_match_before_missing_scale(monkeypatch):
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    config = {"rig_origins_m": {"gantry": [0, 0, 0], "trolley": [0, 0, 0]},
              "cameras": {"D01": {"rig": "gantry", "center_in_rig_m": [0, 0, 10],
                                   "horizontal_fov_deg": 60},
                          "D02": {"rig": "trolley", "center_in_rig_m": [2, 0, 10],
                                   "horizontal_fov_deg": 60}}}

    selected = [[200, 200], [300, 300]]
    def known_match(*_args, **_kwargs):
        return {"status": "matched_candidate", "pixel_points": selected,
                "_source_inliers": np.zeros((0, 2))}

    monkeypatch.setattr("pixel_distance.auto3d.match_view", known_match)
    result = auto_measure({"D01": blank, "D02": blank}, "D01",
                          selected, config, matcher="sift")
    assert result["status"] == "unknown"
    assert result["views"]["D02"]["status"] == "matched_no_scale"
    assert result["views"]["D02"]["pixel_points"] == selected
    assert "Current relative position" in result["views"]["D02"]["reason"]
    assert "D02 已匹配/缺少本帧跨组位置" in format_result("auto-3d", result)
    with pytest.raises(ValueError, match="three finite"):
        auto_measure({"D01": blank, "D02": blank}, "D01",
                     [[200, 200], [300, 300]], config, matcher="sift",
                     trolley_origin_m=(float("nan"), 0, 0))


def test_cross_rig_runtime_origin_changes_baseline(monkeypatch):
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    config = {"rig_origins_m": {"gantry": [0, 0, 0], "trolley": [0, 0, 0]},
              "cameras": {"D01": {"rig": "gantry", "center_in_rig_m": [0, 0, 10],
                                   "horizontal_fov_deg": 60},
                          "D02": {"rig": "trolley", "center_in_rig_m": [1, 0, 10],
                                   "horizontal_fov_deg": 60}}}
    selected = [[200, 200], [300, 300]]
    monkeypatch.setattr("pixel_distance.auto3d.match_view",
                        lambda *_args, **_kwargs: {"status": "matched_candidate",
                                                   "pixel_points": selected})
    monkeypatch.setattr("pixel_distance.auto3d._triangulate_pair",
                        lambda _selected, _matched, baseline: {"distance_m": baseline,
                                                                "nominal_baseline_m": baseline})
    first = auto_measure({"D01": blank, "D02": blank}, "D01", selected, config,
                         matcher="sift", trolley_origin_m=(2, 0, 0))
    second = auto_measure({"D01": blank, "D02": blank}, "D01", selected, config,
                          matcher="sift", trolley_origin_m=(4, 0, 0))
    assert first["views"]["D02"]["metric"]["nominal_baseline_m"] == pytest.approx(3)
    assert second["views"]["D02"]["metric"]["nominal_baseline_m"] == pytest.approx(5)
    assert config["rig_origins_m"]["trolley"] == [0, 0, 0]


def test_pose_matching_and_distance_use_arbitrary_height_points(monkeypatch):
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    K = np.array([[800, 0, 639.5], [0, 800, 359.5], [0, 0, 1]], dtype=float)
    rng = np.random.default_rng(21)
    world = rng.uniform([-0.3, -1.0, 6.0], [2.4, 1.0, 11.0], size=(80, 3))
    world[:2] = [[0.4, -0.3, 6.2], [1.6, 0.5, 9.3]]
    reference_pixels = cv2.projectPoints(world, np.zeros(3), np.zeros(3), K, None)[0].reshape(-1, 2)
    target_t = np.array([-2.0, 0.0, 0.0])
    target_pixels = cv2.projectPoints(world, np.zeros(3), target_t, K, None)[0].reshape(-1, 2)
    monkeypatch.setattr("pixel_distance.auto3d._symmetric_sift_matches",
                        lambda *_args, **_kwargs: (reference_pixels.copy(), target_pixels.copy()))
    entries = {
        "D04": {"K": K.tolist(), "D": [0, 0, 0, 0, 0], "R": np.eye(3).tolist(),
                "t_m": [0, 0, 0], "image_size": [1280, 720]},
        "D03": {"K": K.tolist(), "D": [0, 0, 0, 0, 0], "R": np.eye(3).tolist(),
                "t_m": target_t.tolist(), "image_size": [1280, 720]},
    }
    from pixel_distance.pose_state import decoded_hash
    for entry in entries.values():
        entry["decoded_image_sha256"] = decoded_hash(image)
    matched = match_view_with_pose(image, image, reference_pixels[:2],
                                   entries["D04"], entries["D03"])
    np.testing.assert_allclose(matched["pixel_points"], target_pixels[:2], atol=1e-6)
    partial = match_view_with_pose(image, image, np.array([reference_pixels[0], [1150, 650]]),
                                   entries["D04"], entries["D03"], allow_partial=True)
    assert partial["status"] == "partial_coverage"
    assert partial["pixel_points"][1] is None
    config = {"rig_origins_m": {"gantry": [0, 0, 0], "trolley": [0, 0, 0]},
              "cameras": {name: {"rig": "gantry" if name == "D03" else "trolley"}
                          for name in entries}}
    state = {"method": "surveyed_common_world_poses", "scale_status": "synthetic",
             "cameras": entries}
    result = auto_measure({"D04": image, "D03": image}, "D04", reference_pixels[:2],
                          config, pose_state=state)
    assert result["distance_m"] == pytest.approx(np.linalg.norm(world[1] - world[0]), abs=1e-6)
    assert result["height_difference_m"] == pytest.approx(world[1, 2] - world[0, 2], abs=1e-6)
    assert result["scale_source"] == "current_pose_state"


def test_auto_matcher_tries_rectified_patch_after_feature_matchers(monkeypatch):
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    entry = {"K": [[100, 0, 60], [0, 100, 40], [0, 0, 1]],
             "D": [0, 0, 0, 0, 0], "R": np.eye(3).tolist(),
             "t_m": [0, 0, 0]}
    target = {**entry, "t_m": [-1, 0, 0]}
    monkeypatch.setattr("pixel_distance.auto3d._symmetric_sift_matches",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("no SIFT")))
    monkeypatch.setattr("pixel_distance.auto3d._loftr_matches",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("no LoFTR")))
    monkeypatch.setattr("pixel_distance.auto3d.match_rectified_patches",
                        lambda *_args, **_kwargs: {"status": "matched_candidate",
                                                   "matcher": "patch",
                                                   "pixel_points": [[45, 40], [55, 40]],
                                                   "point_checks": [{"status": "matched_candidate"}] * 2})
    result = match_view_with_pose(image, image, [[50, 40], [60, 40]],
                                  entry, target, matcher="auto", allow_partial=True)
    assert result["matcher"] == "patch"


def test_auto_matcher_combines_independently_verified_points(monkeypatch):
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    entry = {"K": [[100, 0, 60], [0, 100, 40], [0, 0, 1]],
             "D": [0, 0, 0, 0, 0], "R": np.eye(3).tolist(),
             "t_m": [0, 0, 0]}
    target = {**entry, "t_m": [-1, 0, 0]}
    source = np.array([[40, 40], [50, 40], [60, 40]], dtype=float)
    monkeypatch.setattr("pixel_distance.auto3d._symmetric_sift_matches",
                        lambda *_args, **_kwargs: (source.copy(), source - [5, 0]))

    def sift_local(point, *_args):
        if point[0] > 55:
            raise ValueError("B lacks SIFT support")
        return np.array([45.0, 40.0]), {"method": "nearby_matched_feature"}

    monkeypatch.setattr("pixel_distance.auto3d._local_match", sift_local)
    monkeypatch.setattr("pixel_distance.auto3d._loftr_matches",
                        lambda *_args: (_ for _ in ()).throw(ValueError("no LoFTR")))
    monkeypatch.setattr("pixel_distance.auto3d.match_rectified_patches",
                        lambda *_args, **_kwargs: {
                            "status": "partial_coverage", "matcher": "patch",
                            "pixel_points": [None, [55, 40]],
                            "point_checks": [{"status": "unknown"},
                                             {"status": "matched_candidate"}]})
    result = match_view_with_pose(image, image, [[50, 40], [60, 40]],
                                  entry, target, matcher="auto", allow_partial=True)
    assert result["status"] == "matched_candidate"
    assert result["matcher"] == "auto_combined"
    assert result["pixel_points"] == [[45.0, 40.0], [55, 40]]
    assert [check["source_matcher"] for check in result["point_checks"]] == ["sift", "patch"]


@pytest.mark.parametrize("patch_x, rejected", [(52, False), (53, False),
                                                 (54, True), (65, True)])
def test_auto_matcher_rejects_conflicting_point_predictions(monkeypatch, patch_x, rejected):
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    entry = {"K": [[100, 0, 60], [0, 100, 40], [0, 0, 1]],
             "D": [0, 0, 0, 0, 0], "R": np.eye(3).tolist(),
             "t_m": [0, 0, 0]}
    target = {**entry, "t_m": [-1, 0, 0]}
    source = np.array([[40, 40], [50, 40], [60, 40]], dtype=float)
    monkeypatch.setattr("pixel_distance.auto3d._symmetric_sift_matches",
                        lambda *_args, **_kwargs: (source.copy(), source - [5, 0]))
    monkeypatch.setattr("pixel_distance.auto3d._loftr_matches",
                        lambda *_args: (_ for _ in ()).throw(ValueError("no LoFTR")))
    monkeypatch.setattr("pixel_distance.auto3d.match_rectified_patches",
                        lambda *_args, **_kwargs: {
                            "status": "partial_coverage", "matcher": "patch",
                            "pixel_points": [[patch_x, 40], None],
                            "point_checks": [{"status": "matched_candidate"},
                                             {"status": "unknown"}]})
    def first_only(point, *_args):
        if point[0] > 55:
            raise ValueError("B lacks support")
        return np.array([45.0, 40.0]), {"method": "nearby_matched_feature"}
    monkeypatch.setattr("pixel_distance.auto3d._local_match", first_only)
    result = match_view_with_pose(image, image, [[50, 40], [60, 40]],
                                  entry, target, matcher="auto", allow_partial=True)
    assert result["pixel_points"] == ([None, None] if rejected else [[45.0, 40.0], None])
    if rejected:
        assert result["point_checks"][0]["candidate_disagreement_px"] == pytest.approx(patch_x - 45)


def test_roma_candidates_require_visual_and_nominal_geometry_agreement(monkeypatch):
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    entry = {"K": [[100, 0, 60], [0, 100, 40], [0, 0, 1]],
             "D": [0, 0, 0, 0, 0], "R": np.eye(3).tolist(),
             "t_m": [0, 0, 0]}
    target = {**entry, "t_m": [-1, 0, 0]}
    points = [{"target_candidate_pixel": [45, 40], "visual_consistent": True,
               "nominal_geometry_consistent": True},
              {"target_candidate_pixel": [55, 40], "visual_consistent": True,
               "nominal_geometry_consistent": False}]
    monkeypatch.setattr("pixel_distance.roma_probe.probe_with_entries",
                        lambda *_args, **_kwargs: {"points": points})
    result = match_view_with_pose(image, image, [[50, 40], [60, 40]],
                                  entry, target, matcher="roma", allow_partial=True)
    assert result["status"] == "partial_coverage"
    assert result["pixel_points"] == [[45, 40], None]
    with pytest.raises(ValueError, match="未通过"):
        match_view_with_pose(image, image, [[50, 40], [60, 40]],
                             entry, target, matcher="roma")


def test_pose_state_can_triangulate_points_from_different_auxiliary_views(monkeypatch):
    from pixel_distance.pose_state import decoded_hash

    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    K = np.array([[900, 0, 639.5], [0, 900, 359.5], [0, 0, 1]], dtype=float)
    world = np.array([[0.4, -0.2, 6.0], [1.1, 0.5, 8.0]])
    translations = {"D04": np.array([0.0, 0.0, 0.0]),
                    "D02": np.array([-2.0, 0.0, 0.0]),
                    "D03": np.array([0.0, -2.0, 0.0])}
    projected = {name: cv2.projectPoints(world, np.zeros(3), t, K, None)[0].reshape(-1, 2)
                 for name, t in translations.items()}
    entries = {name: {"K": K.tolist(), "D": [0, 0, 0, 0, 0],
                      "R": np.eye(3).tolist(), "t_m": t.tolist(),
                      "image_size": [1280, 720], "decoded_image_sha256": decoded_hash(image),
                      "camera_id": name}
               for name, t in translations.items()}

    def partial_match(_reference, _target, _points, _reference_entry, target_entry,
                      *_args, **_kwargs):
        index = 0 if target_entry["camera_id"] == "D02" else 1
        pixels = [None, None]
        pixels[index] = projected[target_entry["camera_id"]][index].tolist()
        checks = [{"status": "unknown"}, {"status": "unknown"}]
        checks[index] = {"status": "matched_candidate"}
        return {"status": "partial_coverage", "pixel_points": pixels,
                "point_checks": checks, "matcher": "sift"}

    monkeypatch.setattr("pixel_distance.auto3d.match_view_with_pose", partial_match)
    config = {"rig_origins_m": {"gantry": [0, 0, 0], "trolley": [0, 0, 0]},
              "cameras": {name: {"rig": "trolley" if name != "D03" else "gantry"}
                          for name in entries}}
    state = {"method": "surveyed_common_world_poses", "scale_status": "synthetic",
             "cameras": entries}
    result = auto_measure({name: image for name in entries}, "D04", projected["D04"],
                          config, pose_state=state)
    assert result["distance_m"] == pytest.approx(np.linalg.norm(world[1] - world[0]), abs=1e-6)
    assert result["point_support"] == {"A": ["D02"], "B": ["D03"]}
    assert result["metric_pairs"] == []
    assert not result["cross_view_consistency_checked"]

    wrong_world = world[0] * 1.3
    wrong_d03_a = cv2.projectPoints(wrong_world[None], np.zeros(3),
                                    translations["D03"], K, None)[0].reshape(2)

    def conflicting_match(_reference, _target, _points, _reference_entry, target_entry,
                          *_args, **_kwargs):
        name = target_entry["camera_id"]
        pixels = projected[name].copy()
        if name == "D03":
            pixels[0] = wrong_d03_a
        return {"status": "matched_candidate", "pixel_points": pixels.tolist(),
                "point_checks": [{"status": "matched_candidate"},
                                 {"status": "matched_candidate"}], "matcher": "sift"}

    monkeypatch.setattr("pixel_distance.auto3d.match_view_with_pose", conflicting_match)
    conflict = auto_measure({name: image for name in entries}, "D04", projected["D04"],
                            config, pose_state=state)
    assert conflict["status"] == "unknown"
    assert "views disagree" in conflict["reason"]


def test_vehicle_diagnostic_pose_is_rejected_by_runtime_api():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    config = {"cameras": {"D04": {"rig": "trolley"}, "D03": {"rig": "gantry"}}}
    state = {"method": "vehicle_length_diagnostic_only"}
    with pytest.raises(ValueError, match="不能用于正式测距"):
        auto_measure({"D04": image, "D03": image}, "D04", [[200, 200], [300, 300]],
                     config, pose_state=state)


def test_xfeat_requires_local_files_and_keeps_visible_edge_features(monkeypatch, tmp_path):
    monkeypatch.delenv("PIXEL_DISTANCE_XFEAT_REPO", raising=False)
    monkeypatch.delenv("PIXEL_DISTANCE_XFEAT_WEIGHTS", raising=False)
    frame = np.full((100, 160, 3), 120, dtype=np.uint8)
    with pytest.raises(ValueError, match="本地"):
        _xfeat_matches(frame, frame)

    weights = tmp_path / "xfeat.pt"
    weights.touch()
    monkeypatch.setenv("PIXEL_DISTANCE_XFEAT_REPO", str(tmp_path))
    monkeypatch.setenv("PIXEL_DISTANCE_XFEAT_WEIGHTS", str(weights))

    class StubModel:
        def match_xfeat(self, *_args, **_kwargs):
            points = np.array([[40, 5], [50, 40], [80, 50], [100, 96],
                               [-2, 50], [160, 50]], dtype=float)
            return points, points + [1, 0]

    monkeypatch.setattr("pixel_distance.auto3d._xfeat_model",
                        lambda *_args: StubModel())
    source, target = _xfeat_matches(frame, frame)
    np.testing.assert_array_equal(source, [[40, 5], [50, 40], [80, 50], [100, 96]])
    np.testing.assert_array_equal(target, [[41, 5], [51, 40], [81, 50], [101, 96]])
