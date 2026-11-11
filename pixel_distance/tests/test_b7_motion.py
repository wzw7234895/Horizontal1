import cv2
import numpy as np
import pytest

from pixel_distance import b7_motion
from pixel_distance.cli import format_result, parser
from pixel_distance.geometry import load_json
from pixel_distance.pose_state import nominal_camera_entry


CONFIG = load_json("pixel_distance/camera_presets.json")
NAMES = ("D01", "D02", "D03", "D04")


def _bundles(count=17, step=0.2):
    return [(index * step, {name: np.zeros((120, 160, 3), np.uint8)
                            for name in NAMES}) for index in range(count)]


def _fake_motion(vectors):
    centers = {name: tuple(CONFIG["nominal_world_poses"][name]["center_m"])
               for name in NAMES}

    def estimate(_previous, _current, camera, _mask):
        R = np.asarray(camera["R"])
        t = np.asarray(camera["t_m"])
        center = tuple(np.round(-R.T @ t, 6))
        name = next(name for name, value in centers.items()
                    if tuple(np.round(value, 6)) == center)
        return {"status": "estimated", "displacement_xy_m": vectors[name]}

    return estimate


def test_b7_requires_external_load_and_consecutive_two_axis_motion(monkeypatch):
    vectors = {"D01": [0, 0.2], "D03": [0, 0.2],
               "D02": [0.16, 0.2], "D04": [0.16, 0.2]}
    monkeypatch.setattr(b7_motion, "estimate_camera_motion", _fake_motion(vectors))
    result = b7_motion.analyze_b7(_bundles(), True, CONFIG)
    assert result["status"] == "alarm_candidate"
    assert result["b7_alarm"] is True
    assert result["distance_geometry_updated"] is False
    assert all(item["simultaneous_motion"] for item in result["intervals"])
    assert result["intervals"][0]["trolley_axis_speed_mps"] == pytest.approx(0.8)
    assert result["longest_confirmed_duration_s"] == pytest.approx(3.2)
    assert b7_motion.analyze_b7([], False, CONFIG)["b7_alarm"] is False
    assert b7_motion.analyze_b7([], None, CONFIG)["status"] == "unknown_load_state"


def test_b7_rejects_single_axis_or_conflicting_camera_motion(monkeypatch):
    gantry_only = {"D01": [0, 0.2], "D03": [0, 0.2],
                   "D02": [0, 0.2], "D04": [0, 0.2]}
    monkeypatch.setattr(b7_motion, "estimate_camera_motion", _fake_motion(gantry_only))
    result = b7_motion.analyze_b7(_bundles(), True, CONFIG)
    assert result["status"] == "no_simultaneous_motion"
    assert result["b7_alarm"] is False
    disagreement = dict(gantry_only, D03=[0, -0.2])
    monkeypatch.setattr(b7_motion, "estimate_camera_motion", _fake_motion(disagreement))
    result = b7_motion.analyze_b7(_bundles(), True, CONFIG)
    assert result["status"] == "unknown"
    assert "不一致" in result["intervals"][0]["reason"]


def test_b7_checks_all_cameras_and_does_not_require_rigid_cross_group():
    config = dict(CONFIG, all_cameras_rigid=False)
    with pytest.raises(ValueError, match="D01 至 D04"):
        b7_motion.analyze_b7([(0, {})], True, config)
    image = np.zeros((120, 160, 3), np.uint8)
    assert nominal_camera_entry("D04", image, config)["image_size"] == [160, 120]
    args = parser().parse_args(["b7-motion", "--time", "2026-10-20T15:57:32",
                                "--load-state", "confirmed"])
    assert args.window_radius_s == 3.0
    assert args.sample_step_s == 0.2
    assert args.min_motion_duration_s == 3.0
    assert "候选告警" in format_result("b7-motion", {"status": "alarm_candidate"})


def test_camera_motion_recovers_known_ground_shift_and_rejects_blank():
    rng = np.random.default_rng(7)
    image = rng.integers(50, 220, (480, 640, 3), dtype=np.uint8)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, -6], [0, 1, 0]]),
                             (640, 480), borderMode=cv2.BORDER_REFLECT)
    camera = nominal_camera_entry("D04", image, CONFIG)
    result = b7_motion.estimate_camera_motion(image, shifted, camera)
    assert result["status"] == "estimated"
    assert result["displacement_xy_m"][0] == pytest.approx(0.16, abs=0.06)
    assert abs(result["displacement_xy_m"][1]) < 0.04
    blank = np.zeros_like(image)
    assert b7_motion.estimate_camera_motion(blank, blank, camera)["status"] == "unknown"


def test_b7_short_motion_and_direction_reversal_do_not_alarm(monkeypatch):
    moving = {"D01": [0, 0.04], "D03": [0, 0.04],
              "D02": [0.032, 0.04], "D04": [0.032, 0.04]}
    stopped = {name: [0, 0] for name in NAMES}
    calls = 0

    def estimate(previous, current, camera, mask):
        nonlocal calls
        interval = calls // 4
        calls += 1
        vectors = moving if interval < 4 else stopped
        return _fake_motion(vectors)(previous, current, camera, mask)

    monkeypatch.setattr(b7_motion, "estimate_camera_motion", estimate)
    result = b7_motion.analyze_b7(_bundles(7), True, CONFIG)
    assert result["b7_alarm"] is False
    assert result["longest_confirmed_duration_s"] == pytest.approx(0.8)

    calls = 0

    def reversing(previous, current, camera, mask):
        nonlocal calls
        interval = calls // 4
        calls += 1
        vectors = moving if interval < 3 else {
            name: [-xy[0], -xy[1]] for name, xy in moving.items()}
        return _fake_motion(vectors)(previous, current, camera, mask)

    monkeypatch.setattr(b7_motion, "estimate_camera_motion", reversing)
    result = b7_motion.analyze_b7(_bundles(7), True, CONFIG)
    assert result["b7_alarm"] is False
    assert result["longest_confirmed_duration_s"] == pytest.approx(0.6)


def test_b7_stream_resets_on_load_change_or_frame_gap(monkeypatch):
    moving = {"D01": [0, 0.04], "D03": [0, 0.04],
              "D02": [0.032, 0.04], "D04": [0.032, 0.04]}
    monkeypatch.setattr(b7_motion, "estimate_camera_motion", _fake_motion(moving))
    detector = b7_motion.B7MotionDetector(CONFIG)
    for timestamp, images in _bundles():
        result = detector.update(timestamp, images, True)
    assert result["b7_alarm"] is True
    assert detector.update(1.4, None, False)["status"] == "not_loaded"
    result = detector.update(1.6, _bundles(1)[0][1], True)
    assert result["b7_alarm"] is False
    result = detector.update(2.5, _bundles(1)[0][1], True)
    assert result["status"] == "unknown"
    assert result["intervals"][-1]["reason"] == "相邻画面间隔超出上限"
    with pytest.raises(ValueError, match="严格递增"):
        detector.update(2.5, _bundles(1)[0][1], True)


def test_b7_requires_full_three_seconds(monkeypatch):
    moving = {"D01": [0, 0.04], "D03": [0, 0.04],
              "D02": [0.032, 0.04], "D04": [0.032, 0.04]}
    monkeypatch.setattr(b7_motion, "estimate_camera_motion", _fake_motion(moving))
    assert b7_motion.analyze_b7(_bundles(15), True, CONFIG)["b7_alarm"] is False
    result = b7_motion.analyze_b7(_bundles(16), True, CONFIG)
    assert result["b7_alarm"] is True
    assert result["longest_confirmed_duration_s"] == pytest.approx(3.0)


def test_b7_stream_keeps_previous_frame_when_decoder_reuses_buffer():
    detector = b7_motion.B7MotionDetector(CONFIG)
    images = _bundles(1)[0][1]
    detector.update(0.0, images, True)
    images["D01"][0, 0, 0] = 123
    assert detector.previous[1]["D01"][0, 0, 0] == 0
