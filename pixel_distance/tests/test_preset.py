import copy

import cv2
import numpy as np
import pytest

from pixel_distance.cli import click_points, format_result
from pixel_distance.preset import estimate_on_plane


CONFIG = {"expected_aspect_ratio": 16 / 9,
          "cameras": {"D02": {"name": "top", "height_m": 10.0,
                               "horizontal_fov_deg": 90.0,
                               "down_tilt_deg": 90.0, "roll_deg": 0.0},
                      "D01": {"name": "side", "height_m": 10.0,
                               "horizontal_fov_deg": 60.0,
                               "down_tilt_deg": 45.0, "roll_deg": 0.0}}}


def test_nadir_known_geometry_and_parameter_scaling():
    center = (959.5, 539.5)
    right = (1151.5, 539.5)
    result = estimate_on_plane(CONFIG, "D02", (1920, 1080), center, right)
    assert result["distance_m"] == pytest.approx(2.0, abs=1e-10)
    elevated = estimate_on_plane(CONFIG, "D02", (1920, 1080), center, right, 3.0)
    assert elevated["distance_m"] == pytest.approx(1.4, abs=1e-10)
    higher = copy.deepcopy(CONFIG)
    higher["cameras"]["D02"]["height_m"] = 12.0
    assert estimate_on_plane(higher, "D02", (1920, 1080), center, right)["distance_m"] == pytest.approx(2.4)


def test_slanted_view_has_position_dependent_scale():
    near = estimate_on_plane(CONFIG, "D01", (1920, 1080), (959.5, 700), (1059.5, 700))
    far = estimate_on_plane(CONFIG, "D01", (1920, 1080), (959.5, 400), (1059.5, 400))
    assert far["distance_m"] > near["distance_m"]


def test_rejects_crop_and_rays_above_horizon():
    with pytest.raises(ValueError, match="aspect ratio"):
        estimate_on_plane(CONFIG, "D02", (1000, 1000), (10, 10), (20, 20))
    shallow = copy.deepcopy(CONFIG)
    shallow["cameras"]["D01"]["down_tilt_deg"] = 10.0
    with pytest.raises(ValueError, match="does not intersect"):
        estimate_on_plane(shallow, "D01", (1920, 1080), (959.5, 0), (959.5, 800))
    with pytest.raises(ValueError, match="between the floor"):
        estimate_on_plane(CONFIG, "D02", (1920, 1080), (100, 100), (200, 200), 11)


def test_click_prompts_once_per_point(monkeypatch, capsys):
    state = {"callback": None, "polls": 0}
    monkeypatch.setattr(cv2, "namedWindow", lambda *_args: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *_args: None)
    monkeypatch.setattr(cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *_args: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda _name, callback: state.update(callback=callback))

    def wait_key(_delay):
        state["polls"] += 1
        if state["polls"] == 3:
            state["callback"](cv2.EVENT_LBUTTONDOWN, 10, 20, 0, None)
        if state["polls"] == 6:
            state["callback"](cv2.EVENT_LBUTTONDOWN, 30, 40, 0, None)
        return -1

    monkeypatch.setattr(cv2, "waitKey", wait_key)
    points = click_points(np.zeros((100, 100, 3), np.uint8), ["A", "B"])
    assert points == [[10, 20], [30, 40]]
    assert capsys.readouterr().out.splitlines() == ["请点击 A 点（1/2）", "请点击 B 点（2/2）"]


def test_estimate_summary_is_short_chinese():
    result = estimate_on_plane(CONFIG, "D02", (1920, 1080), (959.5, 539.5), (1151.5, 539.5))
    summary = format_result("estimate", result)
    assert summary.count("\n") == 1
    assert "估算距离：2.000 米" in summary
    assert "参数未标定" in summary
