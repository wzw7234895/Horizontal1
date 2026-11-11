from pathlib import Path

import numpy as np
import pytest

from pixel_distance.geometry import load_image, load_json
from pixel_distance.pose_state import state_from_vehicle_diagnostic
from pixel_distance.vehicle_anchor import (estimate_cart_anchor,
                                           measure_on_cart_plane,
                                           triangulate_with_cart_anchor)


ROOT = Path(__file__).resolve().parents[2]
FRAMES = ROOT / "汇报材料" / "现场同刻画面_2026-11-05"
CONFIG = ROOT / "pixel_distance" / "camera_presets.json"


def site_pair():
    if not all((FRAMES / name).is_file() for name in (
            "02_主钩吊点_15-57-32.png", "04_左侧人员侦测_15-57-32.png")):
        pytest.skip("现场样例图片未随代码仓库提供")
    return (load_image(str(FRAMES / "02_主钩吊点_15-57-32.png")),
            load_image(str(FRAMES / "04_左侧人员侦测_15-57-32.png")))


def test_vehicle_markers_recover_scale_and_project_cart_points():
    reference, target = site_pair()
    config = load_json(str(CONFIG))
    anchor = estimate_cart_anchor(reference, target,
                                  config["cameras"]["D04"],
                                  config["cameras"]["D03"], (4, 5))
    result = measure_on_cart_plane([[1205.4743, 412.8686],
                                    [1200.3771, 858.8686]], anchor)
    assert 4 < result["distance_m"] < 5
    assert result["length_only_range_m"][0] < result["distance_m"]
    assert result["distance_m"] < result["length_only_range_m"][1]
    assert result["edge_surface_ambiguous"] is True
    assert result["edge_snap_px"][1] == pytest.approx(5.15, abs=0.1)
    np.testing.assert_allclose(result["pixel_points"],
                               [[895, 912], [381, 937]], atol=8)
    assert anchor["checks"]["max_marker_reprojection_error_px"] < 6
    with pytest.raises(ValueError, match="visible cart top"):
        measure_on_cart_plane([[200, 200], [300, 300]], anchor)


def test_cart_pose_triangulates_independently_corresponded_off_cart_points():
    reference, target = site_pair()
    config = load_json(str(CONFIG))
    anchor = estimate_cart_anchor(reference, target,
                                  config["cameras"]["D04"],
                                  config["cameras"]["D03"], (4, 5))
    world = np.array([[2.0, 0.7, 1.0], [3.0, 0.8, 1.8]])
    projected = []
    for fit in (anchor["_reference_fit"], anchor["_target_fit"]):
        camera = (fit["R"] @ world.T + fit["t"][:, None]).T
        xy = camera[:, :2] / camera[:, 2, None]
        projected.append((xy @ fit["K"][:2, :2].T + fit["K"][:2, 2]).tolist())
    result = triangulate_with_cart_anchor(projected[0], projected[1], anchor)
    assert result["distance_m"] == pytest.approx(np.linalg.norm(world[1] - world[0]),
                                                 abs=1e-6)
    assert max(result["reprojection_error_px"]) < 1e-6


def test_vehicle_anchor_only_creates_diagnostic_state():
    reference, target = site_pair()
    config = load_json(str(CONFIG))
    anchor = estimate_cart_anchor(reference, target, config["cameras"]["D04"],
                                  config["cameras"]["D03"], (4, 5))
    state = state_from_vehicle_diagnostic(
        {"D04": reference, "D03": target}, anchor, {"D04": 10, "D03": 10},
        "D04", "D03")
    assert state["method"] == "vehicle_length_diagnostic_only"
    assert state["scale_status"] == "approximate_vehicle_length_not_validated"
    assert set(state["cameras"]) == {"D03", "D04"}


def test_vehicle_anchor_rejects_missing_markers_and_invalid_length():
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    config = load_json(str(CONFIG))
    with pytest.raises(ValueError, match="length range"):
        estimate_cart_anchor(blank, blank, config["cameras"]["D04"],
                             config["cameras"]["D03"], (5, 4))
    with pytest.raises(ValueError, match="four cart-end markers"):
        estimate_cart_anchor(blank, blank, config["cameras"]["D04"],
                             config["cameras"]["D03"], (4, 5))
