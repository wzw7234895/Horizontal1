from datetime import datetime
from io import BytesIO
from subprocess import CompletedProcess

import cv2
import numpy as np
import pytest

from pixel_distance.cli import format_result
from pixel_distance.geometry import load_json
from pixel_distance.video_distance import (extract_frame, measure_video_views,
                                           parse_instant)


def test_second_resolution_video_extraction_applies_configured_offset(tmp_path):
    path = tmp_path / "D04_20261020155700.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"),
                             10.0, (64, 48))
    assert writer.isOpened()
    for index in range(100):
        writer.write(np.full((48, 64, 3), index, dtype=np.uint8))
    writer.release()
    frame, info = extract_frame(path, "D04", datetime(2026, 10, 20, 15, 57, 5), 2.0)
    assert info["frame_index"] == 70
    assert frame.shape == (48, 64, 3)
    assert abs(float(frame.mean()) - 70) < 3
    assert info["decoder_warning_lines"] == 0
    with pytest.raises(ValueError, match="不覆盖"):
        extract_frame(path, "D04", datetime(2026, 10, 20, 15, 58, 0))
    with pytest.raises(ValueError, match="文件名"):
        extract_frame(path, "D03", datetime(2026, 10, 20, 15, 57, 5))


def test_decoder_warnings_are_captured_without_hiding_decode_failure(monkeypatch, tmp_path, capsys):
    path = tmp_path / "D04_20261020155700.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"),
                             10.0, (64, 48))
    assert writer.isOpened()
    writer.write(np.full((48, 64, 3), 80, dtype=np.uint8))
    writer.release()
    data = BytesIO()
    np.save(data, np.full((48, 64, 3), 80, dtype=np.uint8), allow_pickle=False)
    monkeypatch.setattr("pixel_distance.video_distance.subprocess.run",
                        lambda *_args, **_kwargs: CompletedProcess(
                            [], 0, data.getvalue(), b"[h264] reference picture missing\n"))
    frame, info = extract_frame(path, "D04", datetime(2026, 10, 20, 15, 57, 0))
    assert frame.shape == (48, 64, 3)
    assert info["decoder_warning_lines"] == 1
    assert capsys.readouterr().err == ""
    monkeypatch.setattr("pixel_distance.video_distance.subprocess.run",
                        lambda *_args, **_kwargs: CompletedProcess([], 2, b"", b"decode failed"))
    with pytest.raises(ValueError, match="解码失败"):
        extract_frame(path, "D04", datetime(2026, 10, 20, 15, 57, 0))


def test_time_format_requires_exact_second():
    assert parse_instant("2026-10-20T15:57:32").second == 32
    for value in ("2026-10-20", "2026-10-20T15:57:32.500", "15:57:32"):
        with pytest.raises(ValueError, match="时间格式"):
            parse_instant(value)


def test_video_distance_requires_explicit_plane_assumption(monkeypatch):
    config = load_json("pixel_distance/camera_presets.json")
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    images = {"D04": image, "D03": image}
    selected = np.array([[950, 500], [1050, 500]], dtype=float)
    monkeypatch.setattr("pixel_distance.video_distance.auto_measure",
                        lambda *_args, **_kwargs: {"status": "unknown", "views": {"D03": {"status": "unknown"}}})
    result = measure_video_views(images, "D04", selected, config)
    assert result["status"] == "unknown"
    assert result["distance_m"] is None
    result = measure_video_views(images, "D04", selected, config, allow_assumed_plane=True)
    assert result["status"] == "single_view_assumed_plane_estimate"
    assert result["actual_distance_determined"] is False
    assert result["height_difference_determined"] is False
    assert result["height_difference_m"] is None
    assert result["plane_height_m"] == 0.0
    assert result["multiview_attempt"]["views"]["D03"]["status"] == "unknown"

    result["multiview_attempt"]["views"]["D03"]["reason"] = (
        "sift: 极线候选特征不足（1/30；最小极线误差 1.0 像素）")
    assert "D03 1/30" in format_result("video-distance", result)
    with pytest.raises(ValueError, match="非负"):
        measure_video_views(images, "D04", selected, config, plane_height_m=-1)

    monkeypatch.setattr("pixel_distance.video_distance.auto_measure",
                        lambda *_args, **_kwargs: {"status": "nominal_3d_candidate_not_calibrated",
                                                   "distance_m": 3.2, "metric_pairs": ["D03"]})
    result = measure_video_views(images, "D04", selected, config)
    assert result["measurement_mode"] == "same_group_nominal_3d"
    assert result["distance_m"] == 3.2
    assert result["single_view_fallback"] is False
    monkeypatch.setattr("pixel_distance.video_distance.auto_measure",
                        lambda *_args, **_kwargs: {
                            "status": "surveyed_cross_group_height_requires_validation",
                            "measurement_mode": "cross_group_height_only",
                            "distance_m": None, "height_difference_m": 0.4,
                            "height_difference_interval_m": [0.2, 0.6]})
    height_only = measure_video_views(images, "D04", selected, config)
    assert height_only["measurement_mode"] == "cross_group_height_only"
    assert height_only["distance_m"] is None
    assert "完整三维距离不可判定" in format_result("video-distance", height_only)
    with pytest.raises(ValueError, match="只支持地面"):
        measure_video_views(images, "D04", selected, config,
                            plane_height_m=1, ground_model_dir="local-model")


def test_single_view_range_fallback_when_plane_ray_is_invalid(monkeypatch):
    config = load_json("pixel_distance/camera_presets.json")
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    monkeypatch.setattr("pixel_distance.video_distance.estimate_on_plane",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("horizon")))
    result = measure_video_views({"D04": image}, "D04",
                                 [[950, 500], [1050, 500]], config,
                                 allow_assumed_plane=True)
    assert result["status"] == "single_view_assumed_range_estimate"
    assert result["distance_m"] > 0


@pytest.mark.parametrize("ground_candidate", [False, True])
def test_ground_segmentation_gates_only_single_view_fallback(monkeypatch, ground_candidate):
    config = load_json("pixel_distance/camera_presets.json")
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    selected = [[950, 500], [1050, 500]]
    monkeypatch.setattr("pixel_distance.video_distance.auto_measure",
                        lambda *_args, **_kwargs: {"status": "unknown", "views": {}})
    calls = []

    def fake_ground(image_arg, points, model_dir, threshold):
        calls.append((model_dir, threshold, np.asarray(points).tolist()))
        return {"all_points_ground_candidates": ground_candidate,
                "point_checks": [{"status": "ground_candidate_unverified" if ground_candidate
                                  else "unknown"}] * 2}, np.zeros(image_arg.shape[:2], dtype=np.uint8)

    monkeypatch.setattr("pixel_distance.video_distance.segment_ground", fake_ground)
    result = measure_video_views({"D04": image, "D03": image}, "D04", selected,
                                 config, ground_model_dir="local-model",
                                 allow_assumed_plane=True)
    assert calls == [("local-model", 0.75, selected)]
    if ground_candidate:
        assert result["status"] == "single_view_assumed_plane_estimate"
        assert result["actual_distance_determined"] is False
    else:
        assert result["status"] == "unknown"
        assert result["distance_m"] is None
    with pytest.raises(ValueError, match="只支持地面"):
        measure_video_views({"D04": image}, "D04", selected, config,
                            plane_height_m=1, ground_model_dir="local-model")
    assert result["actual_distance_determined"] is False
