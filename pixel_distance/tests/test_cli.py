import json
import sys
from pathlib import Path

import pytest

from pixel_distance.cli import format_result, main, parser


def test_plane_estimate_requires_explicit_assumption():
    with pytest.raises(SystemExit):
        parser().parse_args(["estimate", "--image", "frame.png", "--camera", "D04"])
    args = parser().parse_args(["estimate", "--image", "frame.png", "--camera", "D04",
                                "--assume-same-plane"])
    assert args.assume_same_plane


def test_auto_3d_uses_lightweight_matcher_by_default():
    args = parser().parse_args(["auto-3d", "--reference", "D04"])
    assert args.matcher == "sift"
    assert args.trolley_origin_m is None
    runtime = parser().parse_args(["auto-3d", "--reference", "D04",
                                   "--trolley-origin-m", "3,4,0"])
    assert runtime.trolley_origin_m == (3.0, 4.0, 0.0)
    optional = parser().parse_args(["auto-3d", "--reference", "D04",
                                    "--matcher", "xfeat"])
    assert optional.matcher == "xfeat"


def test_help_text_is_chinese_without_changing_command_names():
    root = parser().format_help()
    assert "用法：pixel_distance" in root
    assert "子命令" in root
    assert "显示帮助信息并退出" in root
    assert "在同步多路画面中自动匹配所选两点" in root
    assert "usage:" not in root
    automatic = parser()._subparsers._group_actions[0].choices["auto-3d"].format_help()
    assert "用法：pixel_distance auto-3d" in automatic
    assert "当前两组独立配置会拒绝" in automatic
    assert "--trolley-origin-m" in automatic
    assert "--height-alignment" in automatic


def test_nominal_3d_direct_command_calculates_site_example(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    frames = root / "汇报材料" / "现场同刻画面_2026-11-05"
    if not all((frames / name).is_file() for name in (
            "02_主钩吊点_15-57-32.png", "04_左侧人员侦测_15-57-32.png")):
        pytest.skip("现场样例图片未随代码仓库提供")
    output = tmp_path / "distance.json"
    monkeypatch.setattr(sys, "argv", [
        "pixel_distance", "nominal-3d",
        "--reference", "D04", "--reference-image", str(frames / "02_主钩吊点_15-57-32.png"),
        "--target", "D03", "--target-image", str(frames / "04_左侧人员侦测_15-57-32.png"),
        "--point-a", "1205.4743,412.8686", "--point-b", "1200.3771,858.8686",
        "--target-point-a", "893.75,866.90", "--target-point-b", "402.85,871.87",
        "--out", str(output),
    ])
    main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "nominal_3d_candidate_not_calibrated"
    assert result["distance_m"] == pytest.approx(3.460739067, abs=1e-6)
    assert result["correspondence_source"] == "explicit_pixels"
    assert result["actual_distance_determined"] is False


def test_nominal_3d_rejection_reports_each_ray_gap():
    result = {"status": "unknown", "reason": "同名点视线未在容差内交会",
              "ray_gap_m": [0.263, 0.211], "max_ray_gap_m": 0.25}
    message = format_result("nominal-3d", result)
    assert "A 0.263 米" in message
    assert "B 0.211 米" in message
    assert "上限 0.250 米" in message


def test_nominal_commands_default_to_half_metre_ray_gap():
    nominal = parser().parse_args(["nominal-3d", "--reference", "D04",
                                   "--reference-image", "a.png", "--target", "D03",
                                   "--target-image", "b.png"])
    manual = parser().parse_args(["manual-nominal-3d", "--views", "views.json"])
    assert nominal.max_ray_gap_m == 0.5
    assert manual.max_ray_gap_m == 0.5
