"""Command-line and interactive image workflow for planar metric distance."""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np

from .auto3d import auto_measure, match_view
from .b7_motion import B7MotionDetector, analyze_b7
from .geometry import (calibrate_intrinsics, fit_plane, fit_pose, image_sha256,
                       load_image, load_json, measure_3d, measure_ground, save_json)
from .ground_segmentation import save_ground_overlay, segment_ground
from .group_height import compare_group_points
from .manual_nominal import estimate_manual_nominal
from .pose_state import (state_from_surveyed_poses, state_from_surveyed_group_poses,
                          state_from_vehicle_diagnostic,
                         update_pose_state)
from .preset import estimate_on_plane
from .reference_check import check_reference_length
from .temporal import motion_evidence
from .vehicle_anchor import estimate_cart_anchor, measure_on_cart_plane
from .video_distance import extract_views, measure_video_views, parse_instant


class ChineseArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, add_help=False, **kwargs)
        self._positionals.title = "位置参数"
        self._optionals.title = "选项"
        self.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")

    def format_help(self):
        return super().format_help().replace("usage: ", "用法：", 1)

    def format_usage(self):
        return super().format_usage().replace("usage: ", "用法：", 1)


def click_points(image, labels: list[str]) -> list[list[float]]:
    selected: list[list[float]] = []
    display = image.copy()
    max_width, max_height = 1400, 850
    scale = min(1.0, max_width / image.shape[1], max_height / image.shape[0])
    window_size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
    window = "Pixel distance: click in listed order; Backspace undo; Esc cancel"

    def redraw():
        nonlocal display
        display = image.copy()
        for i, (x, y) in enumerate(selected):
            cv2.circle(display, (round(x), round(y)), 6, (0, 0, 255), -1)
            cv2.putText(display, labels[i], (round(x) + 8, round(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
            cv2.putText(display, labels[i], (round(x) + 8, round(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        if scale < 1:
            display = cv2.resize(display, window_size, interpolation=cv2.INTER_AREA)
        cv2.imshow(window, display)

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(selected) < len(labels):
            selected.append([min(image.shape[1] - 1, x / scale), min(image.shape[0] - 1, y / scale)])
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, *window_size)
    cv2.setMouseCallback(window, on_mouse)
    redraw()
    try:
        announced_index = -1
        while len(selected) < len(labels):
            if len(selected) != announced_index:
                announced_index = len(selected)
                print(f"请点击 {labels[announced_index]} 点（{announced_index+1}/{len(labels)}）", flush=True)
            key = cv2.waitKey(100) & 0xFF
            if key == 27:
                raise KeyboardInterrupt("Point selection cancelled")
            if key in (8, 127) and selected:
                selected.pop()
                redraw()
    finally:
        cv2.destroyWindow(window)
    return selected


def pair(text: str) -> tuple[float, float]:
    try:
        x, y = text.split(",")
        return float(x), float(y)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("坐标格式应为 x,y") from exc


def triple(text: str) -> tuple[float, float, float]:
    try:
        x, y, z = text.split(",")
        return float(x), float(y), float(z)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("坐标格式应为 x,y,z，单位米") from exc


def format_result(command: str, result: dict) -> str:
    if command == "compare-height":
        if result["measurement_mode"] == "same_group_3d":
            return (f"组内三维距离：{result['distance_m']:.3f} 米；"
                    f"B 相对 A 的高度差：{result['height_difference_m']:+.3f} 米（待现场误差验收）。")
        low, high = result["height_difference_interval_m"]
        return (f"跨组高度差：{result['height_difference_m']:+.3f} 米；"
                f"校验误差范围 [{low:+.3f}, {high:+.3f}] 米。完整三维距离不可判定。")
    if command == "b7-motion":
        if result["status"] == "alarm_candidate":
            return "B7：大车与小车同时运行候选告警（已挂载；名义相机参数，待现场验收）。"
        if result["status"] == "no_simultaneous_motion":
            return "B7：未检测到持续的大车与小车同时运行。"
        if result["status"] in ("not_loaded", "unknown_load_state"):
            return f"B7：不触发；{result['reason']}。"
        return "B7：运动证据不足，无法判定。"
    if command == "segment-ground":
        checks = result["point_checks"]
        if not checks:
            return f"地面候选分割完成；候选像素占比 {result['floor_candidate_fraction']:.1%}（未现场验证）。"
        names = ("A", "B")
        details = "，".join(
            f"{names[index]} {'地面候选' if check['status'] == 'ground_candidate_unverified' else '无法确认地面'}"
            for index, check in enumerate(checks))
        return f"地面分割：{details}；模型结果未现场验证。"
    if command == "video-distance":
        if result.get("measurement_mode") == "cross_group_height_only":
            low, high = result["height_difference_interval_m"]
            return (f"跨组高度差：{result['height_difference_m']:+.3f} 米；"
                    f"校验误差范围 [{low:+.3f}, {high:+.3f}] 米。完整三维距离不可判定。")
        views = result.get("multiview_attempt", {}).get("views", {})
        details = []
        for camera_id, view in views.items():
            if view.get("status") == "excluded_cross_group":
                continue
            reason = view.get("reason", "")
            count = re.search(r"极线候选特征不足（(\d+/\d+)", reason)
            details.append(f"{camera_id} {count.group(1)}" if count
                           else f"{camera_id} 未匹配")
        note = ("\n名义极线检查通过数：" + "，".join(details) + "。") if details else ""
        if result["status"] == "single_view_assumed_plane_estimate":
            return (f"单目假定平面估算：{result['distance_m']:.3f} 米"
                    f"（{result['camera_id']}；假定两点均在距地 {result['plane_height_m']:g} 米的水平面；"
                    f"不是已确认的三维距离）；高度差无法确定。{note}")
        if result["status"] == "single_view_assumed_range_estimate":
            return (f"单目假定距离估算：{result['distance_m']:.3f} 米"
                    f"（假定两点距相机均为 {result['assumed_camera_range_m']:.2f} 米；"
                    f"不是已确认的三维距离）；高度差无法确定。{note}")
        if result["status"] == "unknown":
            checks = result.get("ground_segmentation", {}).get("point_checks", [])
            ground = ("\n地面分割：" + "，".join(
                f"{name} {'候选' if check['status'] == 'ground_candidate_unverified' else '未确认'}"
                for name, check in zip(("A", "B"), checks)) + "。") if checks else ""
            return f"无法给出距离：{result['reason']}。{ground}"
        height = result.get("height_difference_m")
        height_text = f"；B 相对 A 的高度差：{height:+.3f} 米" if height is not None else ""
        return (f"多视角名义三维估算：{result['distance_m']:.3f} 米{height_text}"
                f"（辅助视角：{', '.join(result.get('metric_pairs', []))}；参数未标定）")
    if command == "nominal-3d":
        gaps = result.get("ray_gap_m")
        gap_text = (f"\n同名点视线差：A {gaps[0]:.3f} 米，B {gaps[1]:.3f} 米；"
                    f"上限 {result['max_ray_gap_m']:.3f} 米。") if gaps else ""
        if result["status"] == "unknown":
            return f"无法计算三维估算距离：{result['reason']}。{gap_text}"
        source = "自动对应" if result["correspondence_source"].startswith("auto_") else "指定对应"
        return (f"三维估算距离：{result['distance_m']:.3f} 米"
                f"（{source}；使用可修改的名义参数，非实测真值）{gap_text}")
    if command == "manual-nominal-3d":
        if result["status"] == "unknown":
            return f"无法计算名义三维距离：{result['reason']}。"
        message = (f"条件试算：{result['distance_m']:.2f} 米；真实距离尚无法确定。"
                   f"\n跨组位置未实测，两点最大视线差 {max(result['ray_gap_m']):.2f} 米。")
        check = result.get("reference_length_check")
        if check:
            lower, upper = check["range_m"]
            verdict = "一致" if check["status"] == "within_range" else "不符"
            message += f"\n若 A/B 为该参照物的同名端点：与已知 {lower:g}–{upper:g} 米范围{verdict}。"
        return message
    if command == "auto-3d":
        if result.get("measurement_mode") == "cross_group_height_only":
            low, high = result["height_difference_interval_m"]
            return (f"跨组高度差：{result['height_difference_m']:+.3f} 米；"
                    f"校验误差范围 [{low:+.3f}, {high:+.3f}] 米。完整三维距离不可判定。")
        if result.get("measurement_mode") in ("same_group_3d", "same_group_nominal_3d"):
            return (f"{result['metric_group']} 组内三维候选距离：{result['distance_m']:.3f} 米；"
                    f"B 相对 A 高度差：{result['height_difference_m']:+.3f} 米"
                    "（仍需现场精度验收）。")
        labels = []
        for camera_id, view in result["views"].items():
            if view["status"] == "matched_no_scale":
                labels.append(f"{camera_id} 已匹配/缺少本帧跨组位置")
            elif view["status"] == "matched_candidate":
                labels.append(f"{camera_id} 已匹配" if "metric" in view else f"{camera_id} 已匹配/未通过三维检查")
            elif view["status"] == "partial_coverage":
                count = sum(check.get("status") == "matched_candidate"
                            for check in view.get("point_checks", []))
                labels.append(f"{camera_id} 部分匹配（{count}/2 点）")
            else:
                reason = view.get("reason", "")
                if "Too few symmetric feature matches" in reason:
                    count = reason.rsplit("(", 1)[-1].rstrip(")")
                    labels.append(f"{camera_id} 未匹配（双向特征仅 {count} 组）")
                else:
                    labels.append(f"{camera_id} 未匹配")
        overview = "，".join(labels)
        if result["status"] == "unknown":
            message = f"无法可靠计算三维距离。\n视角结果：{overview}。"
        else:
            if result.get("scale_source") == "current_pose_state":
                message = (f"当前位姿三维候选距离：{result['distance_m']:.3f} 米"
                           f"（{result['pair_count']} 路辅助相机；精度未验收）"
                           f"\n视角结果：{overview}。")
                if result.get("pose_state_method") == "vehicle_length_diagnostic_only":
                    message += "\n当前位姿来自车长诊断，仅供离线验证。"
            else:
                message = (f"三维候选距离：{result['distance_m']:.3f} 米"
                           f"（{result['pair_count']} 个相机对；不假定地面；未标定/未验收）"
                           f"\n视角结果：{overview}。")
        if "temporal" in result:
            wording = {"background_consistent": "随背景运动", "background_deviation": "偏离背景运动",
                       "unknown": "无法跟踪"}
            point_status = result["temporal"].get("points", {})
            message += "\n连续帧：" + "，".join(
                f"{name} {wording.get(point_status.get(name, {}).get('status'), '无法判定')}"
                for name in ("A", "B")) + "（不据此判定离地高度）。"
        return message
    if command == "estimate":
        a, b = result["pixel_points"]
        plane = result["plane_height_m"]
        surface = "地面" if plane == 0 else f"距地 {plane:g} 米的水平面"
        return (f"A({a[0]:.1f}, {a[1]:.1f})  B({b[0]:.1f}, {b[1]:.1f})\n"
                f"估算距离：{result['distance_m']:.3f} 米"
                f"（{result['camera_id']}；假定两点均在{surface}；参数未标定）")
    if command == "measure":
        return f"地面距离：{result['distance_m']:.3f} 米（独立检查点最大误差：{result['calibration_check_max_m']:.3f} 米）"
    if command == "measure-3d":
        return f"三维距离：{result['distance_m']:.3f} 米（画面时间差：{result['frame_time_offset_ms']:.1f} 毫秒）"
    if command == "calibrate-intrinsics":
        return f"内参标定完成：{len(result['used_images'])} 张图，重投影均方根误差 {result['reprojection_rms_px']:.2f} 像素。"
    if command == "calibrate-plane":
        stats = result["diagnostics"]
        return f"地面标定完成：{stats['control_count']} 个控制点，{stats['check_count']} 个检查点；检查点最大误差 {stats['check_max_m']:.3f} 米。"
    if command == "calibrate-pose":
        stats = result["diagnostics"]
        return f"相机位姿标定完成：检查点最大重投影误差 {stats['check_max_px']:.2f} 像素。"
    if command == "vehicle-diagnostic":
        if "surface_measure" not in result:
            return f"离线车体诊断完成：名义相机基线 {result['anchor']['nominal_baseline_m']:.2f} 米（未验收）。"
        measure = result["surface_measure"]
        lower, upper = measure["length_only_range_m"]
        return (f"离线车顶条件距离：{measure['distance_m']:.2f} 米"
                f"（仅车长先验换算 {lower:.2f}–{upper:.2f} 米；非独立测量）。")
    if command == "reference-check":
        verdict = "范围内" if result["comparison"] == "within_range" else "范围外"
        return (f"离线参照物核对：{result['reference_name']}，候选距离 "
                f"{result['candidate_distance_m']:.3f} 米，位于已知长度{verdict}；"
                "不修改运行时参数。")
    if command == "init-pose-state":
        return f"共享坐标系状态已建立：{len(result['cameras'])} 路相机；仍需独立精度验收。"
    if command == "init-height-state":
        return "两组独立测高状态已建立：各 2 路；跨组比较仍需离线竖直基准文件。"
    if command == "track-pose":
        return f"相机位姿已更新：{len(result['updated_cameras'])} 路；连续跟踪存在累计漂移。"
    raise ValueError(f"Unsupported command: {command}")


def save_annotated(image, a, b, distance_m: float, path: str,
                   label: str = "Ground") -> None:
    canvas = image.copy()
    start, end = tuple(map(round, a)), tuple(map(round, b))
    cv2.line(canvas, start, end, (0, 255, 255), 3)
    for point, point_label in ((start, "A"), (end, "B")):
        cv2.circle(canvas, point, 8, (0, 0, 255), -1)
        cv2.putText(canvas, point_label, (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    message = f"{label}: {distance_m:.3f} m"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    (text_width, text_height), baseline = cv2.getTextSize(message, font, font_scale, 2)
    middle_x = (start[0] + end[0]) // 2
    lower_y = max(start[1], end[1]) + text_height + 36
    upper_y = min(start[1], end[1]) - 26
    text_y = lower_y if lower_y + baseline + 8 < image.shape[0] else upper_y
    text_x = max(8, min(image.shape[1] - text_width - 8, middle_x - text_width // 2))
    cv2.rectangle(canvas, (text_x - 5, text_y - text_height - 6),
                  (text_x + text_width + 5, text_y + baseline + 6), (20, 20, 20), -1)
    cv2.putText(canvas, message, (text_x, text_y), font, font_scale,
                (255, 255, 255), 2)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg"):
        raise ValueError("Annotated output must be .png or .jpg")
    ok, encoded = cv2.imencode(suffix, canvas)
    if not ok:
        raise ValueError("Could not encode annotated output image")
    encoded.tofile(str(target))


def save_matched_points(image, points, path: str, label: str,
                        markers=None) -> None:
    canvas = image.copy()
    if markers is not None:
        quad = [markers[index] for index in (0, 1, 3, 2)]
        cv2.polylines(canvas, [np.asarray(quad, dtype=np.int32)],
                      True, (0, 255, 255), 2)
        for index, point in enumerate(markers, 1):
            pixel = tuple(map(round, point))
            cv2.circle(canvas, pixel, 6, (0, 255, 255), -1)
            cv2.putText(canvas, f"M{index}", (pixel[0] + 8, pixel[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for point, name in zip(points, ("A", "B")):
        if point is None:
            continue
        pixel = tuple(map(round, point))
        cv2.circle(canvas, pixel, 8, (0, 0, 255), -1)
        cv2.putText(canvas, name, (pixel[0] + 10, pixel[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(canvas, label, (20, image.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(canvas, label, (20, image.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise ValueError("Could not encode matched-point image")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(target))


def parser() -> argparse.ArgumentParser:
    root = ChineseArgumentParser(prog="pixel_distance", description="根据相机标定或预设参数估算图像中两点的实际距离")
    sub = root.add_subparsers(dest="command", required=True, title="子命令", metavar="命令")
    intrinsic = sub.add_parser("calibrate-intrinsics", help="用棋盘格图像标定镜头内参")
    intrinsic.add_argument("--images", required=True, help="原始全幅棋盘格图像路径，支持通配符")
    intrinsic.add_argument("--corners", required=True, type=pair, help="棋盘格内角点列数,行数")
    intrinsic.add_argument("--square-m", required=True, type=float, help="实测棋盘格边长，单位米")
    intrinsic.add_argument("--model", choices=["pinhole", "fisheye"], required=True,
                           help="镜头模型：普通针孔镜头或鱼眼镜头")
    intrinsic.add_argument("--out", required=True, help="输出内参 JSON 文件")
    plane = sub.add_parser("calibrate-plane", help="用实测地面控制点标定地面映射")
    plane.add_argument("--image", required=True, help="标定用原始图像")
    plane.add_argument("--points", required=True, help="控制点 JSON，包含 id、xy_m 和可选的 pixel")
    plane.add_argument("--intrinsics", help="与图像尺寸一致的内参 JSON")
    plane.add_argument("--ransac-threshold-m", type=float, default=0.05,
                       help="RANSAC 地面拟合阈值，单位米")
    plane.add_argument("--max-check-error-m", type=float, required=True,
                       help="现场允许的独立检查点最大误差，单位米")
    plane.add_argument("--out", required=True, help="输出地面标定 JSON 文件")
    pose = sub.add_parser("calibrate-pose", help="用实测三维控制点标定当前画面相机位姿")
    pose.add_argument("--image", required=True, help="标定用原始图像")
    pose.add_argument("--points", required=True, help="控制点 JSON，包含 id、xyz_m 和可选的 pixel")
    pose.add_argument("--intrinsics", required=True, help="内参 JSON 文件")
    pose.add_argument("--max-check-error-px", type=float, required=True,
                      help="允许的独立检查点最大重投影误差，单位像素")
    pose.add_argument("--out", required=True, help="输出相机位姿 JSON 文件")
    measure = sub.add_parser("measure", help="测量两个已确认位于地面的点")
    measure.add_argument("--image", required=True, help="待测原始图像")
    measure.add_argument("--calibration", required=True, help="该图像的地面标定 JSON")
    measure.add_argument("--surface", choices=["ground"], required=True,
                         help="确认两个物理点均位于已标定地面，取值 ground")
    measure.add_argument("--point-a", type=pair, help="A 点原图像素坐标 x,y；省略时可点击")
    measure.add_argument("--point-b", type=pair, help="B 点原图像素坐标 x,y；省略时可点击")
    measure.add_argument("--allow-extrapolation", action="store_true",
                         help="允许在控制点覆盖范围外估算，仅供核验")
    measure.add_argument("--out", help="输出详细结果 JSON 文件")
    measure.add_argument("--annotated", help="输出带点位和距离的 PNG/JPG 图片")
    estimate = sub.add_parser("estimate", help="用名义相机参数估算同一水平面上两点的距离")
    estimate.add_argument("--image", required=True, help="单路相机未裁剪的原始画面")
    estimate.add_argument("--camera", required=True, choices=["D01", "D02", "D03", "D04"],
                          help="相机编号")
    estimate.add_argument("--config", default=str(Path(__file__).with_name("camera_presets.json")),
                          help="相机预设参数 JSON 文件")
    estimate.add_argument("--assume-same-plane", action="store_true", required=True,
                          help="明确确认两个物理点位于同一水平面")
    estimate.add_argument("--plane-height-m", type=float, default=0.0,
                          help="两点所在水平面的离地高度，单位米；默认 0（地面）")
    estimate.add_argument("--point-a", type=pair, help="A 点原图像素坐标 x,y；省略时可点击")
    estimate.add_argument("--point-b", type=pair, help="B 点原图像素坐标 x,y；省略时可点击")
    estimate.add_argument("--out", help="输出详细结果 JSON 文件")
    estimate.add_argument("--annotated", help="输出带点位和距离的 PNG/JPG 图片")
    spatial = sub.add_parser("measure-3d", help="根据同步画面的对应点和已标定相机位姿计算三维距离")
    spatial.add_argument("--views", required=True,
                         help="视角 JSON，包含 image、pose、timestamp_ms 和可选的 pixels")
    spatial.add_argument("--max-time-offset-ms", required=True, type=float,
                         help="允许的最大画面时间差，单位毫秒")
    spatial.add_argument("--max-reprojection-error-px", required=True, type=float,
                         help="允许的最大重投影误差，单位像素")
    spatial.add_argument("--min-ray-angle-deg", default=1.0, type=float,
                         help="最小视线交会角，单位度；默认 1")
    spatial.add_argument("--out", help="输出详细结果 JSON 文件")
    height = sub.add_parser("compare-height", help="两点各由同组双目定位；跨组仅比较已对齐的高度")
    height.add_argument("--observations", required=True,
                        help="A/B 各两路同组同名像素及原图路径的 JSON，含秒级 time")
    height.add_argument("--state", required=True, help="与本秒图像哈希一致的现场标定位姿状态")
    height.add_argument("--alignment", help="跨组时必需：离线验证的高度零点偏移与误差上界 JSON")
    height.add_argument("--config", default=str(Path(__file__).with_name("camera_presets.json")))
    height.add_argument("--out", help="输出详细结果 JSON 文件")
    automatic = sub.add_parser("auto-3d", help="在同步多路画面中自动匹配所选两点并尝试三维测距")
    automatic.add_argument("--reference", required=True, choices=["D01", "D02", "D03", "D04"],
                           help="选点所在的参考相机编号")
    for camera_id in ("d01", "d02", "d03", "d04"):
        automatic.add_argument(f"--{camera_id}", help=f"{camera_id.upper()} 相机的原始全幅画面")
    automatic.add_argument("--point-a", type=pair, help="A 点原图像素坐标 x,y；省略时可点击")
    automatic.add_argument("--point-b", type=pair, help="B 点原图像素坐标 x,y；省略时可点击")
    automatic.add_argument("--config", default=str(Path(__file__).with_name("camera_presets.json")),
                           help="相机预设参数 JSON 文件")
    automatic.add_argument("--trolley-origin-m", type=triple,
                            help="仅旧四机固定试验配置可用；当前两组独立配置会拒绝")
    automatic.add_argument("--pose-state", help="本秒相机位姿状态 JSON；两组可使用独立水平原点")
    automatic.add_argument("--height-alignment", help="可选：已验证的跨组竖直高度基准 JSON")
    automatic.add_argument("--allow-diagnostic-pose", action="store_true",
                           help="仅离线试验：允许使用车长诊断生成的非验收位姿")
    automatic.add_argument("--matcher", choices=["auto", "sift", "loftr", "xfeat", "patch", "roma"], default="sift",
                           help="匹配方法；patch 用已标定位姿沿极线比较图像块，xfeat 需本地官方模型文件")
    automatic.add_argument("--out", help="输出详细诊断结果 JSON 文件")
    automatic.add_argument("--annotated-dir", help="输出各视角匹配点标注图的目录")
    automatic.add_argument("--later-reference", help="参考相机的较晚原始帧，用于检查两点运动")
    video = sub.add_parser("video-distance", help="按同一秒自动截取四路视频，匹配参考画面的两个点并估算距离")
    video.add_argument("--videos-dir", default=".", help="D01–D04 视频所在目录，默认当前目录")
    for camera_id in ("d01", "d02", "d03", "d04"):
        video.add_argument(f"--{camera_id}", help=f"可选：显式指定 {camera_id.upper()} 视频路径")
    video.add_argument("--time", required=True, help="目标画面时间 YYYY-MM-DDTHH:MM:SS")
    video.add_argument("--reference", required=True, choices=["D01", "D02", "D03", "D04"],
                       help="只在该路画面选择 A/B")
    video.add_argument("--point-a", type=pair, help="参考画面 A 点原始视频像素 x,y；省略时点击")
    video.add_argument("--point-b", type=pair, help="参考画面 B 点原始视频像素 x,y；省略时点击")
    video.add_argument("--matcher", choices=["auto", "sift", "loftr", "xfeat", "patch", "roma"],
                       default="auto", help="跨视角自动匹配器；默认先尝试 sift，失败时尝试 loftr")
    video.add_argument("--fallback-plane-height-m", type=float, default=0.0,
                        help="显式启用单目平面假设时的平面离地高度，默认地面 0 米")
    video.add_argument("--assume-same-plane", action="store_true",
                       help="仅离线试算：组内三维失败时，允许假定 A/B 在同一水平面")
    video.add_argument("--ground-model", help="可选：本地地面分割模型目录；未确认两个地面候选时禁止单目地面回退")
    video.add_argument("--ground-threshold", type=float, default=0.75,
                       help="地面候选概率阈值，默认 0.75；不是已验证精度")
    video.add_argument("--config", default=str(Path(__file__).with_name("camera_presets.json")),
                       help="相机参数与各视频时间偏移 JSON")
    video.add_argument("--pose-state", help="可选：与抽取帧绑定的组内相机位姿状态 JSON")
    video.add_argument("--height-alignment", help="可选：已验证的跨组竖直高度基准 JSON")
    video.add_argument("--out", help="保存完整结果 JSON")
    video.add_argument("--frames-out", help="可选：保存本次从四路视频抽取的原始帧")
    b7 = sub.add_parser("b7-motion", help="在外部确认挂载后，用四路相机短时运动判断大车和小车是否同时运行")
    b7.add_argument("--videos-dir", default=".", help="D01–D04 视频目录，默认当前目录")
    for camera_id in ("d01", "d02", "d03", "d04"):
        b7.add_argument(f"--{camera_id}", help=f"可选：显式指定 {camera_id.upper()} 视频路径")
    b7.add_argument("--time", required=True, help="检测中心时间 YYYY-MM-DDTHH:MM:SS")
    b7.add_argument("--load-state", required=True,
                    choices=["confirmed", "not-loaded", "unknown"],
                    help="由外部模块提供挂载状态；只有 confirmed 才执行运动分析")
    b7.add_argument("--window-radius-s", type=float, default=3.0,
                    help="中心时间前后各取视频秒数，默认各 3 秒")
    b7.add_argument("--sample-step-s", type=float, default=0.2,
                    help="相邻采样画面的最大秒数，默认 0.2 秒")
    b7.add_argument("--min-motion-duration-s", type=float, default=3.0,
                    help="大车和小车同向持续运行的最短秒数，默认 3 秒")
    b7.add_argument("--min-speed-mps", type=float, default=0.08,
                    help="各轴运动速度阈值，默认 0.08 米/秒，需现场调整")
    b7.add_argument("--max-group-disagreement-mps", type=float, default=0.12,
                    help="同组相机速度差上限，默认 0.12 米/秒，需现场调整")
    b7.add_argument("--ground-mask-dir",
                    help="可选：D01.png 至 D04.png 静态地面候选掩码目录；白色为候选")
    b7.add_argument("--config", default=str(Path(__file__).with_name("camera_presets.json")),
                    help="相机名义参数和大车/小车方向配置")
    b7.add_argument("--out", help="输出完整运动诊断 JSON")
    ground = sub.add_parser("segment-ground", help="对原图生成地面候选分割和选点诊断")
    ground.add_argument("--image", required=True, help="原始全幅图片")
    ground.add_argument("--model", required=True, help="本地 SegFormer ADE20K 模型目录")
    ground.add_argument("--point-a", type=pair, help="可选：A 点原图像素 x,y")
    ground.add_argument("--point-b", type=pair, help="可选：B 点原图像素 x,y")
    ground.add_argument("--threshold", type=float, default=0.75,
                        help="地面候选概率阈值，默认 0.75")
    ground.add_argument("--overlay", help="可选：输出绿色地面候选叠加图 PNG/JPG")
    ground.add_argument("--out", help="可选：输出详细诊断 JSON")
    nominal = sub.add_parser("nominal-3d", help="用可修改的假定相机位姿测量两路图像中任意高度的对应点")
    nominal.add_argument("--reference", required=True, choices=["D01", "D02", "D03", "D04"])
    nominal.add_argument("--reference-image", required=True, help="选点所在的原始全幅图像")
    nominal.add_argument("--target", required=True, choices=["D01", "D02", "D03", "D04"])
    nominal.add_argument("--target-image", required=True, help="另一视角同秒原始全幅图像")
    nominal.add_argument("--point-a", type=pair, help="参考图 A 点 x,y；省略时点击")
    nominal.add_argument("--point-b", type=pair, help="参考图 B 点 x,y；省略时点击")
    nominal.add_argument("--target-point-a", type=pair, help="另一视角同一物理 A 点 x,y")
    nominal.add_argument("--target-point-b", type=pair, help="另一视角同一物理 B 点 x,y")
    nominal.add_argument("--click-target", action="store_true", help="自动匹配失败时在另一视角点击 A、B")
    nominal.add_argument("--matcher", choices=["auto", "sift", "loftr", "xfeat"], default="sift")
    nominal.add_argument("--config", default=str(Path(__file__).with_name("camera_presets.json")),
                         help="含 nominal_world_poses 的可修改假定参数 JSON")
    nominal.add_argument("--max-ray-gap-m", type=float, default=0.5,
                         help="同名视线最大间距，默认 0.5 米")
    nominal.add_argument("--out", help="输出详细结果 JSON")
    manual = sub.add_parser("manual-nominal-3d", help="用手工同名点和假定相机位姿估算三维距离")
    manual.add_argument("--views", required=True, help="两路图像、同名点、假定相机中心及朝向的 JSON 文件")
    manual.add_argument("--config", default=str(Path(__file__).with_name("camera_presets.json")),
                        help="相机预设参数 JSON 文件")
    manual.add_argument("--max-ray-gap-m", type=float, default=0.5,
                        help="两路视线允许的最大间距，单位米；默认 0.5")
    manual.add_argument("--max-time-offset-ms", type=float,
                        help="若两路 JSON 含曝光时间戳，允许的最大时间差，单位毫秒")
    manual.add_argument("--known-length-range-m", type=pair,
                        help="若 A/B 正是已知长度的两个端点，输入参考长度范围 最小值,最大值（米）用于独立核对")
    manual.add_argument("--out", help="输出详细诊断结果 JSON 文件")
    vehicle = sub.add_parser("vehicle-diagnostic", help="离线用运输车四标记检查相机几何，不作为运行时尺度来源")
    vehicle.add_argument("--reference", required=True, choices=["D01", "D02", "D03", "D04"])
    vehicle.add_argument("--target", required=True, choices=["D01", "D02", "D03", "D04"])
    vehicle.add_argument("--reference-image", required=True)
    vehicle.add_argument("--target-image", required=True)
    vehicle.add_argument("--length-range-m", type=pair, required=True,
                         help="运输车前后总长的近似范围 最小值,最大值（米）")
    vehicle.add_argument("--point-a", type=pair, help="可选：参考图 A 点 x,y")
    vehicle.add_argument("--point-b", type=pair, help="可选：参考图 B 点 x,y")
    vehicle.add_argument("--config", default=str(Path(__file__).with_name("camera_presets.json")))
    vehicle.add_argument("--state-out", help="仅离线试验：输出车长先验的诊断位姿状态")
    vehicle.add_argument("--annotated-dir", help="输出车体标记及投影点检查图目录")
    vehicle.add_argument("--out", help="输出详细诊断 JSON")
    reference_check = sub.add_parser("reference-check", help="用任意已知长度物体离线检查测距偏差，不修改运行时尺度")
    reference_check.add_argument("--measurement", required=True, help="待核对的测距结果 JSON")
    reference_check.add_argument("--reference-name", required=True, help="参照物名称")
    reference_check.add_argument("--length-range-m", type=pair, required=True,
                                 help="独立获知的长度范围 最小值,最大值（米）")
    reference_check.add_argument("--out", help="输出离线核对结果 JSON")
    initialize = sub.add_parser("init-pose-state", help="用共同世界坐标的现场标定结果建立运行时位姿状态")
    initialize.add_argument("--views", required=True, help="相机 image/pose 列表 JSON")
    initialize.add_argument("--ground-z-m", type=float, required=True,
                            help="共同世界坐标系中的地面标高，单位米")
    initialize.add_argument("--out", required=True, help="输出初始位姿状态 JSON")
    height_state = sub.add_parser("init-height-state", help="用两组各自的现场标定建立独立测高状态")
    height_state.add_argument("--views", required=True, help="四路本秒 image 与离线组内 pose 的列表 JSON")
    height_state.add_argument("--config", default=str(Path(__file__).with_name("camera_presets.json")))
    height_state.add_argument("--out", required=True, help="输出本秒两组位姿状态 JSON")
    tracking = sub.add_parser("track-pose", help="由相邻帧的静态地面特征更新相机位姿")
    tracking.add_argument("--state", required=True, help="上一次的位姿状态 JSON")
    tracking.add_argument("--frames", required=True, help="每路相机 previous/current 路径列表 JSON")
    tracking.add_argument("--max-step-m", type=float, default=2.0,
                          help="相邻帧相机允许的最大位移，默认 2 米")
    tracking.add_argument("--out", required=True, help="输出更新后的位姿状态 JSON")
    for command in (intrinsic, plane, pose, measure, estimate, spatial, height, automatic, video, b7, ground, nominal, manual,
                     vehicle, reference_check, initialize, height_state, tracking):
        command.add_argument("--json", action="store_true", help="在终端打印完整 JSON，而非简短中文结果")
    return root


def main() -> None:
    args = parser().parse_args()
    try:
        if args.command == "calibrate-intrinsics":
            files = sorted(glob.glob(args.images))
            result = calibrate_intrinsics(files, tuple(map(int, args.corners)), args.square_m, args.model)
            save_json(args.out, result)
        elif args.command == "calibrate-plane":
            image = load_image(args.image)
            data = load_json(args.points)
            anchors = data["points"]
            missing = [p for p in anchors if p.get("pixel") is None]
            if missing:
                clicks = click_points(image, [p["id"] for p in missing])
                for point, pixel in zip(missing, clicks):
                    point["pixel"] = pixel
            intrinsic = load_json(args.intrinsics) if args.intrinsics else None
            result = fit_plane((image.shape[1], image.shape[0]), anchors, intrinsic,
                               args.ransac_threshold_m, args.max_check_error_m)
            result["calibration_image_sha256"] = image_sha256(args.image)
            save_json(args.out, result)
        elif args.command == "calibrate-pose":
            image = load_image(args.image)
            anchors = load_json(args.points)["points"]
            missing = [p for p in anchors if p.get("pixel") is None]
            if missing:
                clicks = click_points(image, [p["id"] for p in missing])
                for point, pixel in zip(missing, clicks):
                    point["pixel"] = pixel
            result = fit_pose((image.shape[1], image.shape[0]), anchors,
                              load_json(args.intrinsics), args.max_check_error_px)
            result["calibration_image_sha256"] = image_sha256(args.image)
            save_json(args.out, result)
        elif args.command == "measure":
            image = load_image(args.image)
            calibration = load_json(args.calibration)
            if calibration.get("calibration_image_sha256") != image_sha256(args.image):
                raise ValueError("Ground calibration belongs to a different image; fit control points again for this camera pose")
            if (args.point_a is None) != (args.point_b is None):
                raise ValueError("Supply both --point-a and --point-b, or neither to click")
            a, b = (args.point_a, args.point_b) if args.point_a else click_points(image, ["A", "B"])
            result = measure_ground(calibration, (image.shape[1], image.shape[0]), a, b,
                                    args.allow_extrapolation)
            if args.out:
                save_json(args.out, result)
            if args.annotated:
                save_annotated(image, a, b, result["distance_m"], args.annotated)
        elif args.command == "estimate":
            image = load_image(args.image)
            if (args.point_a is None) != (args.point_b is None):
                raise ValueError("Supply both --point-a and --point-b, or neither to click")
            a, b = (args.point_a, args.point_b) if args.point_a else click_points(image, ["A", "B"])
            result = estimate_on_plane(load_json(args.config), args.camera,
                                       (image.shape[1], image.shape[0]), a, b,
                                       args.plane_height_m)
            if args.out:
                save_json(args.out, result)
            if args.annotated:
                save_annotated(image, a, b, result["distance_m"], args.annotated,
                               label="NOMINAL plane")
        elif args.command == "b7-motion":
            if (not math.isfinite(args.window_radius_s) or args.window_radius_s <= 0
                    or not math.isfinite(args.sample_step_s) or args.sample_step_s <= 0
                    or args.sample_step_s > 0.5):
                raise ValueError("时间窗和采样间隔须为正；采样间隔不得超过 0.5 秒")
            config = load_json(args.config)
            instant = parse_instant(args.time)
            loaded = {"confirmed": True, "not-loaded": False,
                      "unknown": None}[args.load_state]
            if loaded is not True:
                result = analyze_b7([], loaded, config)
            else:
                directory = Path(args.videos_dir)
                videos = {}
                for camera_id in ("D01", "D02", "D03", "D04"):
                    explicit = getattr(args, camera_id.lower())
                    if explicit:
                        videos[camera_id] = explicit
                        continue
                    candidates = [path for path in directory.glob(f"{camera_id}_*")
                                  if path.suffix.lower() in (".mp4", ".avi", ".mkv")]
                    if len(candidates) != 1:
                        raise ValueError(f"{camera_id} 需恰好有一个视频，或显式传 --{camera_id.lower()}")
                    videos[camera_id] = candidates[0]
                masks = {}
                if args.ground_mask_dir:
                    for camera_id in ("D01", "D02", "D03", "D04"):
                        mask_path = Path(args.ground_mask_dir) / f"{camera_id}.png"
                        if not mask_path.is_file():
                            raise ValueError(f"缺少地面候选掩码：{mask_path}")
                        masks[camera_id] = cv2.cvtColor(load_image(mask_path), cv2.COLOR_BGR2GRAY)
                interval_count = math.ceil(2 * args.window_radius_s / args.sample_step_s)
                actual_step = 2 * args.window_radius_s / interval_count
                detector = B7MotionDetector(
                    config, masks, args.min_speed_mps,
                    args.max_group_disagreement_mps, args.min_motion_duration_s,
                    2 * args.window_radius_s, max(0.5, actual_step * 1.5))
                metadata = {}
                for index in range(interval_count + 1):
                    offset_s = -args.window_radius_s + index * actual_step
                    moment = instant + timedelta(seconds=offset_s)
                    images, details = extract_views(videos, moment, config)
                    metadata[moment.isoformat()] = details
                    if len(images) != 4:
                        missing = [name for name in ("D01", "D02", "D03", "D04")
                                   if name not in images]
                        raise ValueError(f"{moment.isoformat()} 缺少相机帧：{', '.join(missing)}")
                    result = detector.update(float(index * actual_step), images, loaded)
                result["video_frames"] = metadata
                result["sample_step_s"] = actual_step
            result["requested_time"] = instant.isoformat()
            if args.out:
                save_json(args.out, result)
        elif args.command == "video-distance":
            if (args.point_a is None) != (args.point_b is None):
                raise ValueError("A/B 像素坐标必须同时提供或同时省略")
            directory = Path(args.videos_dir)
            videos = {}
            for camera_id in ("D01", "D02", "D03", "D04"):
                explicit = getattr(args, camera_id.lower())
                if explicit:
                    videos[camera_id] = explicit
                    continue
                candidates = [path for path in directory.glob(f"{camera_id}_*")
                              if path.suffix.lower() in (".mp4", ".avi", ".mkv")]
                if len(candidates) != 1:
                    raise ValueError(f"{camera_id} 在 {directory} 中需恰好有一个视频，或显式传 --{camera_id.lower()}")
                videos[camera_id] = candidates[0]
            config = load_json(args.config)
            instant = parse_instant(args.time)
            images, frames = extract_views(videos, instant, config)
            if args.reference not in images:
                raise ValueError(frames[args.reference]["reason"])
            if args.frames_out:
                frame_dir = Path(args.frames_out)
                frame_dir.mkdir(parents=True, exist_ok=True)
                for camera_id, image in images.items():
                    destination = frame_dir / f"{camera_id}_{instant:%Y%m%d_%H%M%S}.png"
                    ok, encoded = cv2.imencode(".png", image)
                    if not ok:
                        raise ValueError(f"无法编码 {camera_id} 截帧")
                    encoded.tofile(str(destination))
                    frames[camera_id]["saved_frame"] = str(destination.resolve())
            selected = (args.point_a, args.point_b) if args.point_a else click_points(
                images[args.reference], ["A", "B"])
            result = measure_video_views(images, args.reference, selected, config,
                                         args.matcher, args.fallback_plane_height_m,
                                         args.ground_model, args.ground_threshold,
                                         args.assume_same_plane,
                                         load_json(args.pose_state) if args.pose_state else None,
                                         load_json(args.height_alignment) if args.height_alignment else None)
            result["requested_time"] = instant.isoformat()
            result["video_frames"] = frames
            if args.out:
                save_json(args.out, result)
        elif args.command == "segment-ground":
            if (args.point_a is None) != (args.point_b is None):
                raise ValueError("A/B 像素坐标必须同时提供或同时省略")
            image = load_image(args.image)
            selected = [args.point_a, args.point_b] if args.point_a is not None else None
            result, mask = segment_ground(image, selected, args.model, args.threshold)
            if args.out:
                save_json(args.out, result)
            if args.overlay:
                save_ground_overlay(image, mask, args.overlay, selected)
        elif args.command == "nominal-3d":
            if args.reference == args.target:
                raise ValueError("参考相机和辅助相机必须不同")
            if (args.point_a is None) != (args.point_b is None):
                raise ValueError("参考图 A/B 点需同时提供或同时省略")
            if (args.target_point_a is None) != (args.target_point_b is None):
                raise ValueError("辅助图 A/B 点需同时提供或同时省略")
            config = load_json(args.config)
            poses = config["nominal_world_poses"]
            reference = load_image(args.reference_image)
            target = load_image(args.target_image)
            selected = (args.point_a, args.point_b) if args.point_a else click_points(reference, ["A", "B"])
            if args.target_point_a is not None:
                transferred = [args.target_point_a, args.target_point_b]
                source = "explicit_pixels"
            else:
                try:
                    matched = match_view(reference, target, np.asarray(selected),
                                         config["cameras"][args.reference],
                                         config["cameras"][args.target], args.matcher)
                    transferred = matched["pixel_points"]
                    source = f"auto_{matched['matcher']}"
                except (ValueError, cv2.error, ImportError, RuntimeError) as exc:
                    if not args.click_target:
                        result = {"status": "unknown", "reason": f"自动同名点匹配失败：{exc}",
                                  "reference_pixels": [list(p) for p in selected]}
                        if args.out:
                            save_json(args.out, result)
                        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json
                              else format_result(args.command, result))
                        return
                    transferred = click_points(target, ["A", "B"])
                    source = "interactive_click"
            views = []
            for camera_id, image, pixels in ((args.reference, reference, selected),
                                             (args.target, target, transferred)):
                pose = poses[camera_id]
                views.append({"camera_id": camera_id, "image_size": [image.shape[1], image.shape[0]],
                              "pixels": pixels, "center_m": pose["center_m"],
                              "azimuth_deg": pose["azimuth_deg"]})
            result = estimate_manual_nominal(views, config, args.max_ray_gap_m)
            result["correspondence_source"] = source
            result["pose_source"] = "editable_nominal_world_poses"
            result["config_file"] = args.config
            if args.out:
                save_json(args.out, result)
        elif args.command == "auto-3d":
            paths = {camera_id: getattr(args, camera_id.lower())
                     for camera_id in ("D01", "D02", "D03", "D04")
                     if getattr(args, camera_id.lower())}
            if args.reference not in paths or len(paths) < 2:
                raise ValueError("Provide the reference image and at least one other camera image")
            if (args.point_a is None) != (args.point_b is None):
                raise ValueError("Supply both --point-a and --point-b, or neither to click")
            images = {camera_id: load_image(path) for camera_id, path in paths.items()}
            selected = (args.point_a, args.point_b) if args.point_a else click_points(images[args.reference], ["A", "B"])
            pose_state = load_json(args.pose_state) if args.pose_state else None
            result = auto_measure(images, args.reference, selected, load_json(args.config),
                                  args.matcher, args.trolley_origin_m, pose_state,
                                  args.allow_diagnostic_pose,
                                  load_json(args.height_alignment) if args.height_alignment else None)
            if args.later_reference:
                result["temporal"] = motion_evidence(
                    images[args.reference], load_image(args.later_reference),
                    {label: point for label, point in zip(("A", "B"), selected)})
            if args.out:
                save_json(args.out, result)
            if args.annotated_dir:
                directory = Path(args.annotated_dir)
                reference_markers = next(
                    (view["anchor"]["reference_markers"] for view in result["views"].values()
                     if "anchor" in view), None)
                save_matched_points(images[args.reference], selected,
                                    directory / f"{args.reference}_reference.png", "REFERENCE",
                                    reference_markers)
                for camera_id, view in result["views"].items():
                    if any(point is not None for point in view.get("pixel_points", [])):
                        save_matched_points(images[camera_id], view["pixel_points"],
                                            directory / f"{camera_id}_matched.png",
                                            "AUTO MATCH")
        elif args.command == "vehicle-diagnostic":
            if args.reference == args.target:
                raise ValueError("离线诊断必须使用两路不同相机")
            if (args.point_a is None) != (args.point_b is None):
                raise ValueError("A/B 点需同时提供或同时省略")
            images = {args.reference: load_image(args.reference_image),
                      args.target: load_image(args.target_image)}
            cameras = load_json(args.config)["cameras"]
            anchor = estimate_cart_anchor(images[args.reference], images[args.target],
                                          cameras[args.reference], cameras[args.target],
                                          args.length_range_m)
            result = {"status": "diagnostic_only",
                      "anchor": {key: value for key, value in anchor.items()
                                 if not key.startswith("_")}}
            if args.point_a is not None:
                result["surface_measure"] = measure_on_cart_plane(
                    [args.point_a, args.point_b], anchor)
            if args.state_out:
                state = state_from_vehicle_diagnostic(
                    images, anchor, {name: cameras[name]["height_m"] for name in images},
                    args.reference, args.target)
                save_json(args.state_out, state)
                result["diagnostic_state_file"] = args.state_out
            if args.annotated_dir:
                directory = Path(args.annotated_dir)
                first_points = [args.point_a, args.point_b] if args.point_a else []
                target_points = result.get("surface_measure", {}).get("pixel_points", [])
                save_matched_points(images[args.reference], first_points,
                                    directory / f"{args.reference}_reference.png", "OFFLINE CART",
                                    anchor["reference_markers"])
                save_matched_points(images[args.target], target_points,
                                    directory / f"{args.target}_projected.png", "OFFLINE CART",
                                    anchor["target_markers"])
            if args.out:
                save_json(args.out, result)
        elif args.command == "reference-check":
            result = check_reference_length(load_json(args.measurement), args.reference_name,
                                            args.length_range_m)
            if args.out:
                save_json(args.out, result)
        elif args.command in ("init-pose-state", "init-height-state"):
            views = load_json(args.views)["views"]
            prepared = []
            for view in views:
                image = load_image(view["image"])
                pose_data = load_json(view["pose"])
                if (args.command == "init-pose-state"
                        and pose_data.get("calibration_image_sha256") != image_sha256(view["image"])):
                    raise ValueError("标定文件与对应图像不一致")
                prepared.append({"camera_id": view["camera_id"], "image": image,
                                 "pose": pose_data})
            result = (state_from_surveyed_group_poses(prepared, load_json(args.config))
                      if args.command == "init-height-state"
                      else state_from_surveyed_poses(prepared, args.ground_z_m))
            save_json(args.out, result)
        elif args.command == "track-pose":
            state = load_json(args.state)
            pairs = {}
            for item in load_json(args.frames)["frames"]:
                name = item["camera_id"]
                if name in pairs:
                    raise ValueError("相机编号不能重复")
                pairs[name] = (load_image(item["previous"]), load_image(item["current"]))
            result = update_pose_state(state, pairs, args.max_step_m)
            result["updated_cameras"] = list(pairs)
            save_json(args.out, result)
        elif args.command == "manual-nominal-3d":
            views = load_json(args.views)["views"]
            for view in views:
                image = load_image(view["image"])
                view["image_size"] = [image.shape[1], image.shape[0]]
                if isinstance(view.get("intrinsics"), str):
                    view["intrinsics"] = load_json(view["intrinsics"])
            result = estimate_manual_nominal(views, load_json(args.config),
                                             args.max_ray_gap_m, args.max_time_offset_ms,
                                             args.known_length_range_m)
            if args.out:
                save_json(args.out, result)
        elif args.command == "compare-height":
            observations = load_json(args.observations)
            instant = parse_instant(observations["time"])
            image_paths = {}
            for label in ("A", "B"):
                for view in observations[label]:
                    name = view["camera_id"]
                    path = view["image"]
                    if name in image_paths and image_paths[name] != path:
                        raise ValueError(f"{name} 在同一次测量中不能使用不同图像")
                    image_paths[name] = path
            images = {name: load_image(path) for name, path in image_paths.items()}
            result = compare_group_points(observations, images, load_json(args.state),
                                          load_json(args.config),
                                          load_json(args.alignment) if args.alignment else None)
            result["requested_time"] = instant.isoformat()
            if args.out:
                save_json(args.out, result)
        else:
            views = load_json(args.views)["views"]
            if len({v["camera_id"] for v in views}) != len(views):
                raise ValueError("Views must come from distinct cameras")
            for view in views:
                image = load_image(view["image"])
                view["image_size"] = [image.shape[1], image.shape[0]]
                view["pose"] = load_json(view["pose"])
                if view["pose"].get("calibration_image_sha256") != image_sha256(view["image"]):
                    raise ValueError(f"Pose calibration does not belong to image of camera {view['camera_id']}")
                if view.get("pixels") is None:
                    view["pixels"] = click_points(image, ["A", "B"])
            result = measure_3d(views, args.max_time_offset_ms,
                                args.max_reprojection_error_px, args.min_ray_angle_deg)
            if args.out:
                save_json(args.out, result)
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json
              else format_result(args.command, result))
    except (ValueError, KeyError, ImportError, RuntimeError, cv2.error) as exc:
        parser().exit(2, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
