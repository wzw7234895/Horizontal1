"""Extract a common wall-clock second from recorded camera streams."""

from __future__ import annotations

import math
import re
import subprocess
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np

from .auto3d import auto_measure
from .ground_segmentation import segment_ground
from .manual_nominal import _rays
from .preset import estimate_on_plane


VIDEO_NAME = re.compile(r"^(D0[1-4])_(\d{14})\.(?:mp4|avi|mkv)$", re.IGNORECASE)


def parse_instant(value: str) -> datetime:
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", value):
            raise ValueError
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
    except ValueError as exc:
        raise ValueError("时间格式应为 YYYY-MM-DDTHH:MM:SS") from exc


def extract_frame(path: str | Path, camera_id: str, instant: datetime,
                  offset_s: float = 0.0) -> tuple[np.ndarray, dict]:
    path = Path(path)
    matched = VIDEO_NAME.fullmatch(path.name)
    if matched is None or matched.group(1).upper() != camera_id:
        raise ValueError(f"{camera_id} 视频文件名应为 {camera_id}_YYYYMMDDHHMMSS.mp4 等格式")
    if not math.isfinite(offset_s):
        raise ValueError(f"{camera_id} 的时间偏移不是有限数值")
    start = datetime.strptime(matched.group(2), "%Y%m%d%H%M%S")
    elapsed_s = (instant - start).total_seconds() + offset_s
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"无法打开 {camera_id} 视频：{path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count <= 0:
            raise ValueError(f"{camera_id} 视频缺少可用帧率或总帧数")
        index = round(elapsed_s * fps)
        if index < 0 or index >= count:
            raise ValueError(f"{camera_id} 视频不覆盖所选时间")
        decoded = subprocess.run(
            [sys.executable, "-m", "pixel_distance._decode_frame", str(path), str(index)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if decoded.returncode:
            raise ValueError(f"{camera_id} 视频第 {index} 帧解码失败；请检查视频文件")
        warning_count = len(decoded.stderr.splitlines())
        try:
            frame = np.load(BytesIO(decoded.stdout), allow_pickle=False)
        except (ValueError, EOFError) as exc:
            raise ValueError(f"{camera_id} 视频第 {index} 帧数据不完整") from exc
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"{camera_id} 视频第 {index} 帧格式异常")
        return frame, {"path": str(path.resolve()), "frame_index": index,
                       "fps": float(fps), "filename_start": start.isoformat(),
                       "clock_offset_s": float(offset_s),
                       "decoder_warning_lines": warning_count,
                       "image_size": [frame.shape[1], frame.shape[0]]}
    finally:
        cap.release()


def extract_views(videos: dict[str, str | Path], instant: datetime,
                  config: dict) -> tuple[dict[str, np.ndarray], dict]:
    images, metadata = {}, {}
    offsets = config.get("video_clock_offset_s", {})
    for camera_id, path in videos.items():
        try:
            images[camera_id], metadata[camera_id] = extract_frame(
                path, camera_id, instant, float(offsets.get(camera_id, 0.0)))
            metadata[camera_id]["status"] = "extracted"
        except (ValueError, cv2.error) as exc:
            metadata[camera_id] = {"status": "unavailable", "reason": str(exc),
                                   "path": str(path)}
    return images, metadata


def measure_video_views(images: dict[str, np.ndarray], reference_id: str,
                        selected: np.ndarray, config: dict,
                        matcher: str = "auto", plane_height_m: float = 0.0,
                        ground_model_dir: str | Path | None = None,
                        ground_threshold: float = 0.75,
                        allow_assumed_plane: bool = False,
                        pose_state: dict | None = None,
                        height_alignment: dict | None = None) -> dict:
    if reference_id not in images:
        raise ValueError("参考视频在该时间没有可用画面")
    selected = np.asarray(selected, dtype=float)
    if selected.shape != (2, 2) or not np.isfinite(selected).all():
        raise ValueError("请在参考图像中指定两个有限像素坐标")
    if not math.isfinite(plane_height_m) or plane_height_m < 0:
        raise ValueError("单目兜底平面高度必须是非负有限数值")
    if ground_model_dir is not None and plane_height_m != 0:
        raise ValueError("地面分割只支持地面 0 米，不支持其他假定平面高度")
    if not allow_assumed_plane and (plane_height_m != 0 or ground_model_dir is not None):
        raise ValueError("单目平面参数需同时显式启用 --assume-same-plane")
    if len(images) > 1:
        multiview = auto_measure(images, reference_id, selected, config, matcher=matcher,
                                 pose_state=pose_state, height_alignment=height_alignment)
        if multiview["status"] != "unknown":
            return {**multiview, "measurement_mode": multiview.get("measurement_mode", "same_group_nominal_3d"),
                    "single_view_fallback": False,
                    "height_difference_estimated": multiview.get("height_difference_m") is not None,
                    "height_difference_determined": False}
    else:
        multiview = {"status": "unknown", "reason": "没有其他可用视角", "views": {}}
    if not allow_assumed_plane:
        return {"status": "unknown", "reason": "四路视觉候选未形成可用的组内双目测量",
                "reference_camera": reference_id, "reference_pixels": selected.tolist(),
                "distance_m": None, "height_difference_m": None,
                "actual_distance_determined": False, "height_difference_determined": False,
                "measurement_mode": "same_group_unavailable",
                "multiview_attempt": multiview}
    image = images[reference_id]
    ground_check = None
    if ground_model_dir is not None:
        ground_check, _ = segment_ground(image, selected, ground_model_dir,
                                         ground_threshold)
        if not ground_check["all_points_ground_candidates"]:
            return {"status": "unknown",
                    "reason": "多视角匹配失败，且地面分割未确认两个选点均为地面候选",
                    "reference_camera": reference_id,
                    "reference_pixels": selected.tolist(),
                    "distance_m": None, "height_difference_m": None,
                    "actual_distance_determined": False,
                    "height_difference_determined": False,
                    "multiview_attempt": multiview,
                    "ground_segmentation": ground_check}
    try:
        fallback = estimate_on_plane(config, reference_id,
                                     (image.shape[1], image.shape[0]),
                                     selected[0], selected[1], plane_height_m)
        return {**fallback, "status": "single_view_assumed_plane_estimate",
                "measurement_mode": "single_view_assumed_plane",
                "single_view_fallback": True, "multiview_attempt": multiview,
                **({"ground_segmentation": ground_check} if ground_check else {}),
                "actual_distance_determined": False,
                "height_difference_m": None, "height_difference_determined": False,
                "uncertainty_status": "unbounded_without_point_depths"}
    except ValueError as plane_error:
        camera = config["cameras"][reference_id]
        height = float(camera["height_m"])
        tilt = math.radians(float(camera["down_tilt_deg"]))
        if height <= 0 or not 0 < tilt <= math.pi / 2:
            raise ValueError("单目估算缺少有效相机高度或俯角") from plane_error
        range_m = height / math.sin(tilt)
        view = {"pixels": selected.tolist(), "azimuth_deg": 0.0}
        rays = _rays(view, camera, (image.shape[1], image.shape[0]))
        return {"status": "single_view_assumed_range_estimate",
                "measurement_mode": "single_view_assumed_range",
                "single_view_fallback": True,
                "reference_camera": reference_id,
                "reference_pixels": selected.tolist(),
                "assumed_camera_range_m": range_m,
                "distance_m": float(np.linalg.norm((rays[1] - rays[0]) * range_m)),
                "assumed_plane_error": str(plane_error),
                "multiview_attempt": multiview,
                **({"ground_segmentation": ground_check} if ground_check else {}),
                "actual_distance_determined": False,
                "height_difference_m": None, "height_difference_determined": False,
                "uncertainty_status": "unbounded_without_point_depths"}
