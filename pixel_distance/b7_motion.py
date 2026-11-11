"""Visual B7 candidate: simultaneous gantry and trolley travel under load."""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from .pose_state import nominal_camera_entry


CAMERA_GROUPS = {"gantry": ("D01", "D03"), "trolley": ("D02", "D04")}
MAX_SIDE = 960


def _ground_xy(pixels: np.ndarray, camera: dict, ground_z_m: float = 0.0
               ) -> tuple[np.ndarray, np.ndarray]:
    K = np.asarray(camera["K"], dtype=np.float64)
    D = np.asarray(camera["D"], dtype=np.float64)
    R = np.asarray(camera["R"], dtype=np.float64)
    t = np.asarray(camera["t_m"], dtype=np.float64)
    center = -R.T @ t
    normalized = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    rays = np.column_stack((normalized, np.ones(len(pixels)))) @ R
    valid = np.abs(rays[:, 2]) > 1e-4
    depths = np.full(len(pixels), np.nan)
    depths[valid] = (ground_z_m - center[2]) / rays[valid, 2]
    valid &= (depths > 0) & (depths < 100)
    return center[:2] + depths[:, None] * rays[:, :2], valid


def estimate_camera_motion(previous: np.ndarray, current: np.ndarray,
                           camera: dict, ground_mask: np.ndarray | None = None) -> dict:
    """Estimate horizontal camera displacement from static ground candidates."""
    if previous.shape != current.shape or previous.ndim != 3 or previous.shape[2] != 3:
        raise ValueError("同一路相邻画面须为相同尺寸的彩色原始帧")
    height, width = previous.shape[:2]
    if [width, height] != camera["image_size"]:
        raise ValueError("画面尺寸与相机参数不一致")
    if ground_mask is not None and ground_mask.shape != (height, width):
        raise ValueError("地面候选掩码尺寸与画面不一致")
    scale = min(1.0, MAX_SIDE / max(width, height))
    size = (round(width * scale), round(height * scale))
    first = cv2.resize(previous, size, interpolation=cv2.INTER_AREA)
    second = cv2.resize(current, size, interpolation=cv2.INTER_AREA)
    gray_first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    gray_second = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray_first, 35, 245)
    mask[:round(size[1] * 0.12)] = 0
    mask[round(size[1] * 0.92):] = 0
    if ground_mask is not None:
        resized = cv2.resize(ground_mask, size, interpolation=cv2.INTER_NEAREST)
        mask[resized == 0] = 0
    corners = cv2.goodFeaturesToTrack(gray_first, maxCorners=1200,
                                      qualityLevel=0.008, minDistance=8, mask=mask)
    if corners is None or len(corners) < 70:
        return {"status": "unknown", "reason": "静态地面候选特征不足"}
    forward, ok_forward, _ = cv2.calcOpticalFlowPyrLK(
        gray_first, gray_second, corners, None, winSize=(21, 21), maxLevel=3)
    if forward is None or ok_forward is None:
        return {"status": "unknown", "reason": "正向光流失败"}
    backward, ok_backward, _ = cv2.calcOpticalFlowPyrLK(
        gray_second, gray_first, forward, None, winSize=(21, 21), maxLevel=3)
    if backward is None or ok_backward is None:
        return {"status": "unknown", "reason": "反向光流失败"}
    good = ((ok_forward.ravel() > 0) & (ok_backward.ravel() > 0)
            & (np.linalg.norm(backward.reshape(-1, 2) - corners.reshape(-1, 2),
                              axis=1) < 1.5))
    source_px = corners.reshape(-1, 2)[good] / scale
    target_px = forward.reshape(-1, 2)[good] / scale
    if len(source_px) < 60:
        return {"status": "unknown", "reason": "双向光流有效点不足"}
    source_xy, valid_source = _ground_xy(source_px, camera)
    target_xy, valid_target = _ground_xy(target_px, camera)
    valid = valid_source & valid_target
    source_xy, target_xy, source_px = source_xy[valid], target_xy[valid], source_px[valid]
    if len(source_xy) < 60:
        return {"status": "unknown", "reason": "有效地面射线不足"}
    center = -np.asarray(camera["R"]).T @ np.asarray(camera["t_m"])
    transform, inliers = cv2.estimateAffinePartial2D(
        source_xy - center[:2], target_xy - center[:2], method=cv2.RANSAC,
        ransacReprojThreshold=0.10, maxIters=2000, confidence=0.995,
        refineIters=10)
    if transform is None or inliers is None:
        return {"status": "unknown", "reason": "静态地面运动模型无法拟合"}
    keep = inliers.ravel().astype(bool)
    inlier_count = int(keep.sum())
    inlier_fraction = float(keep.mean())
    if inlier_count < 45 or inlier_fraction < 0.55:
        return {"status": "unknown", "reason": "静态地面运动模型支持不足",
                "tracked_points": int(len(source_xy)), "inliers": inlier_count}
    hull = cv2.convexHull(source_px[keep].astype(np.float32))
    coverage = float(cv2.contourArea(hull) / (width * height))
    if coverage < 0.04:
        return {"status": "unknown", "reason": "静态地面特征覆盖过于集中",
                "coverage_fraction": coverage}
    A, b = transform[:, :2], transform[:, 2]
    scale_change = float(np.linalg.norm(A[:, 0]))
    rotation_deg = float(math.degrees(math.atan2(A[1, 0], A[0, 0])))
    if abs(scale_change - 1) > 0.04 or abs(rotation_deg) > 3:
        return {"status": "unknown", "reason": "相机旋转或尺度变化超出短时模型",
                "scale_change": scale_change, "rotation_deg": rotation_deg}
    residual = np.linalg.norm((source_xy[keep] - center[:2]) @ A.T + b
                              - (target_xy[keep] - center[:2]), axis=1)
    median_error = float(np.median(residual))
    p90_error = float(np.percentile(residual, 90))
    if median_error > 0.06 or p90_error > 0.15:
        return {"status": "unknown", "reason": "静态地面运动残差过大",
                "median_error_m": median_error, "p90_error_m": p90_error}
    displacement = -np.linalg.solve(A, b)
    return {"status": "estimated", "displacement_xy_m": displacement.tolist(),
            "tracked_points": int(len(source_xy)), "inliers": inlier_count,
            "inlier_fraction": inlier_fraction, "coverage_fraction": coverage,
            "median_error_m": median_error, "p90_error_m": p90_error,
            "rotation_deg": rotation_deg, "scale_change": scale_change,
            "ground_mask_provided": ground_mask is not None}


def _unit_axis(value: object, name: str) -> np.ndarray:
    axis = np.asarray(value, dtype=float)
    if axis.shape != (2,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8:
        raise ValueError(f"{name} 轴必须是有效二维方向")
    return axis / np.linalg.norm(axis)


class B7MotionDetector:
    """Consume synchronized four-camera frames and retain a short motion window."""

    def __init__(self, config: dict, ground_masks: dict[str, np.ndarray] | None = None,
                 min_speed_mps: float = 0.08,
                 max_group_disagreement_mps: float = 0.12,
                 min_motion_duration_s: float = 3.0,
                 window_s: float = 6.0, max_frame_gap_s: float = 0.5):
        values = (min_speed_mps, max_group_disagreement_mps,
                  min_motion_duration_s, window_s, max_frame_gap_s)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("B7 速度、持续时间及采样间隔参数必须为正数")
        if window_s < min_motion_duration_s:
            raise ValueError("B7 时间窗不能短于最短持续时间")
        axes = config.get("b7_axes", {"gantry_xy": [0, 1], "trolley_xy": [1, 0]})
        self.gantry_axis = _unit_axis(axes["gantry_xy"], "大车")
        self.trolley_axis = _unit_axis(axes["trolley_xy"], "小车")
        if abs(float(self.gantry_axis @ self.trolley_axis)) > 0.1:
            raise ValueError("大车和小车轴向须近似正交")
        self.required = set(CAMERA_GROUPS["gantry"] + CAMERA_GROUPS["trolley"])
        self.masks = ground_masks or {}
        if set(self.masks) - self.required:
            raise ValueError("地面掩码包含未知相机")
        self.config = config
        self.min_speed_mps = min_speed_mps
        self.max_group_disagreement_mps = max_group_disagreement_mps
        self.min_motion_duration_s = min_motion_duration_s
        self.window_s = window_s
        self.max_frame_gap_s = max_frame_gap_s
        self.previous: tuple[float, dict[str, np.ndarray]] | None = None
        self.intervals: deque[dict] = deque()

    def update(self, timestamp_s: float, images: dict[str, np.ndarray] | None,
               loaded: bool | None) -> dict:
        """Return the current B7 result; load state comes from another module."""
        if loaded is not None and not isinstance(loaded, bool):
            raise ValueError("loaded 必须为 True、False 或 None")
        if not math.isfinite(timestamp_s):
            raise ValueError("四路画面时间戳须为有限数值")
        if loaded is not True:
            self.previous = None
            self.intervals.clear()
            return {"status": "not_loaded" if loaded is False else "unknown_load_state",
                    "b7_alarm": False, "reason": "未确认挂载重物"}
        if images is None or set(images) != self.required:
            raise ValueError("B7 每个时刻都必须提供 D01 至 D04 四路画面")
        if self.previous is not None:
            start, before = self.previous
            dt = timestamp_s - start
            if dt <= 0:
                raise ValueError("四路画面时间戳须严格递增")
            if dt > self.max_frame_gap_s:
                self.intervals.append({"start_s": start, "end_s": timestamp_s,
                                       "status": "unknown", "cameras": {},
                                       "reason": "相邻画面间隔超出上限"})
            else:
                self.intervals.append(self._interval(start, before, timestamp_s, images))
        self.previous = (timestamp_s, {name: images[name].copy() for name in self.required})
        while self.intervals and self.intervals[0]["end_s"] <= timestamp_s - self.window_s:
            self.intervals.popleft()
        return self.result()

    def _interval(self, start: float, before: dict[str, np.ndarray],
                  end: float, after: dict[str, np.ndarray]) -> dict:
        dt = end - start
        cameras = {}
        for name in sorted(self.required):
            entry = nominal_camera_entry(name, before[name], self.config)
            cameras[name] = estimate_camera_motion(before[name], after[name],
                                                    entry, self.masks.get(name))
        interval = {"start_s": start, "end_s": end, "cameras": cameras,
                    "status": "unknown"}
        if any(value["status"] != "estimated" for value in cameras.values()):
            interval["reason"] = "至少一路相机未取得可靠静态地面运动"
            return interval
        velocity = {name: np.asarray(value["displacement_xy_m"]) / dt
                    for name, value in cameras.items()}
        disagreement = {group: float(np.linalg.norm(
            velocity[names[0]] - velocity[names[1]]))
            for group, names in CAMERA_GROUPS.items()}
        interval["group_disagreement_mps"] = disagreement
        if any(value > self.max_group_disagreement_mps for value in disagreement.values()):
            interval["reason"] = "同组两路相机运动不一致"
            return interval
        gantry_velocity = np.mean([velocity[n] for n in CAMERA_GROUPS["gantry"]], axis=0)
        trolley_total = np.mean([velocity[n] for n in CAMERA_GROUPS["trolley"]], axis=0)
        trolley_relative = trolley_total - gantry_velocity
        gantry_speed = float(gantry_velocity @ self.gantry_axis)
        trolley_speed = float(trolley_relative @ self.trolley_axis)
        cross_axis = max(abs(float(gantry_velocity @ self.trolley_axis)),
                         abs(float(trolley_relative @ self.gantry_axis)))
        interval.update({"gantry_velocity_xy_mps": gantry_velocity.tolist(),
                         "trolley_total_velocity_xy_mps": trolley_total.tolist(),
                         "trolley_relative_velocity_xy_mps": trolley_relative.tolist(),
                         "gantry_axis_speed_mps": gantry_speed,
                         "trolley_axis_speed_mps": trolley_speed,
                         "cross_axis_residual_mps": cross_axis})
        if cross_axis > self.max_group_disagreement_mps:
            interval["reason"] = "运动方向与大车/小车轴不一致"
        else:
            interval["status"] = "analyzed"
            interval["simultaneous_motion"] = (abs(gantry_speed) >= self.min_speed_mps
                                                and abs(trolley_speed) >= self.min_speed_mps)
        return interval

    def result(self) -> dict:
        intervals = list(self.intervals)
        longest = 0.0
        run_start = run_end = None
        direction = None
        for item in intervals:
            if item["status"] != "analyzed" or not item["simultaneous_motion"]:
                run_start = run_end = direction = None
                continue
            current = (math.copysign(1, item["gantry_axis_speed_mps"]),
                       math.copysign(1, item["trolley_axis_speed_mps"]))
            if direction != current or run_end is None or abs(item["start_s"] - run_end) > 1e-6:
                run_start = item["start_s"]
            run_end, direction = item["end_s"], current
            longest = max(longest, run_end - run_start)
        alarm = longest + 1e-6 >= self.min_motion_duration_s
        status = ("alarm_candidate" if alarm else
                  "unknown" if not intervals or any(i["status"] == "unknown" for i in intervals)
                  else "no_simultaneous_motion")
        return {"status": status, "b7_alarm": alarm, "loaded": True,
                "method": "four_camera_ground_flow_axis_decomposition",
                "validation_status": "nominal_camera_geometry_not_site_validated",
                "min_speed_mps": self.min_speed_mps,
                "max_group_disagreement_mps": self.max_group_disagreement_mps,
                "min_motion_duration_s": self.min_motion_duration_s,
                "max_frame_gap_s": self.max_frame_gap_s,
                "window_s": self.window_s,
                "longest_confirmed_duration_s": longest,
                "confirmed_interval_count": sum(i.get("simultaneous_motion", False)
                                                for i in intervals),
                "axes": {"gantry_xy": self.gantry_axis.tolist(),
                         "trolley_xy": self.trolley_axis.tolist()},
                "intervals": intervals,
                "distance_geometry_updated": False}


def analyze_b7(frames: list[tuple[float, dict[str, np.ndarray]]],
               loaded: bool | None, config: dict,
               ground_masks: dict[str, np.ndarray] | None = None,
               min_speed_mps: float = 0.08,
               max_group_disagreement_mps: float = 0.12,
               min_motion_duration_s: float = 3.0,
               window_s: float = 6.0,
               max_frame_gap_s: float = 0.5) -> dict:
    """Batch wrapper for synchronized four-camera frame bundles."""
    detector = B7MotionDetector(config, ground_masks, min_speed_mps,
                                max_group_disagreement_mps, min_motion_duration_s,
                                window_s, max_frame_gap_s)
    if loaded is not True:
        return detector.update(0.0, None, loaded)
    if not frames:
        raise ValueError("B7 至少需要一组四路画面")
    result = {}
    for timestamp_s, images in frames:
        result = detector.update(float(timestamp_s), images, loaded)
    return result
