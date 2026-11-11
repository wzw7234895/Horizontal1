"""Triangulate manually identified points using explicitly assumed camera poses."""

from __future__ import annotations

import math

import numpy as np

from .geometry import _normalized_points, validate_intrinsics


def _rays(view: dict, camera: dict, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    points = np.asarray(view["pixels"], dtype=float)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError("Each view needs two finite corresponding pixels")
    if np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
        raise ValueError("A selected pixel lies outside its image")
    tilt = float(camera["down_tilt_deg"])
    yaw = float(view["azimuth_deg"])
    if not (0 < tilt <= 90 and math.isfinite(yaw)):
        raise ValueError("Invalid camera orientation")
    roll = float(camera.get("roll_deg", 0))
    if not math.isfinite(roll) or abs(roll) > 90:
        raise ValueError("Invalid camera roll")
    intrinsics = view.get("intrinsics")
    if intrinsics is not None:
        validate_intrinsics(intrinsics, image_size)
        normalized = _normalized_points(points, intrinsics)
        xn, yn = normalized[:, 0], normalized[:, 1]
    else:
        fov = float(camera["horizontal_fov_deg"])
        if not 0 < fov < 170:
            raise ValueError("Invalid nominal field of view")
        focal = width / (2 * math.tan(math.radians(fov) / 2))
        xn = (points[:, 0] - (width - 1) / 2) / focal
        yn = (points[:, 1] - (height - 1) / 2) / focal
    r = math.radians(roll)
    xn, yn = math.cos(r) * xn - math.sin(r) * yn, math.sin(r) * xn + math.cos(r) * yn
    a, t = math.radians(yaw), math.radians(tilt)
    right = np.array([math.cos(a), math.sin(a), 0.0])
    forward = np.array([-math.sin(a), math.cos(a), 0.0])
    rays = xn[:, None] * right + (math.cos(t) - yn * math.sin(t))[:, None] * forward
    rays[:, 2] = -math.sin(t) - yn * math.cos(t)
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def estimate_manual_nominal(views: list[dict], config: dict,
                            max_ray_gap_m: float = 0.5,
                            max_time_offset_ms: float | None = None,
                            known_length_range_m: tuple[float, float] | None = None) -> dict:
    """Return a nominal candidate only when both corresponding ray pairs meet."""
    if len(views) != 2 or views[0]["camera_id"] == views[1]["camera_id"]:
        raise ValueError("Provide two different camera views")
    if not math.isfinite(max_ray_gap_m) or max_ray_gap_m <= 0:
        raise ValueError("Ray-gap tolerance must be positive and finite")
    if max_time_offset_ms is not None and (not math.isfinite(max_time_offset_ms)
                                            or max_time_offset_ms < 0):
        raise ValueError("Time-offset tolerance must be nonnegative and finite")
    if known_length_range_m is not None:
        lower, upper = known_length_range_m
        if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower <= upper):
            raise ValueError("Known length range must be positive, finite and ordered")
    times = [view.get("exposure_time_ms") for view in views]
    sync = {"status": "unknown", "reason": "两路真实曝光时间戳未齐备"}
    if all(value is not None for value in times):
        offset = abs(float(times[0]) - float(times[1]))
        if not math.isfinite(offset):
            raise ValueError("Exposure timestamps must be finite")
        sync = {"status": "reported", "offset_ms": offset}
        if max_time_offset_ms is not None and offset > max_time_offset_ms:
            return {"status": "unknown", "reason": "两路曝光时间差超过阈值",
                    "synchronization": sync}
    cameras = config["cameras"]
    centers, rays = [], []
    for view in views:
        camera_id = view["camera_id"]
        center = np.asarray(view["center_m"], dtype=float)
        if camera_id not in cameras or center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("Invalid camera ID or assumed centre")
        size = tuple(view["image_size"])
        if len(size) != 2 or min(size) <= 0:
            raise ValueError("Invalid image size")
        centers.append(center)
        rays.append(_rays(view, cameras[camera_id], size))
    if np.linalg.norm(centers[1] - centers[0]) < 1e-6:
        raise ValueError("Assumed camera centres must differ")
    reconstructed, gaps, angles = [], [], []
    for index in range(2):
        first, second = rays[0][index], rays[1][index]
        angle = math.degrees(math.acos(float(np.clip(first @ second, -1, 1))))
        if min(angle, 180 - angle) < 1:
            return {"status": "unknown", "reason": "视线交会角过小"}
        depths, *_ = np.linalg.lstsq(
            np.column_stack((first, -second)), centers[1] - centers[0], rcond=None)
        if np.any(depths <= 0):
            return {"status": "unknown", "reason": "至少一个点位于相机后方"}
        first_xyz = centers[0] + depths[0] * first
        second_xyz = centers[1] + depths[1] * second
        reconstructed.append((first_xyz + second_xyz) / 2)
        gaps.append(float(np.linalg.norm(first_xyz - second_xyz)))
        angles.append(angle)
    base = {"ray_gap_m": gaps, "ray_angle_deg": angles,
            "synchronization": sync,
            "assumed_centers_m": [center.tolist() for center in centers],
            "camera_ids": [view["camera_id"] for view in views],
            "assumed_azimuth_deg": [float(view["azimuth_deg"]) for view in views],
            "intrinsics_source": ["calibrated" if view.get("intrinsics") else "nominal_fov"
                                  for view in views],
            "pixel_points": [view["pixels"] for view in views],
            "max_ray_gap_m": max_ray_gap_m}
    if max(gaps) > max_ray_gap_m:
        return {**base, "status": "unknown", "reason": "同名点视线未在容差内交会"}
    xyz = np.asarray(reconstructed)
    distance = float(np.linalg.norm(xyz[1] - xyz[0]))
    reference_check = None
    if known_length_range_m is not None:
        lower, upper = known_length_range_m
        reference_check = {"range_m": [lower, upper],
                           "status": "within_range" if lower <= distance <= upper else "outside_range",
                           "applicability": "Only when A and B are the endpoints of the referenced length"}
    return {**base, "status": "nominal_3d_candidate_not_calibrated",
            "distance_m": distance,
            "point_xyz_m": xyz.tolist(),
            **({"reference_length_check": reference_check} if reference_check else {}),
            "actual_distance_determined": False,
            "uncertainty_status": "not_bounded_without_measured_relative_pose_and_error_limits",
            "limitations": [
                "Cross-rig camera translation must be measured or independently recovered to determine metric scale.",
                "Nominal intrinsics ignore lens distortion; provide calibrated intrinsics per view to correct it.",
                "Clicked points are treated as corresponding physical points without independent verification.",
                "Burned-in second-level time labels do not prove exposure synchronization.",
                "Aperture affects image sharpness and point-localization error, not pinhole scale directly.",
                "Ray gap is a consistency check, not an accuracy bound or confidence interval.",
            ]}
