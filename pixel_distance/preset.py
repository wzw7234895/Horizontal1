"""Approximate ray/plane measurement from editable nominal camera settings."""

from __future__ import annotations

import math

import numpy as np


def estimate_on_plane(config: dict, camera_id: str, image_size: tuple[int, int],
                      a: tuple[float, float], b: tuple[float, float],
                      plane_height_m: float = 0.0) -> dict:
    """Intersect two pinhole rays with the same horizontal plane.

    Coordinates are relative to the camera's ground projection; their rotation
    in the horizontal plane is immaterial to the distance between two points.
    """
    camera = config["cameras"][camera_id]
    width, height = image_size
    expected_aspect = float(config["expected_aspect_ratio"])
    if width <= 0 or height <= 0 or abs(width / height - expected_aspect) > 0.02:
        raise ValueError("Image aspect ratio differs from the preset; use an uncropped full frame")
    points = np.asarray([a, b], dtype=float)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError("Select two finite pixel points")
    if np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
        raise ValueError("Selected pixels must lie inside the image")
    height_m = float(camera["height_m"])
    fov_deg = float(camera["horizontal_fov_deg"])
    tilt_deg = float(camera["down_tilt_deg"])
    roll_deg = float(camera.get("roll_deg", 0.0))
    if height_m <= 0 or not 0 < fov_deg < 170 or not 0 < tilt_deg <= 90 or abs(roll_deg) > 90:
        raise ValueError("Invalid camera height, field of view, tilt, or roll")
    if not 0 <= plane_height_m < height_m:
        raise ValueError("Assumed plane must be between the floor and camera")
    fx = width / (2 * math.tan(math.radians(fov_deg) / 2))
    fy = fx  # Nominal square pixels; actual fx/fy require camera calibration.
    xn = (points[:, 0] - (width - 1) / 2) / fx
    yn = (points[:, 1] - (height - 1) / 2) / fy
    roll = math.radians(roll_deg)
    xr = math.cos(roll) * xn - math.sin(roll) * yn
    yr = math.sin(roll) * xn + math.cos(roll) * yn
    tilt = math.radians(tilt_deg)
    horizontal_y = math.cos(tilt) - yr * math.sin(tilt)
    downward = math.sin(tilt) + yr * math.cos(tilt)
    if np.any(downward <= 1e-6):
        raise ValueError("A selected ray does not intersect the assumed plane in front of the camera")
    scale = (height_m - plane_height_m) / downward
    xy = np.column_stack((scale * xr, scale * horizontal_y))
    height_range = camera.get("design_height_range_m")
    height_sensitivity = None
    if height_range is not None:
        low, high = map(float, height_range)
        if not 0 < low <= high:
            raise ValueError("Invalid design height range")
        if plane_height_m < low:
            distance = float(np.linalg.norm(xy[1] - xy[0]))
            height_sensitivity = {
                "height_range_m": [low, high],
                "distance_range_m_if_only_height_changes": [
                    distance * (low - plane_height_m) / (height_m - plane_height_m),
                    distance * (high - plane_height_m) / (height_m - plane_height_m),
                ],
            }
    return {
        "status": "nominal_estimate_not_calibrated",
        "method": "pinhole_ray_intersection_with_assumed_horizontal_plane",
        "camera_id": camera_id,
        "camera_name": camera["name"],
        "pixel_points": points.tolist(),
        "plane_height_m": plane_height_m,
        "distance_m": float(np.linalg.norm(xy[1] - xy[0])),
        "plane_xy_m_relative_to_camera": xy.tolist(),
        "parameters_used": {"height_m": height_m, "horizontal_fov_deg": fov_deg,
                            "down_tilt_deg": tilt_deg, "roll_deg": roll_deg,
                            "principal_point": "image center", "pixel_aspect": "square",
                            "distortion": "ignored"},
        "height_only_sensitivity": height_sensitivity,
        "limitations": [
            "Both selected physical points must lie on the same assumed horizontal plane.",
            "Camera parameters are nominal presets, not measured values; this is not a validated physical distance.",
            "Lens distortion and image crop can cause large errors, especially near wide-angle image edges.",
        ],
    }
