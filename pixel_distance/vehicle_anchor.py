"""Recover a conditional cross-rig pose from four visible cart markers."""

from __future__ import annotations

import itertools
import math

import cv2
import numpy as np


def _intrinsics(camera: dict, image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    fov = float(camera["horizontal_fov_deg"])
    if not 0 < fov < 170:
        raise ValueError("Invalid vehicle-anchor field of view")
    focal = width / (2 * math.tan(math.radians(fov) / 2))
    return np.array([[focal, 0, (width - 1) / 2],
                     [0, focal, (height - 1) / 2], [0, 0, 1]], dtype=float)


def _ordered_corners(points: np.ndarray) -> tuple[np.ndarray, float]:
    centered = points - points.mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered)
    major = vectors[:, np.argmax(values)]
    if major[np.argmax(abs(major))] < 0:
        major = -major
    minor = np.array([-major[1], major[0]])
    if minor[np.argmax(abs(minor))] < 0:
        minor = -minor
    groups = np.argsort(centered @ major).reshape(2, 2)
    ordered = np.vstack([points[group[np.argsort(centered[group] @ minor)]]
                         for group in groups])
    length = np.linalg.norm(ordered[2:].mean(axis=0) - ordered[:2].mean(axis=0))
    width = np.mean(np.linalg.norm(ordered[::2] - ordered[1::2], axis=1))
    if width <= 0 or not 2 <= length / width <= 8:
        raise ValueError("Red markers do not form a cart-length quadrilateral")
    polygon = ordered[[0, 1, 3, 2]].astype(np.float32)
    area = abs(cv2.contourArea(polygon))
    if area < 3000 or not cv2.isContourConvex(polygon):
        raise ValueError("Red markers do not enclose a visible cart surface")
    return ordered, area


def detect_cart_markers(image: np.ndarray) -> np.ndarray:
    """Return two marker pairs ordered along the green-track direction."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (75, 30, 50), (105, 180, 220))
    distance_to_green = cv2.distanceTransform(255 - green, cv2.DIST_L2, 5)
    red = (cv2.inRange(hsv, (0, 90, 80), (12, 255, 255)) |
           cv2.inRange(hsv, (170, 90, 80), (179, 255, 255)))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(red)
    candidates = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        center = centroids[index]
        if (60 <= area <= 500 and 4 <= width <= 35 and 4 <= height <= 35
                and 0.4 <= width / height <= 2.5
                and y >= image.shape[0] * 0.12):
            candidates.append((center, float(distance_to_green[round(center[1]), round(center[0])])))
    nearest = [center for center, distance in candidates if distance <= 12.5]
    if len(nearest) < 4:
        nearest = [center for center, distance in candidates if distance <= 18]
    candidates = nearest
    if len(candidates) < 4 or len(candidates) > 12:
        raise ValueError(f"Expected four cart-end markers near green track; found {len(candidates)}")
    choices = []
    for group in itertools.combinations(candidates, 4):
        try:
            ordered, area = _ordered_corners(np.asarray(group, dtype=float))
        except ValueError:
            continue
        choices.append((area, ordered))
    if not choices:
        raise ValueError("Cart-end markers do not form a valid quadrilateral")
    choices.sort(key=lambda item: item[0], reverse=True)
    if len(choices) > 1 and choices[1][0] > 0.75 * choices[0][0]:
        raise ValueError("Multiple cart-marker quadrilaterals are plausible")
    return choices[0][1]


def _basis(yaw_deg: float, tilt_deg: float) -> np.ndarray:
    yaw, tilt = np.deg2rad([yaw_deg, tilt_deg])
    right = np.array([np.cos(yaw), np.sin(yaw), 0])
    forward = np.array([-np.sin(yaw), np.cos(yaw), 0])
    down = -np.sin(tilt) * forward + np.array([0, 0, -np.cos(tilt)])
    optical = np.cos(tilt) * forward + np.array([0, 0, -np.sin(tilt)])
    return np.column_stack((right, down, optical))


def _rotation_error_deg(actual: np.ndarray, expected: np.ndarray) -> float:
    cosine = np.clip((np.trace(expected.T @ actual) - 1) / 2, -1, 1)
    return math.degrees(math.acos(float(cosine)))


def _fit_view(markers: np.ndarray, K: np.ndarray,
              length: float, width: float) -> dict:
    obj = np.array([[0, 0, 0], [0, width, 0],
                    [length, 0, 0], [length, width, 0]], dtype=float)
    success, rvec, tvec = cv2.solvePnP(obj, markers, K, None,
                                        flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        raise ValueError("Cart-marker pose fit failed")
    R = cv2.Rodrigues(rvec)[0]
    projected = cv2.projectPoints(obj, rvec, tvec, K, None)[0].reshape(-1, 2)
    errors = np.linalg.norm(projected - markers, axis=1)
    center = (-R.T @ tvec).reshape(3)
    return {"R": R, "t": tvec.reshape(3), "center": center,
            "errors": errors, "K": K, "markers": markers}


def _fit_width(reference: np.ndarray, target: np.ndarray,
               K_reference: np.ndarray, K_target: np.ndarray,
               length: float) -> tuple[float, dict, dict]:
    best = None
    for width in np.linspace(0.5, min(4.0, length), 71):
        try:
            first = _fit_view(reference, K_reference, length, float(width))
            second = _fit_view(target, K_target, length, float(width))
        except (ValueError, cv2.error):
            continue
        score = float(np.sum(first["errors"] ** 2) + np.sum(second["errors"] ** 2))
        if best is None or score < best[0]:
            best = (score, float(width), first, second)
    if best is None:
        raise ValueError("No cart width gives valid camera poses")
    return best[1:]


def estimate_cart_anchor(reference: np.ndarray, target: np.ndarray,
                         reference_camera: dict, target_camera: dict,
                         length_range_m: tuple[float, float]) -> dict:
    lower, upper = map(float, length_range_m)
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower <= upper):
        raise ValueError("Cart length range must be positive, finite and ordered")
    if "azimuth_deg" not in reference_camera or "azimuth_deg" not in target_camera:
        raise ValueError("Nominal camera azimuths are needed to disambiguate cart ends")
    length = (lower + upper) / 2
    first_markers = detect_cart_markers(reference)
    second_markers = detect_cart_markers(target)
    K_first = _intrinsics(reference_camera, reference)
    K_second = _intrinsics(target_camera, target)
    expected = (_basis(target_camera["azimuth_deg"], target_camera["down_tilt_deg"]).T @
                _basis(reference_camera["azimuth_deg"], reference_camera["down_tilt_deg"]))
    accepted = []
    diagnostics = []
    for reverse_length, reverse_width in itertools.product((False, True), repeat=2):
        ordered = second_markers.reshape(2, 2, 2).copy()
        if reverse_length:
            ordered = ordered[::-1]
        if reverse_width:
            ordered = ordered[:, ::-1]
        try:
            width, first, second = _fit_width(first_markers, ordered.reshape(4, 2),
                                               K_first, K_second, length)
        except (ValueError, cv2.error):
            continue
        rotation = second["R"] @ first["R"].T
        rotation_error = _rotation_error_deg(rotation, expected)
        camera_height_difference = abs(first["center"][2] - second["center"][2])
        max_error = max(np.max(first["errors"]), np.max(second["errors"]))
        diagnostic = {"reverse_length": reverse_length, "reverse_width": reverse_width,
                      "rotation_error_deg": rotation_error,
                      "camera_height_difference_m": camera_height_difference,
                      "max_marker_reprojection_error_px": float(max_error)}
        diagnostics.append(diagnostic)
        if (first["center"][2] > 0 and second["center"][2] > 0
                and rotation_error <= 45 and camera_height_difference <= 1
                and max_error <= 6):
            accepted.append((diagnostic, width, first, second))
    if len(accepted) != 1:
        raise ValueError(f"Cart pose is {'ambiguous' if accepted else 'inconsistent'}: {diagnostics}")
    check, width, first, second = accepted[0]
    R = second["R"] @ first["R"].T
    t = second["t"] - R @ first["t"]
    return {"length_range_m": [lower, upper], "nominal_length_m": length,
            "estimated_width_m": width, "reference_markers": first_markers.tolist(),
            "target_markers": second["markers"].tolist(),
            "relative_rotation": R.tolist(), "relative_translation_m": t.tolist(),
            "nominal_baseline_m": float(np.linalg.norm(second["center"] - first["center"])),
            "camera_height_above_cart_m": [float(first["center"][2]),
                                           float(second["center"][2])],
            "checks": check, "_reference_fit": first, "_target_fit": second}


def measure_on_cart_plane(points: np.ndarray, anchor: dict,
                          outside_tolerance_px: float = 6) -> dict:
    selected = np.asarray(points, dtype=float)
    corners = np.asarray(anchor["reference_markers"])[[0, 1, 3, 2]].astype(np.float32)
    edge_distances = [cv2.pointPolygonTest(corners, tuple(map(float, point)), True)
                      for point in selected]
    if min(edge_distances) < -outside_tolerance_px:
        raise ValueError("Selected points are not both on the visible cart top")
    used = selected.copy()
    for index, signed_distance in enumerate(edge_distances):
        if signed_distance >= 0:
            continue
        edges = zip(corners, np.roll(corners, -1, axis=0))
        nearest = []
        for start, end in edges:
            direction = end - start
            fraction = np.clip(np.dot(selected[index] - start, direction) /
                               np.dot(direction, direction), 0, 1)
            nearest.append(start + fraction * direction)
        used[index] = min(nearest, key=lambda point: np.linalg.norm(point - selected[index]))
    first = anchor["_reference_fit"]
    second = anchor["_target_fit"]
    H_first = first["K"] @ np.column_stack((first["R"][:, :2], first["t"]))
    H_second = second["K"] @ np.column_stack((second["R"][:, :2], second["t"]))
    cart_xy = cv2.perspectiveTransform(used.reshape(-1, 1, 2),
                                       np.linalg.inv(H_first)).reshape(-1, 2)
    mapped = cv2.perspectiveTransform(cart_xy.reshape(-1, 1, 2),
                                      H_second).reshape(-1, 2)
    distance = float(np.linalg.norm(cart_xy[1] - cart_xy[0]))
    length = anchor["nominal_length_m"]
    return {"distance_m": distance,
            "length_only_range_m": [distance * bound / length
                                    for bound in anchor["length_range_m"]],
            "reference_frame_xyz_m": np.column_stack((cart_xy, np.zeros(2))).tolist(),
            "pixel_points": mapped.tolist(),
            "correspondence_method": "cart_marker_plane_projection",
            "independently_matched_selected_points": False,
            "assumed_surface": "cart_top_plane",
            "edge_surface_ambiguous": min(edge_distances) < outside_tolerance_px,
            "signed_distance_to_cart_quad_px": [float(value) for value in edge_distances],
            "reference_pixels_used": used.tolist(),
            "edge_snap_px": np.linalg.norm(used - selected, axis=1).tolist(),
            "nominal_baseline_m": anchor["nominal_baseline_m"]}


def triangulate_with_cart_anchor(reference_points: np.ndarray,
                                 target_points: np.ndarray, anchor: dict) -> dict:
    """Triangulate separately matched off-cart points using the cart pose."""
    first = anchor["_reference_fit"]
    second = anchor["_target_fit"]
    R = np.asarray(anchor["relative_rotation"])
    t = np.asarray(anchor["relative_translation_m"])
    a = np.asarray(reference_points, dtype=float)
    b = np.asarray(target_points, dtype=float)
    if a.shape != (2, 2) or b.shape != (2, 2):
        raise ValueError("Two corresponding pixels are required in each view")
    norm_a = cv2.undistortPoints(a.reshape(-1, 1, 2), first["K"], None).reshape(-1, 2)
    norm_b = cv2.undistortPoints(b.reshape(-1, 1, 2), second["K"], None).reshape(-1, 2)
    P_first = np.column_stack((np.eye(3), np.zeros(3)))
    P_second = np.column_stack((R, t))
    homogeneous = cv2.triangulatePoints(P_first, P_second, norm_a.T, norm_b.T)
    if np.any(abs(homogeneous[3]) < 1e-9):
        raise ValueError("Selected rays do not intersect stably")
    xyz = (homogeneous[:3] / homogeneous[3]).T
    target_xyz = (R @ xyz.T + t[:, None]).T
    if np.any(xyz[:, 2] <= 0) or np.any(target_xyz[:, 2] <= 0):
        raise ValueError("At least one matched point lies behind a camera")
    reproj_a = xyz[:, :2] / xyz[:, 2, None]
    reproj_b = target_xyz[:, :2] / target_xyz[:, 2, None]
    errors = [float(np.linalg.norm(reproj_a - norm_a, axis=1).max() * first["K"][0, 0]),
              float(np.linalg.norm(reproj_b - norm_b, axis=1).max() * second["K"][0, 0])]
    if max(errors) > 3:
        raise ValueError("Matched points exceed cart-pose reprojection tolerance")
    center = -R.T @ t
    angles = []
    for point in xyz:
        ray_a = point / np.linalg.norm(point)
        ray_b = (point - center) / np.linalg.norm(point - center)
        angles.append(math.degrees(math.acos(float(np.clip(ray_a @ ray_b, -1, 1)))))
    if min(angles) < 1:
        raise ValueError("Stereo ray angle is too small")
    distance = float(np.linalg.norm(xyz[1] - xyz[0]))
    return {"distance_m": distance,
            "length_only_range_m": [distance * bound / anchor["nominal_length_m"]
                                    for bound in anchor["length_range_m"]],
            "reference_frame_xyz_m": xyz.tolist(),
            "pixel_points": b.tolist(),
            "correspondence_method": "independent_feature_match_plus_cart_pose",
            "independently_matched_selected_points": True,
            "reprojection_error_px": errors,
            "ray_angles_deg": angles,
            "nominal_baseline_m": anchor["nominal_baseline_m"]}
