"""Camera calibration and planar image-to-metre geometry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"Cannot read image: {path}") from exc
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def save_json(path: str | Path, data: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def image_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_intrinsics(data: dict, size: tuple[int, int]) -> None:
    if data.get("image_size") != list(size):
        raise ValueError("Intrinsic calibration image size differs from input image; use the original full-resolution image")
    if data.get("model") not in ("pinhole", "fisheye"):
        raise ValueError("Intrinsic model must be pinhole or fisheye")
    k = np.asarray(data["K"], dtype=np.float64)
    d = np.asarray(data["D"], dtype=np.float64).reshape(-1)
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("Invalid intrinsic matrix K")
    if not np.isfinite(d).all() or (data["model"] == "fisheye" and d.size != 4):
        raise ValueError("Invalid distortion coefficients D")


def undistort_points(points: np.ndarray, intrinsics: dict | None) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if intrinsics is None:
        return points.reshape(-1, 2)
    k = np.asarray(intrinsics["K"], dtype=np.float64)
    d = np.asarray(intrinsics["D"], dtype=np.float64)
    if intrinsics["model"] == "fisheye":
        corrected = cv2.fisheye.undistortPoints(points, k, d, P=k)
    else:
        corrected = cv2.undistortPoints(points, k, d, P=k)
    return corrected.reshape(-1, 2)


def project_ground(points: np.ndarray, h: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(points, np.asarray(h, dtype=np.float64))
    result = mapped.reshape(-1, 2)
    if not np.isfinite(result).all():
        raise ValueError("Projection is singular at the selected point")
    return result


def fit_plane(
    image_size: tuple[int, int], anchors: list[dict],
    intrinsics: dict | None = None, threshold_m: float = 0.05,
    max_check_error_m: float | None = None,
) -> dict:
    if threshold_m <= 0 or not np.isfinite(threshold_m):
        raise ValueError("RANSAC threshold must be positive")
    if intrinsics is not None:
        validate_intrinsics(intrinsics, image_size)
    controls = [p for p in anchors if p.get("role", "control") == "control"]
    checks = [p for p in anchors if p.get("role") == "check"]
    if len(controls) + len(checks) != len(anchors):
        raise ValueError("Anchor role must be control or check")
    if len(controls) < 6:
        raise ValueError("At least six measured ground control points are required")
    if len(checks) < 2:
        raise ValueError("At least two independent surveyed check points are required")
    if max_check_error_m is None or max_check_error_m <= 0:
        raise ValueError("Set a positive site-approved maximum check error in metres")
    if len({p["id"] for p in anchors}) != len(anchors):
        raise ValueError("Ground point IDs must be unique")
    ordered = controls + checks
    pixels = np.asarray([p["pixel"] for p in ordered], dtype=np.float64)
    world = np.asarray([p["xy_m"] for p in ordered], dtype=np.float64)
    if pixels.shape != (len(anchors), 2) or world.shape != pixels.shape:
        raise ValueError("Each anchor needs pixel [u,v] and surveyed xy_m [x,y]")
    if not np.isfinite(pixels).all() or not np.isfinite(world).all():
        raise ValueError("Anchor coordinates must be finite")
    width, height = image_size
    if np.any(pixels < 0) or np.any(pixels[:, 0] >= width) or np.any(pixels[:, 1] >= height):
        raise ValueError("An anchor lies outside the original image")
    corrected = undistort_points(pixels, intrinsics)
    source = corrected[:len(controls)]
    target = world[:len(controls)]
    if cv2.contourArea(cv2.convexHull(source.astype(np.float32))) < 100:
        raise ValueError("Control points are almost collinear in the image")
    if cv2.contourArea(cv2.convexHull(target.astype(np.float32))) < 0.01:
        raise ValueError("Measured control points are almost collinear on the floor")
    h, mask = cv2.findHomography(source, target, cv2.RANSAC, threshold_m)
    if h is None or mask is None or int(mask.sum()) < 5:
        raise ValueError("Ground homography could not be fitted")
    inliers = mask.ravel().astype(bool)
    predicted = project_ground(source, h)
    residuals = np.linalg.norm(predicted - target, axis=1)
    check_residuals = []
    if checks:
        predicted_checks = project_ground(corrected[len(controls):], h)
        check_residuals = np.linalg.norm(predicted_checks - world[len(controls):], axis=1).tolist()
    if max(check_residuals) > max_check_error_m:
        raise ValueError(f"Independent check error {max(check_residuals):.3f} m exceeds the allowed {max_check_error_m:.3f} m")
    return {
        "schema": 1,
        "method": "surveyed_ground_plane_homography",
        "image_size": list(image_size),
        "intrinsics": intrinsics,
        "H_undistorted_pixel_to_xy_m": h.tolist(),
        "control_hull_undistorted_pixel": cv2.convexHull(source[inliers].astype(np.float32)).reshape(-1, 2).tolist(),
        "diagnostics": {
            "control_count": len(controls), "inlier_count": int(inliers.sum()),
            "control_residual_m": residuals.tolist(),
            "inlier_rms_m": float(np.sqrt(np.mean(residuals[inliers] ** 2))),
            "check_count": len(checks), "check_residual_m": check_residuals,
            "check_max_m": float(max(check_residuals)) if check_residuals else None,
            "site_approved_max_check_error_m": max_check_error_m,
        },
        "anchors": anchors,
    }


def measure_ground(
    calibration: dict, image_size: tuple[int, int], a: tuple[float, float],
    b: tuple[float, float], allow_extrapolation: bool = False,
) -> dict:
    if calibration.get("schema") != 1 or calibration.get("method") != "surveyed_ground_plane_homography":
        raise ValueError("Unsupported calibration file")
    if calibration["image_size"] != list(image_size):
        raise ValueError("Image dimensions differ from ground calibration")
    raw = np.asarray([a, b], dtype=np.float64)
    width, height = image_size
    if not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw[:, 0] >= width) or np.any(raw[:, 1] >= height):
        raise ValueError("Selected pixels must lie inside the image")
    corrected = undistort_points(raw, calibration.get("intrinsics"))
    hull = np.asarray(calibration["control_hull_undistorted_pixel"], dtype=np.float32)
    outside = [cv2.pointPolygonTest(hull, tuple(map(float, p)), False) < 0 for p in corrected]
    if any(outside) and not allow_extrapolation:
        raise ValueError("A selected point lies outside the ground control area; extrapolation was refused")
    xy = project_ground(corrected, calibration["H_undistorted_pixel_to_xy_m"])
    return {
        "status": "estimated", "assumption": "both selected physical points lie on the surveyed ground plane",
        "distance_m": float(np.linalg.norm(xy[1] - xy[0])),
        "pixel_points": raw.tolist(), "ground_xy_m": xy.tolist(),
        "outside_control_area": outside,
        "calibration_inlier_rms_m": calibration["diagnostics"]["inlier_rms_m"],
        "calibration_check_max_m": calibration["diagnostics"]["check_max_m"],
        "note": "Calibration residuals do not include point-click, floor relief, or camera movement errors.",
    }


def calibrate_intrinsics(images: list[str], pattern: tuple[int, int], square_m: float, model: str) -> dict:
    if square_m <= 0 or pattern[0] < 3 or pattern[1] < 3:
        raise ValueError("Specify positive square size and chessboard inner-corner counts >= 3")
    obj = np.zeros((pattern[0] * pattern[1], 3), dtype=np.float64)
    obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_m
    object_points, image_points, used = [], [], []
    size = None
    for path in images:
        image = load_image(path)
        current = (image.shape[1], image.shape[0])
        if size is not None and current != size:
            raise ValueError("All calibration images must have the same dimensions")
        size = current
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCornersSB(gray, pattern)
        if ok:
            object_points.append(obj.copy())
            image_points.append(corners.astype(np.float64))
            used.append(path)
    if len(used) < 10:
        raise ValueError(f"Only {len(used)} usable chessboard views; collect at least 10 varied full-frame views")
    if model == "fisheye":
        objects = [x.reshape(1, -1, 3) for x in object_points]
        rms, k, d, _, _ = cv2.fisheye.calibrate(
            objects, image_points, size, np.eye(3), np.zeros((4, 1)),
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        )
    elif model == "pinhole":
        rms, k, d, _, _ = cv2.calibrateCamera(
            [x.astype(np.float32) for x in object_points],
            [x.astype(np.float32) for x in image_points], size, None, None,
        )
    else:
        raise ValueError("Model must be pinhole or fisheye")
    return {"schema": 1, "model": model, "image_size": list(size),
            "K": k.tolist(), "D": d.reshape(-1).tolist(),
            "reprojection_rms_px": float(rms), "used_images": used}


def _normalized_points(pixels: np.ndarray, intrinsics: dict) -> np.ndarray:
    k = np.asarray(intrinsics["K"], dtype=np.float64)
    d = np.asarray(intrinsics["D"], dtype=np.float64)
    pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    if intrinsics["model"] == "fisheye":
        return cv2.fisheye.undistortPoints(pts, k, d).reshape(-1, 2)
    return cv2.undistortPoints(pts, k, d).reshape(-1, 2)


def _project_3d(points: np.ndarray, pose: dict, intrinsics: dict) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 1, 3)
    rvec = np.asarray(pose["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(pose["tvec_m"], dtype=np.float64).reshape(3, 1)
    k = np.asarray(intrinsics["K"], dtype=np.float64)
    d = np.asarray(intrinsics["D"], dtype=np.float64)
    if intrinsics["model"] == "fisheye":
        projected, _ = cv2.fisheye.projectPoints(xyz, rvec, tvec, k, d)
    else:
        projected, _ = cv2.projectPoints(xyz, rvec, tvec, k, d)
    return projected.reshape(-1, 2)


def fit_pose(image_size: tuple[int, int], anchors: list[dict], intrinsics: dict,
             max_check_error_px: float) -> dict:
    validate_intrinsics(intrinsics, image_size)
    controls = [p for p in anchors if p.get("role", "control") == "control"]
    checks = [p for p in anchors if p.get("role") == "check"]
    if len(controls) + len(checks) != len(anchors) or len(controls) < 6 or len(checks) < 2:
        raise ValueError("Pose requires at least six 3D control points and two independent check points")
    if max_check_error_px <= 0:
        raise ValueError("Set a positive site-approved maximum pose check error")
    ordered = controls + checks
    pixels = np.asarray([p["pixel"] for p in ordered], dtype=np.float64)
    world = np.asarray([p["xyz_m"] for p in ordered], dtype=np.float64)
    if pixels.shape != (len(ordered), 2) or world.shape != (len(ordered), 3):
        raise ValueError("Each pose anchor needs pixel [u,v] and surveyed xyz_m [x,y,z]")
    if not np.isfinite(pixels).all() or not np.isfinite(world).all():
        raise ValueError("Pose anchors must be finite")
    width, height = image_size
    if np.any(pixels < 0) or np.any(pixels[:, 0] >= width) or np.any(pixels[:, 1] >= height):
        raise ValueError("A pose anchor lies outside the image")
    if np.linalg.matrix_rank(world[:len(controls)] - world[:len(controls)].mean(axis=0)) < 3:
        raise ValueError("Pose control points must span three dimensions, not only the floor")
    normalized = _normalized_points(pixels[:len(controls)], intrinsics)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        world[:len(controls)].astype(np.float64), normalized.astype(np.float64),
        np.eye(3), None, iterationsCount=500, reprojectionError=0.01,
        confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or inliers is None or len(inliers) < 6:
        raise ValueError("Could not solve a consistent camera pose")
    cv2.solvePnPRefineLM(world[inliers.ravel()], normalized[inliers.ravel()],
                         np.eye(3), None, rvec, tvec)
    pose = {"rvec": rvec.reshape(-1).tolist(), "tvec_m": tvec.reshape(-1).tolist()}
    errors = np.linalg.norm(_project_3d(world, pose, intrinsics) - pixels, axis=1)
    check_max = float(max(errors[len(controls):]))
    if check_max > max_check_error_px:
        raise ValueError(f"Pose check error {check_max:.2f} px exceeds allowed {max_check_error_px:.2f} px")
    return {"schema": 1, "method": "surveyed_3d_pnp", "image_size": list(image_size),
            "intrinsics": intrinsics, **pose,
            "diagnostics": {"control_count": len(controls), "inlier_count": len(inliers),
                            "check_max_px": check_max,
                            "site_approved_max_check_error_px": max_check_error_px}}


def measure_3d(views: list[dict], max_time_offset_ms: float,
               max_reprojection_error_px: float, min_ray_angle_deg: float = 1.0) -> dict:
    if len(views) < 2:
        raise ValueError("At least two synchronized camera views are required")
    if max_time_offset_ms < 0 or max_reprojection_error_px <= 0 or min_ray_angle_deg <= 0:
        raise ValueError("Invalid synchronization or geometry tolerance")
    times = [float(v["timestamp_ms"]) for v in views]
    offset = max(times) - min(times)
    if offset > max_time_offset_ms:
        raise ValueError(f"Frame time offset {offset:.2f} ms exceeds allowed {max_time_offset_ms:.2f} ms")
    origins, directions = [], []
    for view in views:
        pose = view["pose"]
        intrinsics = pose["intrinsics"]
        size = tuple(pose["image_size"])
        validate_intrinsics(intrinsics, size)
        if view["image_size"] != list(size):
            raise ValueError("View image dimensions differ from its pose calibration")
        pixels = np.asarray(view["pixels"], dtype=np.float64)
        if pixels.shape != (2, 2) or not np.isfinite(pixels).all():
            raise ValueError("Each view needs corresponding A and B pixels")
        if np.any(pixels < 0) or np.any(pixels[:, 0] >= size[0]) or np.any(pixels[:, 1] >= size[1]):
            raise ValueError("A selected pixel is outside a view")
        normalized = _normalized_points(pixels, intrinsics)
        r, _ = cv2.Rodrigues(np.asarray(pose["rvec"], dtype=np.float64))
        t = np.asarray(pose["tvec_m"], dtype=np.float64)
        origin = -r.T @ t
        rays = np.column_stack((normalized, np.ones(2))) @ r
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        origins.append(np.repeat(origin.reshape(1, 3), 2, axis=0))
        directions.append(rays)
    origins = np.asarray(origins)
    directions = np.asarray(directions)
    results, angles, reprojections = [], [], []
    for index in range(2):
        rays = directions[:, index, :]
        dots = np.clip(np.abs(rays @ rays.T), 0, 1)
        angle = float(np.degrees(np.arccos(np.min(dots[np.triu_indices(len(views), 1)]))))
        if angle < min_ray_angle_deg:
            raise ValueError(f"Ray intersection angle {angle:.2f} degrees is too small")
        projectors = np.eye(3) - rays[:, :, None] * rays[:, None, :]
        system = projectors.sum(axis=0)
        rhs = np.einsum("nij,nj->i", projectors, origins[:, index, :])
        if np.linalg.cond(system) > 1e8:
            raise ValueError("Ray geometry is ill-conditioned")
        point = np.linalg.solve(system, rhs)
        errors = []
        for view in views:
            pose = view["pose"]
            r, _ = cv2.Rodrigues(np.asarray(pose["rvec"], dtype=np.float64))
            if (r @ point + np.asarray(pose["tvec_m"], dtype=np.float64))[2] <= 0:
                raise ValueError("Triangulated point lies behind a camera")
            pixel = _project_3d(point.reshape(1, 3), pose, pose["intrinsics"])[0]
            errors.append(float(np.linalg.norm(pixel - np.asarray(view["pixels"][index]))))
        if max(errors) > max_reprojection_error_px:
            raise ValueError(f"3D reprojection error {max(errors):.2f} px exceeds allowed {max_reprojection_error_px:.2f} px")
        results.append(point)
        angles.append(angle)
        reprojections.append(errors)
    return {"status": "estimated", "method": "multiview_triangulation",
            "distance_m": float(np.linalg.norm(results[1] - results[0])),
            "world_xyz_m": np.asarray(results).tolist(), "ray_angle_deg": angles,
            "reprojection_error_px": reprojections, "frame_time_offset_ms": offset,
            "note": "Requires exact cross-view physical point correspondence and poses at each exposure; validation errors are not total uncertainty."}
