"""Metric camera poses in a common frame, updated from overlapping floor views."""

from __future__ import annotations

import hashlib
import math

import cv2
import numpy as np

from .geometry import validate_intrinsics


def decoded_hash(image: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(image.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def _camera_entry(image: np.ndarray, K: np.ndarray, D: np.ndarray,
                  R: np.ndarray, t: np.ndarray) -> dict:
    return {"image_size": [image.shape[1], image.shape[0]],
            "decoded_image_sha256": decoded_hash(image),
            "K": np.asarray(K, dtype=float).tolist(),
            "D": np.asarray(D, dtype=float).reshape(-1).tolist(),
            "R": np.asarray(R, dtype=float).tolist(),
            "t_m": np.asarray(t, dtype=float).reshape(3).tolist(),
            "tracking_steps": 0, "status": "initialized"}


def state_from_surveyed_poses(views: list[dict], ground_z_m: float) -> dict:
    """Combine independently surveyed poses sharing one world coordinate system."""
    if len(views) < 2 or not math.isfinite(ground_z_m):
        raise ValueError("At least two surveyed views and a finite ground elevation are required")
    cameras = {}
    for view in views:
        name, image, pose = view["camera_id"], view["image"], view["pose"]
        if name in cameras:
            raise ValueError("Camera IDs must be unique")
        size = (image.shape[1], image.shape[0])
        if list(size) != pose["image_size"]:
            raise ValueError("Surveyed pose image size differs from current frame")
        validate_intrinsics(pose["intrinsics"], size)
        intrinsic = pose["intrinsics"]
        if intrinsic["model"] != "pinhole":
            raise ValueError("Pose-state tracking currently requires pinhole intrinsics")
        R = cv2.Rodrigues(np.asarray(pose["rvec"], dtype=float))[0]
        cameras[name] = _camera_entry(image, np.asarray(intrinsic["K"], dtype=float),
                                      np.asarray(intrinsic["D"], dtype=float),
                                      R, np.asarray(pose["tvec_m"], dtype=float))
    return {"schema": 1, "method": "surveyed_common_world_poses",
            "ground_z_m": float(ground_z_m), "cameras": cameras,
            "scale_status": "surveyed_but_requires_independent_accuracy_checks"}


def state_from_surveyed_group_poses(views: list[dict], config: dict) -> dict:
    """Keep each rigid group's surveyed local XY for vertical-only comparison."""
    groups = {}
    for view in views:
        group = config["cameras"][view["camera_id"]]["rig"]
        groups.setdefault(group, set()).add(view["camera_id"])
    if len(groups) != 2 or any(len(names) != 2 for names in groups.values()):
        raise ValueError("高度状态需要大车与小车各两台相机的现场标定位姿")
    state = state_from_surveyed_poses(views, 0.0)
    state["method"] = "surveyed_independent_group_poses"
    state["coordinate_frame"] = "independent_group_xy_with_aligned_vertical"
    state["groups"] = {group: sorted(names) for group, names in groups.items()}
    return state


def state_from_vehicle_diagnostic(images: dict[str, np.ndarray],
                                  anchor: dict, camera_heights_m: dict[str, float],
                                  reference_id: str, target_id: str) -> dict:
    """Create a demonstration state; an approximate car length is not site calibration."""
    fits = {reference_id: anchor["_reference_fit"], target_id: anchor["_target_fit"]}
    if any(name not in images or name not in camera_heights_m for name in fits):
        raise ValueError("Vehicle diagnostic needs both camera frames and nominal heights")
    ground_levels = [float(fit["center"][2]) - float(camera_heights_m[name])
                     for name, fit in fits.items()]
    ground_z = float(np.median(ground_levels))
    cameras = {name: _camera_entry(images[name], fit["K"], np.zeros(5),
                                   fit["R"], fit["t"])
               for name, fit in fits.items()}
    return {"schema": 1, "method": "vehicle_length_diagnostic_only",
            "ground_z_m": ground_z, "cameras": cameras,
            "scale_status": "approximate_vehicle_length_not_validated",
            "anchor_checks": anchor["checks"]}


def nominal_camera_entry(name: str, image: np.ndarray, config: dict) -> dict:
    """Build one camera's nominal optics without assuming cross-group rigidity."""
    optics = config["cameras"][name]
    pose = config["nominal_world_poses"][name]
    height, width = image.shape[:2]
    fov = float(optics["horizontal_fov_deg"])
    tilt = float(optics["down_tilt_deg"])
    yaw = float(pose["azimuth_deg"])
    roll = float(optics.get("roll_deg", 0.0))
    center = np.asarray(pose["center_m"], dtype=float)
    if (not 0 < fov < 170 or not 0 < tilt <= 90
            or not np.isfinite([fov, tilt, yaw, roll]).all()
            or center.shape != (3,) or not np.isfinite(center).all()):
        raise ValueError(f"{name} 的名义相机参数无效")
    focal = width / (2 * math.tan(math.radians(fov) / 2))
    K = np.array([[focal, 0, (width - 1) / 2],
                  [0, focal, (height - 1) / 2], [0, 0, 1]], dtype=float)
    a, q, r = map(math.radians, (yaw, tilt, roll))
    right = np.array([math.cos(a), math.sin(a), 0.0])
    forward = np.array([-math.sin(a), math.cos(a), 0.0])
    down = -math.sin(q) * forward - np.array([0.0, 0.0, math.cos(q)])
    optical = math.cos(q) * forward - np.array([0.0, 0.0, math.sin(q)])
    camera_to_rig = np.column_stack((math.cos(r) * right + math.sin(r) * down,
                                     -math.sin(r) * right + math.cos(r) * down,
                                     optical))
    R = camera_to_rig.T
    return _camera_entry(image, K, np.zeros(5), R, -R @ center)


def state_from_nominal_fixed_rig(images: dict[str, np.ndarray], config: dict) -> dict:
    """Build pairwise camera geometry in a rigid rig frame for these images."""
    if config.get("all_cameras_rigid") is not True:
        raise ValueError("固定四相机关系尚未在配置中启用")
    cameras = {name: nominal_camera_entry(name, image, config)
               for name, image in images.items()}
    return {"schema": 1, "method": "nominal_fixed_four_camera_rig",
            "coordinate_frame": "rig_relative", "ground_z_m": 0.0,
            "cameras": cameras, "scale_status": "assumed_not_calibrated"}


def state_from_nominal_group(images: dict[str, np.ndarray], config: dict) -> dict:
    """Build nominal geometry only inside one rigid camera mounting group."""
    groups = {config["cameras"][name]["rig"] for name in images}
    if len(groups) != 1 or not images:
        raise ValueError("名义组内测距只能使用同一安装组的相机")
    local = {**config, "all_cameras_rigid": True,
             "nominal_world_poses": {
                 name: {"center_m": config["cameras"][name]["center_in_rig_m"],
                        "azimuth_deg": config["nominal_world_poses"][name]["azimuth_deg"]}
                 for name in images}}
    state = state_from_nominal_fixed_rig(images, local)
    state["method"] = "nominal_fixed_camera_group"
    state["coordinate_frame"] = next(iter(groups)) + "_relative"
    return state


def _image_points(image: np.ndarray, max_side: int = 960):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scale = min(1.0, max_side / max(gray.shape))
    if scale < 1:
        gray = cv2.resize(gray, (round(image.shape[1] * scale),
                                 round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    mask = np.where((gray > 45) & (gray < 245), 255, 0).astype(np.uint8)
    mask[:round(gray.shape[0] * 0.12)] = 0
    mask[round(gray.shape[0] * 0.92):] = 0
    return gray, mask, np.array([gray.shape[1] / image.shape[1],
                                 gray.shape[0] / image.shape[0]])


def _ground_world_points(pixels: np.ndarray, entry: dict,
                         ground_z_m: float) -> tuple[np.ndarray, np.ndarray]:
    K = np.asarray(entry["K"], dtype=float)
    D = np.asarray(entry["D"], dtype=float)
    R = np.asarray(entry["R"], dtype=float)
    t = np.asarray(entry["t_m"], dtype=float)
    normalized = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    rays = np.column_stack((normalized, np.ones(len(normalized)))) @ R
    center = -R.T @ t
    valid = np.abs(rays[:, 2]) > 1e-4
    depths = np.full(len(rays), np.nan)
    depths[valid] = (ground_z_m - center[2]) / rays[valid, 2]
    valid &= (depths > 0) & (depths < 100)
    world = center + depths[:, None] * rays
    return world, valid


def _homography_ground_pose(src: np.ndarray, dst: np.ndarray, entry: dict,
                            ground_z_m: float) -> tuple[np.ndarray, np.ndarray, dict]:
    K = np.asarray(entry["K"], dtype=float)
    D = np.asarray(entry["D"], dtype=float)
    R0 = np.asarray(entry["R"], dtype=float)
    t0 = np.asarray(entry["t_m"], dtype=float)
    undistorted_src = cv2.undistortPoints(src.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
    undistorted_dst = cv2.undistortPoints(dst.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
    H, mask = cv2.findHomography(undistorted_src, undistorted_dst,
                                 cv2.RANSAC, 4.0)
    if H is None or mask is None:
        raise ValueError("Static ground homography could not be fitted")
    keep = mask.ravel().astype(bool)
    if keep.sum() < 100 or keep.mean() < 0.35:
        raise ValueError("Static ground homography has insufficient support")
    hull = cv2.convexHull(src[keep].astype(np.float32))
    if cv2.contourArea(hull) < 0.08 * entry["image_size"][0] * entry["image_size"][1]:
        raise ValueError("Static ground homography support is too concentrated")
    center = -R0.T @ t0
    distance_to_ground = abs(center[2] - ground_z_m)
    ground_normal = R0 @ np.array([0.0, 0.0, 1.0])
    _, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)
    candidates = []
    world, valid = _ground_world_points(src[keep], entry, ground_z_m)
    target = dst[keep][valid]
    world = world[valid]
    if len(world) < 80:
        raise ValueError("Ground homography has too few valid ground rays")
    for rotation, translation, normal in zip(rotations, translations, normals):
        alignment = float(np.dot(normal.ravel(), ground_normal))
        if alignment < 0.9:
            continue
        R = rotation @ R0
        t = rotation @ t0 - distance_to_ground * translation.ravel()
        projected = cv2.projectPoints(world, cv2.Rodrigues(R)[0], t, K, D)[0].reshape(-1, 2)
        errors = np.linalg.norm(projected - target, axis=1)
        new_center = -R.T @ t
        height_change = abs(new_center[2] - center[2])
        rotation_step = math.degrees(math.acos(float(np.clip(
            (np.trace(rotation) - 1) / 2, -1, 1))))
        candidates.append((float(np.median(errors)), R, t, {
            "method": "ground_homography",
            "tracked_points": int(len(src)), "ground_inliers": int(keep.sum()),
            "median_reprojection_error_px": float(np.median(errors)),
            "p95_reprojection_error_px": float(np.percentile(errors, 95)),
            "normal_alignment": alignment,
            "camera_step_m": float(np.linalg.norm(new_center - center)),
            "height_change_m": height_change,
            "rotation_step_deg": rotation_step}))
    if not candidates:
        raise ValueError("Ground normal does not agree with homography pose")
    _, R, t, diagnostics = min(candidates, key=lambda item: item[0])
    if (diagnostics["median_reprojection_error_px"] > 3.0
            or diagnostics["p95_reprojection_error_px"] > 6.0):
        raise ValueError("Ground homography reprojection error is too high")
    return R, t, diagnostics


def track_camera_pose(previous: np.ndarray, current: np.ndarray, entry: dict,
                      ground_z_m: float, max_step_m: float = 2.0) -> dict:
    """Update one camera using static floor correspondences; reject weak geometry."""
    if previous.shape != current.shape or [previous.shape[1], previous.shape[0]] != entry["image_size"]:
        raise ValueError("Previous and current frame dimensions differ from pose state")
    if decoded_hash(previous) != entry["decoded_image_sha256"]:
        raise ValueError("Previous frame does not match the stored pose state")
    if not math.isfinite(max_step_m) or max_step_m <= 0:
        raise ValueError("Maximum camera step must be positive and finite")
    first, mask, scale = _image_points(previous)
    second, _, second_scale = _image_points(current)
    if not np.allclose(scale, second_scale):
        raise ValueError("Temporal frame scales differ")
    corners = cv2.goodFeaturesToTrack(first, maxCorners=1200, qualityLevel=0.008,
                                      minDistance=8, mask=mask)
    if corners is None or len(corners) < 80:
        raise ValueError("Too few floor/background corners")
    forward, ok_forward, _ = cv2.calcOpticalFlowPyrLK(first, second, corners, None,
                                                       winSize=(21, 21), maxLevel=3)
    if forward is None or ok_forward is None:
        raise ValueError("Forward background tracking failed")
    backward, ok_backward, _ = cv2.calcOpticalFlowPyrLK(second, first, forward, None,
                                                         winSize=(21, 21), maxLevel=3)
    if backward is None or ok_backward is None:
        raise ValueError("Backward background tracking failed")
    valid = ((ok_forward.ravel() > 0) & (ok_backward.ravel() > 0)
             & (np.linalg.norm(backward.reshape(-1, 2) - corners.reshape(-1, 2), axis=1) < 1.5))
    src = corners.reshape(-1, 2)[valid] / scale
    dst = forward.reshape(-1, 2)[valid] / scale
    if len(src) < 60:
        raise ValueError("Too few stable temporal correspondences")
    if entry.get("tracking_steps", 0) >= 25:
        raise ValueError("Pose has been propagated for too many frames; relocalize")
    displacement = np.linalg.norm(dst - src, axis=1)
    H_still, mask_still = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    if H_still is not None and mask_still is not None:
        still = mask_still.ravel().astype(bool)
        area = cv2.contourArea(cv2.convexHull(src[still].astype(np.float32)))
        p95_step = float(np.percentile(displacement[still], 95)) if still.any() else math.inf
        cumulative_hold = float(entry.get("hold_cumulative_p95_px", 0.0)) + p95_step
        if (still.sum() >= 100 and still.mean() >= 0.75
                and area >= 0.08 * previous.shape[0] * previous.shape[1]
                and p95_step <= 2.5 and cumulative_hold <= 2.5):
            updated = dict(entry)
            updated.update({"decoded_image_sha256": decoded_hash(current),
                            "tracking_steps": entry.get("tracking_steps", 0) + 1,
                            "hold_cumulative_p95_px": cumulative_hold,
                            "status": "held_within_pixel_tolerance",
                            "last_tracking": {
                                "method": "near_static_hold",
                                "tracked_points": int(len(src)),
                                "static_inliers": int(still.sum()),
                                "median_displacement_px": float(np.median(displacement[still])),
                                "p95_displacement_px": p95_step,
                                "hold_cumulative_p95_px": cumulative_hold,
                                "pose_change_applied": False}})
            return updated
    world, on_ground_ray = _ground_world_points(src, entry, ground_z_m)
    world, src, dst = world[on_ground_ray], src[on_ground_ray], dst[on_ground_ray]
    if len(world) < 50:
        raise ValueError("Too few rays intersect the ground in front of the camera")
    K = np.asarray(entry["K"], dtype=float)
    D = np.asarray(entry["D"], dtype=float)
    old_R = np.asarray(entry["R"], dtype=float)
    old_t = np.asarray(entry["t_m"], dtype=float)
    old_rvec = cv2.Rodrigues(old_R)[0]
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        world.astype(np.float64), dst.astype(np.float64), K, D,
        rvec=old_rvec.copy(), tvec=old_t.reshape(3, 1).copy(),
        useExtrinsicGuess=True, iterationsCount=300, reprojectionError=3.0,
        confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
    pnp_ok = success and inliers is not None and len(inliers) >= 40
    if pnp_ok:
        keep = inliers.ravel()
        pnp_ok = len(keep) / len(world) >= 0.30
    if pnp_ok:
        hull_area = cv2.contourArea(cv2.convexHull(src[keep].astype(np.float32)))
        pnp_ok = hull_area >= 0.04 * previous.shape[0] * previous.shape[1]
    if pnp_ok:
        cv2.solvePnPRefineLM(world[keep].astype(np.float64), dst[keep].astype(np.float64),
                             K, D, rvec, tvec)
        projected = cv2.projectPoints(world[keep], rvec, tvec, K, D)[0].reshape(-1, 2)
        errors = np.linalg.norm(projected - dst[keep], axis=1)
        pnp_ok = float(np.median(errors)) <= 1.5 and float(np.percentile(errors, 95)) <= 3.5
    if pnp_ok:
        new_R = cv2.Rodrigues(rvec)[0]
        new_t = tvec.reshape(3)
        diagnostic = {"method": "ground_pnp", "tracked_points": int(len(src)),
                      "ground_inliers": int(len(keep)),
                      "median_reprojection_error_px": float(np.median(errors))}
    else:
        new_R, new_t, diagnostic = _homography_ground_pose(src, dst, entry, ground_z_m)
    old_center = -old_R.T @ old_t
    new_center = -new_R.T @ new_t
    step = float(np.linalg.norm(new_center - old_center))
    rotation_step = math.degrees(math.acos(float(np.clip(
        (np.trace(new_R @ old_R.T) - 1) / 2, -1, 1))))
    if step > max_step_m or abs(new_center[2] - old_center[2]) > 0.5 or rotation_step > 5:
        raise ValueError("Camera motion is outside the permitted adjacent-frame range")
    updated = dict(entry)
    diagnostic.update({"camera_step_m": step, "rotation_step_deg": rotation_step})
    updated.update({"R": new_R.tolist(), "t_m": new_t.tolist(),
                    "decoded_image_sha256": decoded_hash(current),
                    "tracking_steps": entry.get("tracking_steps", 0) + 1,
                    "hold_cumulative_p95_px": 0.0,
                    "status": "tracked_unverified",
                    "last_tracking": diagnostic})
    return updated


def update_pose_state(state: dict, frame_pairs: dict[str, tuple[np.ndarray, np.ndarray]],
                      max_step_m: float = 2.0) -> dict:
    updated = {**state, "cameras": dict(state["cameras"])}
    if not frame_pairs:
        raise ValueError("At least one camera frame pair is required")
    for name, (previous, current) in frame_pairs.items():
        if name not in state["cameras"]:
            raise ValueError(f"Camera {name} has no pose in state")
        updated["cameras"][name] = track_camera_pose(
            previous, current, state["cameras"][name], state["ground_z_m"], max_step_m)
    return updated


def validate_pose_frame(image: np.ndarray, entry: dict) -> None:
    if [image.shape[1], image.shape[0]] != entry["image_size"]:
        raise ValueError("Current image size differs from pose state")
    if decoded_hash(image) != entry["decoded_image_sha256"]:
        raise ValueError("Current image does not match pose state; update or relocalize first")


def _triangulate_state_pixels(reference_pixels: np.ndarray, target_pixels: np.ndarray,
                              reference: dict, target: dict) -> dict:
    a, b = np.asarray(reference_pixels, dtype=float), np.asarray(target_pixels, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or a.shape != b.shape or len(a) not in (1, 2):
        raise ValueError("One or two corresponding pixels are required in each camera")
    cameras = []
    for entry, pixels in ((reference, a), (target, b)):
        K = np.asarray(entry["K"], dtype=float)
        D = np.asarray(entry["D"], dtype=float)
        R = np.asarray(entry["R"], dtype=float)
        t = np.asarray(entry["t_m"], dtype=float)
        normalized = cv2.undistortPoints(pixels.reshape(-1, 1, 2), K, D).reshape(-1, 2)
        cameras.append((K, D, R, t, normalized, pixels))
    _, _, R1, t1, n1, _ = cameras[0]
    _, _, R2, t2, n2, _ = cameras[1]
    P1 = np.column_stack((R1, t1))
    P2 = np.column_stack((R2, t2))
    homogeneous = cv2.triangulatePoints(P1, P2, n1.T, n2.T)
    if np.any(abs(homogeneous[3]) < 1e-9):
        raise ValueError("Selected rays do not intersect stably")
    world = (homogeneous[:3] / homogeneous[3]).T
    errors = []
    for K, D, R, t, _, pixels in cameras:
        camera_points = (R @ world.T + t[:, None]).T
        if np.any(camera_points[:, 2] <= 0):
            raise ValueError("Triangulated point lies behind a camera")
        projected = cv2.projectPoints(world, cv2.Rodrigues(R)[0], t, K, D)[0].reshape(-1, 2)
        errors.append(float(np.linalg.norm(projected - pixels, axis=1).max()))
    if max(errors) > 3:
        raise ValueError("Matched points exceed three-pixel reprojection tolerance")
    centers = [-R.T @ t for _, _, R, t, _, _ in cameras]
    angles = []
    for point in world:
        rays = [(point - center) / np.linalg.norm(point - center) for center in centers]
        angles.append(math.degrees(math.acos(float(np.clip(rays[0] @ rays[1], -1, 1)))))
    if min(angles) < 1:
        raise ValueError("Stereo ray angle is too small")
    result = {"world_xyz_m": world.tolist(), "ray_angles_deg": angles,
            "reprojection_error_px": errors,
            "baseline_m": float(np.linalg.norm(centers[1] - centers[0]))}
    if len(world) == 2:
        result["distance_m"] = float(np.linalg.norm(world[1] - world[0]))
    return result


def triangulate_with_state(reference_pixels: np.ndarray, target_pixels: np.ndarray,
                           reference: dict, target: dict) -> dict:
    if np.asarray(reference_pixels).shape != (2, 2):
        raise ValueError("Two corresponding pixels are required in each camera")
    return _triangulate_state_pixels(reference_pixels, target_pixels, reference, target)


def triangulate_one_with_state(reference_pixel: np.ndarray, target_pixel: np.ndarray,
                               reference: dict, target: dict) -> dict:
    if np.asarray(reference_pixel).shape != (2,) or np.asarray(target_pixel).shape != (2,):
        raise ValueError("One corresponding pixel is required in each camera")
    result = _triangulate_state_pixels(np.asarray(reference_pixel)[None],
                                       np.asarray(target_pixel)[None], reference, target)
    result["world_xyz_m"] = result["world_xyz_m"][0]
    result["ray_angle_deg"] = result.pop("ray_angles_deg")[0]
    return result
