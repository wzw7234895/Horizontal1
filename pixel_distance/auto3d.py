"""Experimental cross-view correspondence and nominal-scale triangulation."""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .pose_state import (state_from_nominal_fixed_rig, state_from_nominal_group, triangulate_one_with_state,
                         triangulate_with_state, validate_pose_frame)
from .stereo_patch import match_rectified_patches


MAX_SIDE = 1280
MAX_MATCHER_DISAGREEMENT_PX = 8.0


def _valid_feature_mask(gray: np.ndarray) -> np.ndarray:
    mask = np.where(gray > 35, 255, 0).astype(np.uint8)
    height, width = gray.shape[:2]
    margin_y = max(1, round(height * 0.005))
    margin_x = max(1, round(width * 0.005))
    mask[:margin_y] = 0
    mask[-margin_y:] = 0
    mask[:, :margin_x] = 0
    mask[:, -margin_x:] = 0
    return mask


def _prepare(image: np.ndarray, max_side: int = MAX_SIDE,
             multiple: int = 1) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if scale < 1.0:
        resized_width = max(multiple, round(width * scale / multiple) * multiple)
        resized_height = max(multiple, round(height * scale / multiple) * multiple)
        gray = cv2.resize(gray, (resized_width, resized_height),
                          interpolation=cv2.INTER_AREA)
    return gray, np.array([gray.shape[1] / width, gray.shape[0] / height])


def _features(gray: np.ndarray):
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.02)
    return sift.detectAndCompute(gray, _valid_feature_mask(gray))


@lru_cache(maxsize=1)
def _loftr_model():
    import torch
    from kornia.feature import LoFTR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return LoFTR(pretrained="outdoor").eval().to(device), device


def _loftr_matches(gray_a: np.ndarray, gray_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model, device = _loftr_model()

    def image_tensor(gray):
        return torch.from_numpy(gray).float()[None, None].to(device) / 255.0

    def valid_mask(gray):
        mask = _valid_feature_mask(gray).astype(np.float32) / 255.0
        return torch.from_numpy(mask)[None].to(device)

    with torch.inference_mode():
        output = model({"image0": image_tensor(gray_a), "image1": image_tensor(gray_b),
                        "mask0": valid_mask(gray_a), "mask1": valid_mask(gray_b)})
    source = output["keypoints0"].cpu().numpy().astype(np.float64)
    target = output["keypoints1"].cpu().numpy().astype(np.float64)
    if len(source) < 24:
        raise ValueError(f"Too few LoFTR matches ({len(source)})")
    return source, target


@lru_cache(maxsize=1)
def _xfeat_model(repo: str, weights: str):
    import torch

    model = torch.hub.load(repo, "XFeat", pretrained=False, top_k=4096,
                           source="local")
    state = torch.load(weights, map_location=model.dev, weights_only=True)
    model.net.load_state_dict(state)
    return model.eval()


def _xfeat_matches(image_a: np.ndarray, image_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match two resized BGR frames with local, explicitly supplied model files."""
    import torch

    repo = os.environ.get("PIXEL_DISTANCE_XFEAT_REPO")
    weights = os.environ.get("PIXEL_DISTANCE_XFEAT_WEIGHTS")
    if not repo or not weights or not Path(repo).is_dir() or not Path(weights).is_file():
        raise ValueError("XFeat 需设置本地 PIXEL_DISTANCE_XFEAT_REPO 和 PIXEL_DISTANCE_XFEAT_WEIGHTS")
    model = _xfeat_model(repo, weights)
    rgb_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB)
    rgb_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB)
    with torch.inference_mode():
        source, target = model.match_xfeat(rgb_a, rgb_b, top_k=4096,
                                            min_cossim=0.82)
    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 2)
    if source.shape != target.shape:
        raise ValueError("XFeat 返回的对应点数量不一致")
    if not len(source):
        raise ValueError("XFeat 未找到跨视角对应点")
    valid = np.ones(len(source), dtype=bool)
    for points, image in ((source, image_a), (target, image_b)):
        height, width = image.shape[:2]
        valid &= ((points[:, 0] >= 0) & (points[:, 0] < width)
                  & (points[:, 1] >= 0) & (points[:, 1] < height))
        x = np.clip(np.rint(points[:, 0]).astype(int), 0, width - 1)
        y = np.clip(np.rint(points[:, 1]).astype(int), 0, height - 1)
        valid &= _valid_feature_mask(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))[y, x] > 0
    source, target = source[valid], target[valid]
    if len(source) < 2:
        raise ValueError(f"XFeat 有效跨视角候选不足（{len(source)}）")
    return source, target


def _xfeat_inputs(reference: np.ndarray, target: np.ndarray):
    ref_gray, ref_scale = _prepare(reference, 960)
    target_gray, target_scale = _prepare(target, 960)
    ref_color = cv2.resize(reference, (ref_gray.shape[1], ref_gray.shape[0]),
                           interpolation=cv2.INTER_AREA)
    target_color = cv2.resize(target, (target_gray.shape[1], target_gray.shape[0]),
                              interpolation=cv2.INTER_AREA)
    return ref_gray, target_gray, ref_scale, target_scale, ref_color, target_color


def _intrinsics(camera: dict, width: int, height: int) -> np.ndarray:
    fov = float(camera["horizontal_fov_deg"])
    if not 0 < fov < 170:
        raise ValueError("Invalid camera field of view")
    focal = width / (2 * math.tan(math.radians(fov) / 2))
    return np.array([[focal, 0, (width - 1) / 2],
                     [0, focal, (height - 1) / 2], [0, 0, 1]], dtype=np.float64)


def _symmetric_sift_matches(features_a, features_b,
                            minimum: int = 24) -> tuple[np.ndarray, np.ndarray]:
    keys_a, desc_a = features_a
    keys_b, desc_b = features_b
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        raise ValueError("Insufficient texture features")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(desc_a, desc_b, k=2)
    reverse = matcher.knnMatch(desc_b, desc_a, k=2)
    good_forward = {first.queryIdx: first.trainIdx for first, second in forward
                    if first.distance < 0.75 * second.distance}
    good_reverse = {first.queryIdx: first.trainIdx for first, second in reverse
                    if first.distance < 0.75 * second.distance}
    pairs = [(a, b) for a, b in good_forward.items() if good_reverse.get(b) == a]
    if len(pairs) < minimum:
        raise ValueError(f"Too few symmetric feature matches ({len(pairs)})")
    source = np.asarray([keys_a[a].pt for a, _ in pairs], dtype=np.float64)
    target = np.asarray([keys_b[b].pt for _, b in pairs], dtype=np.float64)
    return source, target


def _local_match(point: np.ndarray, source: np.ndarray, target: np.ndarray,
                 fundamental: np.ndarray, image_width: int) -> tuple[np.ndarray, dict]:
    distances = np.linalg.norm(source - point, axis=1)
    nearest = int(np.argmin(distances))
    if distances[nearest] <= 2.0:
        mapped = target[nearest]
        line = fundamental @ np.array([point[0], point[1], 1.0])
        denominator = math.hypot(line[0], line[1])
        if denominator > 1e-8:
            epipolar_error = abs(float(np.dot(line, [mapped[0], mapped[1], 1]))) / denominator
            if epipolar_error <= 2.0:
                return mapped, {"method": "nearby_matched_feature",
                                "nearest_feature_px": float(distances[nearest]),
                                "epipolar_error_px": epipolar_error}
    nearby = np.flatnonzero(distances <= (150.0 if image_width >= 1000 else 110.0))
    if len(nearby) < 8 or float(distances.min()) > 50.0:
        raise ValueError("Selected point lacks nearby cross-view features")
    nearby = nearby[np.argsort(distances[nearby])[:40]]
    h, mask = cv2.findHomography(source[nearby], target[nearby], cv2.RANSAC, 4.0)
    if h is None or mask is None or int(mask.sum()) < 7:
        raise ValueError("Local geometry is not consistent")
    inliers = mask.ravel().astype(bool)
    support = source[nearby][inliers].astype(np.float32)
    if cv2.contourArea(cv2.convexHull(support)) < 100:
        raise ValueError("Local features do not span an area")
    hull = cv2.convexHull(support)
    if cv2.pointPolygonTest(hull, tuple(map(float, point)), False) < 0:
        raise ValueError("Selected point lies outside local feature support")
    mapped = cv2.perspectiveTransform(point.reshape(1, 1, 2), h).reshape(2)
    line = fundamental @ np.array([point[0], point[1], 1.0])
    denominator = math.hypot(line[0], line[1])
    if denominator < 1e-8:
        raise ValueError("Degenerate epipolar geometry")
    epipolar_error = abs(float(np.dot(line, [mapped[0], mapped[1], 1]))) / denominator
    if epipolar_error > 3.0:
        raise ValueError(f"Epipolar error {epipolar_error:.1f} px is too large")
    predicted = cv2.perspectiveTransform(source[nearby][inliers].reshape(-1, 1, 2), h).reshape(-1, 2)
    residual = np.linalg.norm(predicted - target[nearby][inliers], axis=1)
    if float(np.median(residual)) > 2.5:
        raise ValueError("Local transfer residual is too large")
    return mapped, {"local_inliers": int(inliers.sum()),
                    "method": "local_homography",
                    "nearest_feature_px": float(distances.min()),
                    "epipolar_error_px": epipolar_error,
                    "local_median_residual_px": float(np.median(residual))}


def _contextual_transfer(point: np.ndarray, source: np.ndarray,
                         target: np.ndarray) -> tuple[np.ndarray, dict]:
    """Transfer a point only when nearby whole-image matches support one surface."""
    distances = np.linalg.norm(source - point, axis=1)
    nearby = np.flatnonzero(distances <= 240.0)
    if len(nearby) < 8 or float(distances.min()) > 55.0:
        raise ValueError("选点附近缺少足够的跨视角环境特征")
    nearby = nearby[np.argsort(distances[nearby])[:60]]
    src, dst = source[nearby], target[nearby]
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if homography is None or mask is None or int(mask.sum()) < 8:
        raise ValueError("选点附近未形成一致的局部表面映射")
    inliers = mask.ravel().astype(bool)
    support = src[inliers].astype(np.float32)
    hull = cv2.convexHull(support)
    if cv2.contourArea(hull) < 200 or cv2.pointPolygonTest(
            hull, tuple(map(float, point)), False) < 0:
        raise ValueError("选点不在局部表面特征覆盖范围内")
    predicted = cv2.perspectiveTransform(support.reshape(-1, 1, 2),
                                         homography).reshape(-1, 2)
    residuals = np.linalg.norm(predicted - dst[inliers], axis=1)
    if float(np.median(residuals)) > 2.5 or float(np.percentile(residuals, 90)) > 5:
        raise ValueError("局部环境映射残差过大")
    mapped = cv2.perspectiveTransform(point.reshape(1, 1, 2),
                                      homography).reshape(2)
    reverse, _ = cv2.findHomography(dst[inliers], src[inliers], 0)
    if reverse is None:
        raise ValueError("局部环境映射无法反向检查")
    returned = cv2.perspectiveTransform(mapped.reshape(1, 1, 2),
                                        reverse).reshape(2)
    round_trip = float(np.linalg.norm(returned - point))
    if round_trip > 2.5:
        raise ValueError("局部环境映射双向回查不一致")
    return mapped, {"method": "contextual_local_homography",
                    "local_inliers": int(inliers.sum()),
                    "nearest_feature_px": float(distances.min()),
                    "local_median_residual_px": float(np.median(residuals)),
                    "round_trip_error_px": round_trip}


def _match_view_with_method(reference: np.ndarray, target: np.ndarray,
                            points: np.ndarray, reference_camera: dict,
                            target_camera: dict, method: str,
                            prepared_reference=None,
                            allow_partial: bool = False) -> dict:
    if method == "xfeat":
        ref_gray, target_gray, ref_scale, target_scale, ref_color, target_color = (
            _xfeat_inputs(reference, target))
        source, destination = _xfeat_matches(ref_color, target_color)
    elif method == "loftr":
        ref_gray, ref_scale = _prepare(reference, 640, 8)
        target_gray, target_scale = _prepare(target, 640, 8)
        source, destination = _loftr_matches(ref_gray, target_gray)
    else:
        if prepared_reference is None:
            ref_gray, ref_scale = _prepare(reference)
            ref_features = _features(ref_gray)
        else:
            ref_gray, ref_scale, ref_features = prepared_reference
        target_gray, target_scale = _prepare(target)
        source, destination = _symmetric_sift_matches(ref_features, _features(target_gray))
    fundamental, mask = cv2.findFundamentalMat(
        source, destination, cv2.FM_RANSAC, 2.0, 0.999)
    if fundamental is None or mask is None or fundamental.shape != (3, 3):
        raise ValueError("Cannot estimate two-view epipolar geometry")
    inliers = mask.ravel().astype(bool)
    if int(inliers.sum()) < 20:
        raise ValueError("Too few epipolar inliers")
    source, destination = source[inliers], destination[inliers]
    query = np.asarray(points, dtype=np.float64) * ref_scale
    transferred, checks = [], []
    for point in query:
        try:
            mapped, diagnostic = _local_match(point, source, destination, fundamental,
                                              ref_gray.shape[1])
            if not 0 <= mapped[0] < target_gray.shape[1] or not 0 <= mapped[1] < target_gray.shape[0]:
                raise ValueError("Matched point falls outside target image")
            transferred.append((mapped / target_scale).tolist())
            checks.append({"status": "matched_candidate", **diagnostic})
        except ValueError as exc:
            if not allow_partial:
                raise
            transferred.append(None)
            checks.append({"status": "unknown", "reason": str(exc)})
    return {
        "status": "partial_coverage" if any(point is None for point in transferred)
                  else "matched_candidate",
        "matcher": method,
        "pixel_points": transferred,
        "matched_point_count": sum(item is not None for item in transferred),
        "feature_matches": int(len(inliers)),
        "epipolar_inliers": int(inliers.sum()),
        "point_checks": checks,
        "_source_inliers": source,
        "_target_inliers": destination,
        "_reference_scale": ref_scale,
        "_target_scale": target_scale,
        "_reference_K": _intrinsics(reference_camera, ref_gray.shape[1], ref_gray.shape[0]),
        "_target_K": _intrinsics(target_camera, target_gray.shape[1], target_gray.shape[0]),
    }


def match_view(reference: np.ndarray, target: np.ndarray,
               points: np.ndarray, reference_camera: dict,
               target_camera: dict, matcher: str = "auto",
               prepared_reference=None, allow_partial: bool = False) -> dict:
    if matcher not in ("auto", "sift", "loftr", "xfeat"):
        raise ValueError("Matcher must be auto, sift, loftr, or xfeat")
    methods = ("sift", "loftr") if matcher == "auto" else (matcher,)
    failures = []
    candidates = []
    for method in methods:
        try:
            candidate = _match_view_with_method(
                reference, target, points, reference_camera, target_camera,
                method, prepared_reference, allow_partial=(matcher == "auto" or allow_partial))
            candidates.append(candidate)
            if all(pixel is not None for pixel in candidate["pixel_points"]):
                break
        except (ValueError, cv2.error, ImportError, RuntimeError) as exc:
            failures.append(f"{method}: {exc}")
    if not candidates:
        raise ValueError("; ".join(failures))
    if matcher != "auto" or len(candidates) == 1:
        result = candidates[0]
    else:
        pixels, checks = [], []
        for index in range(len(points)):
            found = [(candidate["pixel_points"][index], candidate["point_checks"][index],
                      candidate["matcher"]) for candidate in candidates
                     if candidate["pixel_points"][index] is not None]
            if len(found) > 1 and np.linalg.norm(
                    np.asarray(found[0][0]) - np.asarray(found[1][0])) > MAX_MATCHER_DISAGREEMENT_PX:
                pixels.append(None)
                checks.append({"status": "unknown", "reason": "不同视觉方法候选位置冲突"})
            elif found:
                pixels.append(found[0][0])
                checks.append({**found[0][1], "source_matcher": found[0][2]})
            else:
                pixels.append(None)
                checks.append(candidates[-1]["point_checks"][index])
        result = {**{key: value for key, value in candidates[0].items()
                     if not key.startswith("_")},
                  "matcher": "auto_combined", "pixel_points": pixels,
                  "point_checks": checks, "matched_point_count": sum(p is not None for p in pixels),
                  "status": "matched_candidate" if all(p is not None for p in pixels)
                            else "partial_coverage",
                  "matcher_attempts": [{"matcher": item["matcher"],
                                        "matched_point_count": item["matched_point_count"]}
                                       for item in candidates]}
    if not allow_partial and any(pixel is None for pixel in result["pixel_points"]):
        raise ValueError("自动匹配未同时确认两个选点")
    return result


def _fundamental_from_pose(reference_entry: dict, target_entry: dict) -> np.ndarray:
    k_ref = np.asarray(reference_entry["K"], dtype=float)
    k_target = np.asarray(target_entry["K"], dtype=float)
    r_ref = np.asarray(reference_entry["R"], dtype=float)
    r_target = np.asarray(target_entry["R"], dtype=float)
    t_ref = np.asarray(reference_entry["t_m"], dtype=float)
    t_target = np.asarray(target_entry["t_m"], dtype=float)
    rotation = r_target @ r_ref.T
    translation = t_target - rotation @ t_ref
    if np.linalg.norm(translation) < 1e-4:
        raise ValueError("Camera baseline in pose state is too small")
    x, y, z = translation
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    return np.linalg.inv(k_target).T @ cross @ rotation @ np.linalg.inv(k_ref)


def match_view_with_pose(reference: np.ndarray, target: np.ndarray,
                         points: np.ndarray, reference_entry: dict,
                         target_entry: dict, matcher: str = "sift",
                         prepared_reference=None, allow_partial: bool = False,
                         candidate_epipolar_px: float = 3.0) -> dict:
    """Use a stored relative pose to filter correspondences before local transfer."""
    if not math.isfinite(candidate_epipolar_px) or candidate_epipolar_px <= 0:
        raise ValueError("极线候选容差必须为正数")
    if matcher == "auto":
        pixels = [None, None]
        checks = [{"status": "unknown", "reason": "尚无可靠跨视角对应"}
                  for _ in range(2)]
        conflicts = [False, False]
        attempts = []
        for method in ("sift", "loftr", "patch"):
            try:
                candidate = match_view_with_pose(
                    reference, target, points, reference_entry, target_entry,
                    method, prepared_reference, allow_partial=True,
                    candidate_epipolar_px=candidate_epipolar_px)
            except (ValueError, cv2.error, ImportError, RuntimeError) as exc:
                attempts.append({"matcher": method, "reason": str(exc)})
                continue
            attempts.append({"matcher": method, "status": candidate["status"],
                             "point_checks": candidate["point_checks"]})
            for index, pixel in enumerate(candidate["pixel_points"]):
                if pixel is None:
                    if pixels[index] is None and not conflicts[index]:
                        checks[index] = candidate["point_checks"][index]
                    continue
                if conflicts[index]:
                    continue
                if pixels[index] is not None:
                    difference = float(np.linalg.norm(np.asarray(pixel) - pixels[index]))
                    if difference > MAX_MATCHER_DISAGREEMENT_PX:
                        pixels[index] = None
                        conflicts[index] = True
                        checks[index] = {"status": "unknown",
                                         "reason": "不同视觉方法的同名点候选不一致",
                                         "candidate_disagreement_px": difference}
                    continue
                pixels[index] = pixel
                checks[index] = {**candidate["point_checks"][index],
                                 "source_matcher": method}
        if not allow_partial and any(pixel is None for pixel in pixels):
            raise ValueError("自动匹配未同时确认两个选点")
        used = {check.get("source_matcher") for pixel, check in zip(pixels, checks)
                if pixel is not None}
        return {"status": ("matched_candidate" if all(p is not None for p in pixels)
                           else "partial_coverage" if any(p is not None for p in pixels)
                           else "unknown"),
                "matcher": (next(iter(used)) if len(used) == 1 else
                            "auto_combined" if used else "auto"),
                "pixel_points": pixels,
                "point_checks": checks, "matcher_attempts": attempts}
    if matcher == "patch":
        return match_rectified_patches(reference, target, points,
                                       reference_entry, target_entry,
                                       allow_partial=allow_partial)
    if matcher == "roma":
        from .roma_probe import probe_with_entries

        diagnostic = probe_with_entries(reference, target, np.asarray(points, dtype=float),
                                        reference_entry, target_entry)
        accepted = [item["visual_consistent"] and item["nominal_geometry_consistent"]
                    for item in diagnostic["points"]]
        pixel_points = [item["target_candidate_pixel"] if ok else None
                        for item, ok in zip(diagnostic["points"], accepted)]
        checks = [{"status": "matched_candidate" if ok else "unknown",
                   **item} for item, ok in zip(diagnostic["points"], accepted)]
        if not allow_partial and not all(accepted):
            raise ValueError("RoMa 候选未通过双向和名义几何检查")
        return {"status": ("matched_candidate" if all(accepted) else
                           "partial_coverage" if any(accepted) else "unknown"),
                "matcher": "roma", "pixel_points": pixel_points,
                "point_checks": checks}
    if matcher not in ("sift", "loftr", "xfeat"):
        raise ValueError("Matcher must be auto, sift, loftr, xfeat, patch, or roma")
    fundamental = _fundamental_from_pose(reference_entry, target_entry)
    methods = (matcher,)
    failures = []
    for method in methods:
        try:
            if method == "patch":
                candidate = match_rectified_patches(
                    reference, target, points, reference_entry, target_entry,
                    allow_partial=allow_partial)
                if candidate["status"] == "unknown":
                    reasons = "; ".join(check["reason"] for check in candidate["point_checks"])
                    raise ValueError(reasons)
                return candidate
            if method == "xfeat":
                gray_ref, gray_target, scale_ref, scale_target, ref_color, target_color = (
                    _xfeat_inputs(reference, target))
                source, destination = _xfeat_matches(ref_color, target_color)
            elif method == "loftr":
                gray_ref, scale_ref = _prepare(reference, 640, 8)
                gray_target, scale_target = _prepare(target, 640, 8)
                source, destination = _loftr_matches(gray_ref, gray_target)
            else:
                if prepared_reference is None:
                    gray_ref, scale_ref = _prepare(reference)
                    features = _features(gray_ref)
                else:
                    gray_ref, scale_ref, features = prepared_reference
                gray_target, scale_target = _prepare(target)
                source, destination = _symmetric_sift_matches(
                    features, _features(gray_target), minimum=2)
            source /= scale_ref
            destination /= scale_target
            k_ref = np.asarray(reference_entry["K"], dtype=float)
            k_target = np.asarray(target_entry["K"], dtype=float)
            d_ref = np.asarray(reference_entry["D"], dtype=float)
            d_target = np.asarray(target_entry["D"], dtype=float)
            source = cv2.undistortPoints(source.reshape(-1, 1, 2), k_ref,
                                         d_ref, P=k_ref).reshape(-1, 2)
            destination = cv2.undistortPoints(destination.reshape(-1, 1, 2), k_target,
                                              d_target, P=k_target).reshape(-1, 2)
            lines = (fundamental @ np.column_stack((source, np.ones(len(source)))).T).T
            errors = np.abs(np.sum(lines * np.column_stack((destination,
                                                            np.ones(len(destination)))), axis=1))
            errors /= np.maximum(np.linalg.norm(lines[:, :2], axis=1), 1e-12)
            keep = errors <= candidate_epipolar_px
            pose_source, pose_destination = source[keep], destination[keep]
            mapped, checks = [], []
            for point in np.asarray(points, dtype=float):
                try:
                    corrected_point = cv2.undistortPoints(point.reshape(1, 1, 2),
                                                           k_ref, d_ref, P=k_ref).reshape(2)
                    try:
                        if len(pose_source) < 2:
                            raise ValueError(
                                f"名义极线候选特征不足（{len(pose_source)}/{len(source)}；"
                                f"最小误差 {float(errors.min()):.1f} 像素）")
                        location, check = _local_match(
                            corrected_point, pose_source, pose_destination,
                            fundamental, reference.shape[1])
                    except ValueError as geometry_error:
                        try:
                            location, check = _contextual_transfer(
                                corrected_point, source, destination)
                        except ValueError as context_error:
                            raise ValueError(f"{geometry_error}; {context_error}") from context_error
                        line = fundamental @ np.r_[corrected_point, 1.0]
                        epipolar_error = abs(float(line @ np.r_[location, 1.0])) / max(
                            np.linalg.norm(line[:2]), 1e-12)
                        check["nominal_epipolar_error_px"] = epipolar_error
                        check["nominal_geometry_reason"] = str(geometry_error)
                        if epipolar_error > 3.0:
                            normalized = np.array([(location[0] - k_target[0, 2]) / k_target[0, 0],
                                                   (location[1] - k_target[1, 2]) / k_target[1, 1], 1.0])
                            visual_pixel = cv2.projectPoints(normalized.reshape(1, 3),
                                                              np.zeros(3), np.zeros(3),
                                                              k_target, d_target)[0].reshape(2)
                            mapped.append(None)
                            checks.append({"status": "unknown",
                                           "reason": "局部环境候选与当前相机位姿不一致",
                                           "visual_candidate_pixel": visual_pixel.tolist(), **check})
                            continue
                    normalized = np.array([(location[0] - k_target[0, 2]) / k_target[0, 0],
                                           (location[1] - k_target[1, 2]) / k_target[1, 1], 1.0])
                    location = cv2.projectPoints(normalized.reshape(1, 3),
                                                  np.zeros(3), np.zeros(3),
                                                  k_target, d_target)[0].reshape(2)
                    if not (0 <= location[0] < target.shape[1]
                            and 0 <= location[1] < target.shape[0]):
                        raise ValueError("Matched point falls outside target image")
                    mapped.append(location.tolist())
                    checks.append({"status": "matched_candidate", **check})
                except ValueError as exc:
                    if not allow_partial:
                        raise
                    mapped.append(None)
                    checks.append({"status": "unknown", "reason": str(exc)})
            if not any(point is not None for point in mapped):
                if not allow_partial:
                    raise ValueError("Neither selected point has a reliable local match")
                return {"status": "unknown", "matcher": method,
                        "pixel_points": mapped, "epipolar_candidates": int(keep.sum()),
                        "epipolar_candidate_threshold_px": candidate_epipolar_px,
                        "point_checks": checks}
            return {"status": "matched_candidate" if all(point is not None for point in mapped)
                    else "partial_coverage", "matcher": method,
                    "pixel_points": mapped, "epipolar_candidates": int(keep.sum()),
                    "epipolar_candidate_threshold_px": candidate_epipolar_px,
                    "point_checks": checks}
        except (ValueError, cv2.error, ImportError, RuntimeError) as exc:
            failures.append(f"{method}: {exc}")
    raise ValueError("; ".join(failures))


def _nominal_baseline(first: dict, second: dict, rig_origins: dict) -> float:
    a = (np.asarray(first["center_in_rig_m"], dtype=float)
         + np.asarray(rig_origins[first["rig"]], dtype=float))
    b = (np.asarray(second["center_in_rig_m"], dtype=float)
         + np.asarray(rig_origins[second["rig"]], dtype=float))
    if a.shape != (3,) or b.shape != (3,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid nominal camera center coordinates")
    baseline = float(np.linalg.norm(a - b))
    if baseline <= 0:
        raise ValueError("Nominal camera centers must differ")
    return baseline


def _triangulate_pair(selected: np.ndarray, matched: dict, baseline_m: float) -> dict:
    result = _triangulate_points(selected, matched, baseline_m)
    result["distance_m"] = float(np.linalg.norm(
        np.asarray(result["reference_frame_xyz_m"])[1] -
        np.asarray(result["reference_frame_xyz_m"])[0]))
    return result


def _triangulate_points(selected: np.ndarray, matched: dict, baseline_m: float) -> dict:
    selected = np.asarray(selected, dtype=np.float64)
    if selected.ndim != 2 or selected.shape[1] != 2 or len(selected) < 2:
        raise ValueError("At least two corresponding points are required")
    if np.asarray(matched["pixel_points"]).shape != selected.shape:
        raise ValueError("Reference and target point counts differ")
    source = matched["_source_inliers"]
    target = matched["_target_inliers"]
    k_ref = matched["_reference_K"]
    k_tgt = matched["_target_K"]
    norm_ref = cv2.undistortPoints(source.reshape(-1, 1, 2), k_ref, None).reshape(-1, 2)
    norm_tgt = cv2.undistortPoints(target.reshape(-1, 1, 2), k_tgt, None).reshape(-1, 2)
    essential, mask = cv2.findEssentialMat(norm_ref, norm_tgt, np.eye(3),
                                            cv2.RANSAC, 0.999, 0.003)
    if essential is None or mask is None:
        raise ValueError("Relative orientation could not be estimated")
    if essential.shape[0] > 3:
        essential = essential[:3, :]
    positive, rotation, translation, _ = cv2.recoverPose(
        essential, norm_ref, norm_tgt, np.eye(3), mask=mask)
    if positive < 12:
        raise ValueError("Insufficient points in front of both cameras")
    translation = translation.reshape(3) * baseline_m
    p_ref = np.column_stack((np.eye(3), np.zeros(3)))
    p_tgt = np.column_stack((rotation, translation))
    selected_scaled = selected * matched["_reference_scale"]
    target_scaled = np.asarray(matched["pixel_points"]) * matched["_target_scale"]
    a = cv2.undistortPoints(selected_scaled.reshape(-1, 1, 2), k_ref, None).reshape(-1, 2)
    b = cv2.undistortPoints(target_scaled.reshape(-1, 1, 2), k_tgt, None).reshape(-1, 2)
    homogeneous = cv2.triangulatePoints(p_ref, p_tgt, a.T, b.T)
    if np.any(np.abs(homogeneous[3]) < 1e-9):
        raise ValueError("Selected rays do not have a stable intersection")
    xyz = (homogeneous[:3] / homogeneous[3]).T
    target_xyz = (rotation @ xyz.T + translation.reshape(3, 1)).T
    if np.any(xyz[:, 2] <= 0) or np.any(target_xyz[:, 2] <= 0):
        raise ValueError("A selected point lies behind a camera")
    center_tgt = -rotation.T @ translation
    angles = []
    for point in xyz:
        ray_ref = point / np.linalg.norm(point)
        ray_tgt = (point - center_tgt) / np.linalg.norm(point - center_tgt)
        angles.append(math.degrees(math.acos(float(np.clip(ray_ref @ ray_tgt, -1, 1)))))
    if min(angles) < 1.0:
        raise ValueError("Stereo ray angle is too small for stable distance")
    reproj_ref = xyz[:, :2] / xyz[:, 2, None]
    reproj_tgt = target_xyz[:, :2] / target_xyz[:, 2, None]
    errors = [float(np.linalg.norm(reproj_ref - a, axis=1).max() * k_ref[0, 0]),
              float(np.linalg.norm(reproj_tgt - b, axis=1).max() * k_tgt[0, 0])]
    if max(errors) > 3.0:
        raise ValueError("Selected points exceed stereo reprojection tolerance")
    return {"reference_frame_xyz_m": xyz.tolist(), "nominal_baseline_m": baseline_m,
            "ray_angles_deg": angles, "reprojection_error_px": errors,
            "pose_support_count": int(positive)}


def auto_measure(images: dict[str, np.ndarray], reference_id: str,
                 selected: np.ndarray, config: dict, matcher: str = "sift",
                 trolley_origin_m: tuple[float, float, float] | None = None,
                 pose_state: dict | None = None,
                 allow_diagnostic_pose: bool = False,
                 height_alignment: dict | None = None) -> dict:
    if reference_id not in images:
        raise ValueError("参考相机没有可用图像")
    if (pose_state is not None
            and pose_state.get("method") == "vehicle_length_diagnostic_only"
            and not allow_diagnostic_pose):
        raise ValueError("车长诊断位姿不能用于正式测距；仅离线试验可显式开启诊断开关")
    group_only = config.get("all_cameras_rigid") is False
    if trolley_origin_m is not None and group_only:
        raise ValueError("当前组内测距不使用小车相对大车位置；跨组只支持独立测高")
    if group_only:
        return _auto_measure_independent_groups(
            images, reference_id, selected, config, matcher, pose_state,
            height_alignment)
    excluded = {}
    if group_only:
        reference_group = config["cameras"][reference_id]["rig"]
        excluded = {name: {"status": "excluded_cross_group",
                           "reason": "两安装组相对水平位置未知，仅允许组内三维测距"}
                    for name in images if config["cameras"][name]["rig"] != reference_group}
        images = {name: image for name, image in images.items() if name not in excluded}
    if len(images) < 2:
        return {"status": "unknown", "reason": "参考相机所在组缺少另一台可用相机",
                "reference_camera": reference_id, "views": excluded,
                "distance_m": None, "height_difference_m": None}
    if pose_state is None and trolley_origin_m is None and config.get("all_cameras_rigid"):
        pose_state = state_from_nominal_fixed_rig(images, config)
    elif pose_state is None and len(images) > 1 and group_only:
        pose_state = state_from_nominal_group(images, config)
    if matcher == "patch" and pose_state is None:
        raise ValueError("极线图像块匹配需要当前帧的共享位姿状态")
    if reference_id not in images or len(images) < 2:
        raise ValueError("Reference and at least one other image are required")
    cameras = config["cameras"]
    if any(camera_id not in cameras for camera_id in images):
        raise ValueError("Unknown camera ID")
    selected = np.asarray(selected, dtype=float)
    ref = images[reference_id]
    if selected.shape != (2, 2) or not np.isfinite(selected).all():
        raise ValueError("Select A and B pixels in the reference image")
    if np.any(selected < 0) or np.any(selected[:, 0] >= ref.shape[1]) or np.any(selected[:, 1] >= ref.shape[0]):
        raise ValueError("Selected pixel lies outside reference image")
    if pose_state is not None:
        if reference_id not in pose_state.get("cameras", {}):
            raise ValueError("Reference camera has no current pose state")
        validate_pose_frame(ref, pose_state["cameras"][reference_id])
    rig_origins = dict(config["rig_origins_m"])
    if trolley_origin_m is not None:
        origin = np.asarray(trolley_origin_m, dtype=float)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("Current trolley origin must be three finite metre coordinates")
        rig_origins["trolley"] = origin.tolist()
    matches, metric = dict(excluded), []
    point_estimates = [[], []]
    prepared_reference = None

    def reference_features():
        nonlocal prepared_reference
        if prepared_reference is None and matcher in ("sift", "auto"):
            ref_gray, ref_scale = _prepare(ref)
            prepared_reference = (ref_gray, ref_scale, _features(ref_gray))
        return prepared_reference

    for camera_id, image in images.items():
        if camera_id == reference_id:
            continue
        cross_rig_without_origin = (cameras[reference_id]["rig"] != cameras[camera_id]["rig"]
                                    and trolley_origin_m is None)
        try:
            if pose_state is not None:
                if camera_id not in pose_state["cameras"]:
                    raise ValueError("Target camera has no current pose state")
                validate_pose_frame(image, pose_state["cameras"][camera_id])
                candidate = match_view_with_pose(
                    ref, image, selected, pose_state["cameras"][reference_id],
                    pose_state["cameras"][camera_id], matcher, reference_features(),
                    allow_partial=True,
                    candidate_epipolar_px=float(config.get("epipolar_candidate_threshold_px", 3.0)))
            else:
                candidate = match_view(ref, image, selected, cameras[reference_id],
                                       cameras[camera_id], matcher, reference_features())
            output = {key: value for key, value in candidate.items() if not key.startswith("_")}
            if pose_state is not None:
                for index, pixel in enumerate(candidate["pixel_points"]):
                    if pixel is None:
                        continue
                    try:
                        estimate = triangulate_one_with_state(
                            selected[index], pixel,
                            pose_state["cameras"][reference_id],
                            pose_state["cameras"][camera_id])
                        point_estimates[index].append((camera_id, estimate))
                        output["point_checks"][index]["metric"] = estimate
                    except (ValueError, cv2.error) as exc:
                        output["point_checks"][index] = {
                            **output["point_checks"][index], "status": "unknown",
                            "metric_error": str(exc),
                            "reason": "视觉候选未通过三角测量检查"}
                if all(check["status"] == "unknown" for check in output["point_checks"]):
                    output["status"] = "unknown"
                elif any(check["status"] == "unknown" for check in output["point_checks"]):
                    output["status"] = "partial_coverage"
                else:
                    measurement = triangulate_with_state(
                        selected, candidate["pixel_points"],
                        pose_state["cameras"][reference_id],
                        pose_state["cameras"][camera_id])
                    output["metric"] = measurement
                    metric.append((camera_id, measurement))
                matches[camera_id] = output
                continue
            if cross_rig_without_origin:
                matches[camera_id] = {
                    **output, "status": "matched_no_scale",
                    "reason": "Current relative position of trolley and gantry is required for cross-rig metre scale",
                }
                continue
            baseline = _nominal_baseline(cameras[reference_id], cameras[camera_id],
                                         rig_origins)
            try:
                measurement = _triangulate_pair(selected, candidate, baseline)
                output["metric"] = measurement
                metric.append((camera_id, measurement))
            except (ValueError, cv2.error) as exc:
                output["metric_error"] = str(exc)
            matches[camera_id] = output
        except (ValueError, cv2.error) as exc:
            matches[camera_id] = {"status": "unknown", "reason": str(exc)}
    if pose_state is not None and all(point_estimates):
        worlds = []
        for estimates in point_estimates:
            coordinates = np.asarray([item[1]["world_xyz_m"] for item in estimates])
            center = np.median(coordinates, axis=0)
            if len(coordinates) > 1 and np.max(np.linalg.norm(coordinates - center, axis=1)) > 0.25:
                return {"status": "unknown", "reason": "Stereo views disagree on a selected 3D point",
                        "reference_camera": reference_id, "reference_pixels": selected.tolist(),
                        "views": matches}
            worlds.append(center)
        support = {label: [camera_id for camera_id, _ in estimates]
                   for label, estimates in zip(("A", "B"), point_estimates)}
        return {"status": "nominal_3d_candidate_not_calibrated",
                "distance_m": float(np.linalg.norm(worlds[1] - worlds[0])),
                "height_difference_m": float(worlds[1][2] - worlds[0][2]),
                "height_difference_definition": "B_world_z_minus_A_world_z",
                "scale_source": "current_pose_state",
                "pose_state_method": pose_state["method"],
                "pose_state_scale_status": pose_state["scale_status"],
                "reference_camera": reference_id, "reference_pixels": selected.tolist(),
                "world_xyz_m": [point.tolist() for point in worlds],
                "point_support": support,
                "metric_pairs": [camera_id for camera_id, _ in metric],
                "pair_count": len(set(support["A"] + support["B"])),
                "cross_view_consistency_checked": all(len(estimates) > 1 for estimates in point_estimates),
                "views": matches,
                "limitations": ["Pose-state scale and drift need independent validation.",
                                "Feature matches may refer to different physical surface points.",
                                "A single stereo pair per point cannot provide independent agreement.",
                                "Exposure times must be synchronized for moving objects."]}
    if not metric:
        return {"status": "unknown", "reason": "No reliable stereo measurement",
                "reference_camera": reference_id, "reference_pixels": selected.tolist(),
                "views": matches}
    distances = np.asarray([item[1]["distance_m"] for item in metric])
    if len(distances) > 1 and np.ptp(distances) > 0.25 * np.median(distances):
        return {"status": "unknown", "reason": "Stereo pairs disagree on distance",
                "reference_camera": reference_id, "reference_pixels": selected.tolist(),
                "views": matches}
    return {"status": "nominal_3d_candidate_not_calibrated",
            "distance_m": float(np.median(distances)),
            **({"scale_source": "current_pose_state",
                "pose_state_method": pose_state["method"],
                "pose_state_scale_status": pose_state["scale_status"]}
               if pose_state is not None else {}),
            "reference_camera": reference_id, "reference_pixels": selected.tolist(),
            "metric_pairs": [camera_id for camera_id, _ in metric],
            "pair_count": len(metric),
            "cross_view_consistency_checked": len(metric) > 1,
            "views": matches,
            "limitations": ["Camera centers, supplied trolley position (if any), and field of view are not independently verified here.",
                            "Pose-state scale and drift must be independently validated; stale image poses are rejected.",
                            "Local matching may not identify the same physical surface point.",
                            "A single stereo pair cannot provide independent cross-view agreement.",
                            "Burned-in second-level timestamps do not establish exposure synchronization."]}


def _auto_measure_independent_groups(images: dict[str, np.ndarray], reference_id: str,
                                     selected: np.ndarray, config: dict, matcher: str,
                                     pose_state: dict | None,
                                     height_alignment: dict | None) -> dict:
    """Transfer pixels across all views, but triangulate only rigid-group pairs."""
    from .group_height import aligned_height_difference

    cameras = config["cameras"]
    if any(name not in cameras for name in images):
        raise ValueError("未知相机编号")
    selected = np.asarray(selected, dtype=float)
    ref = images[reference_id]
    if (selected.shape != (2, 2) or not np.isfinite(selected).all()
            or np.any(selected < 0) or np.any(selected[:, 0] >= ref.shape[1])
            or np.any(selected[:, 1] >= ref.shape[0])):
        raise ValueError("请在参考图像范围内指定 A、B 两个像素")
    if pose_state is not None:
        if reference_id not in pose_state.get("cameras", {}):
            raise ValueError("参考相机缺少本秒位姿状态")
        validate_pose_frame(ref, pose_state["cameras"][reference_id])

    groups = {}
    for name in images:
        groups.setdefault(cameras[name]["rig"], []).append(name)
    states = {}
    if pose_state is None:
        for group, names in groups.items():
            if len(names) >= 2:
                states[group] = state_from_nominal_group(
                    {name: images[name] for name in names}, config)
    else:
        states = {group: pose_state for group in groups}

    matches = {}
    reference_group = cameras[reference_id]["rig"]
    prepared_reference = None
    if matcher in ("sift", "auto"):
        gray, scale = _prepare(ref)
        prepared_reference = (gray, scale, _features(gray))
    for name, image in images.items():
        if name == reference_id:
            continue
        group = cameras[name]["rig"]
        try:
            if pose_state is not None and name in pose_state.get("cameras", {}):
                validate_pose_frame(image, pose_state["cameras"][name])
            if group == reference_group and group in states and name in states[group]["cameras"]:
                candidate = match_view_with_pose(
                    ref, image, selected, states[group]["cameras"][reference_id],
                    states[group]["cameras"][name], matcher, prepared_reference,
                    allow_partial=True,
                    candidate_epipolar_px=float(config.get("epipolar_candidate_threshold_px", 3.0)))
            else:
                visual_matcher = matcher if matcher in ("auto", "sift", "loftr", "xfeat") else "sift"
                candidate = match_view(ref, image, selected, cameras[reference_id],
                                       cameras[name], visual_matcher, prepared_reference,
                                       allow_partial=True)
            output = {key: value for key, value in candidate.items() if not key.startswith("_")}
            output["role"] = ("reference_group_stereo" if group == reference_group
                              else "cross_group_visual_candidate")
            matches[name] = output
        except (ValueError, cv2.error, ImportError, RuntimeError) as exc:
            matches[name] = {"status": "unknown", "role": ("reference_group_stereo"
                             if group == reference_group else "cross_group_visual_candidate"),
                             "reason": str(exc)}

    estimates = {"A": {}, "B": {}}
    checks = {}
    for group, names in groups.items():
        if len(names) != 2 or group not in states:
            continue
        first, second = names
        entries = states[group].get("cameras", {})
        if first not in entries or second not in entries:
            continue
        for index, label in enumerate(("A", "B")):
            def pixel(name):
                if name == reference_id:
                    return selected[index]
                points = matches.get(name, {}).get("pixel_points", [None, None])
                return points[index]

            p_first, p_second = pixel(first), pixel(second)
            if p_first is None or p_second is None:
                continue
            try:
                estimate = triangulate_one_with_state(
                    np.asarray(p_first, dtype=float), np.asarray(p_second, dtype=float),
                    entries[first], entries[second])
                estimates[label][group] = estimate
                checks.setdefault(label, {})[group] = {
                    "cameras": [first, second], "ray_angle_deg": estimate["ray_angle_deg"],
                    "reprojection_error_px": estimate["reprojection_error_px"]}
            except (ValueError, cv2.error) as exc:
                checks.setdefault(label, {})[group] = {"status": "unknown", "reason": str(exc)}

    common = set(estimates["A"]) & set(estimates["B"])
    base = {"reference_camera": reference_id, "reference_pixels": selected.tolist(),
            "views": matches, "point_checks": checks,
            "coordinate_policy": "independent_rigid_groups"}
    if common:
        distances = {group: float(np.linalg.norm(
            np.asarray(estimates["B"][group]["world_xyz_m"])
            - np.asarray(estimates["A"][group]["world_xyz_m"]))) for group in common}
        if len(distances) > 1 and max(distances.values()) - min(distances.values()) > 0.25 * np.median(list(distances.values())):
            return {**base, "status": "unknown", "distance_m": None,
                    "reason": "两组独立三角测量距离不一致"}
        group = reference_group if reference_group in common else sorted(common)[0]
        a = np.asarray(estimates["A"][group]["world_xyz_m"])
        b = np.asarray(estimates["B"][group]["world_xyz_m"])
        return {**base, "status": "nominal_3d_candidate_not_calibrated"
                if pose_state is None else "surveyed_group_distance_requires_validation",
                "measurement_mode": "same_group_nominal_3d" if pose_state is None else "same_group_3d",
                "distance_m": distances[group], "height_difference_m": float(b[2] - a[2]),
                "metric_group": group, "metric_pairs": [name for name in groups[group]
                                                      if name != reference_id],
                "point_support": {label: groups[group] for label in ("A", "B")},
                "group_distance_checks_m": distances,
                "scale_status": (states[group].get("scale_status") if group in states else None)}

    if (pose_state is not None and pose_state.get("method") in
            ("surveyed_common_world_poses", "surveyed_independent_group_poses")
            and pose_state.get("scale_status") != "assumed_not_calibrated"
            and height_alignment is not None):
        alternatives = [(ga, gb) for ga in estimates["A"] for gb in estimates["B"] if ga != gb]
        if alternatives:
            ga, gb = alternatives[0]
            try:
                difference, bound = aligned_height_difference(
                    {"A": estimates["A"][ga]["world_xyz_m"][2],
                     "B": estimates["B"][gb]["world_xyz_m"][2]},
                    {"A": ga, "B": gb}, height_alignment)
                return {**base, "status": "surveyed_cross_group_height_requires_validation",
                        "measurement_mode": "cross_group_height_only", "distance_m": None,
                        "height_difference_m": difference,
                        "height_difference_interval_m": [difference - bound, difference + bound],
                        "point_support": {"A": groups[ga], "B": groups[gb]}}
            except ValueError as exc:
                return {**base, "status": "unknown", "distance_m": None,
                        "reason": str(exc)}
    return {**base, "status": "unknown", "distance_m": None,
            "height_difference_m": None,
            "reason": "四路视觉候选未形成同组双目测距；跨组高度还需现场标定及竖直基准"}
