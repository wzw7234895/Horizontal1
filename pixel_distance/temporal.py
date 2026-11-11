"""Compare selected point motion with the dominant static scene motion."""

from __future__ import annotations

import cv2
import numpy as np

MAX_SIDE = 960
MAX_BACKGROUND_FEATURES = 800


def motion_evidence(reference: np.ndarray, later: np.ndarray,
                    points: dict[str, tuple[float, float]]) -> dict:
    if reference.shape[:2] != later.shape[:2]:
        raise ValueError("Temporal frames must have the same dimensions")
    if len(points) != 2:
        raise ValueError("Temporal evidence requires exactly the two selected points")
    original_height, original_width = reference.shape[:2]
    scale = min(1.0, MAX_SIDE / max(original_height, original_width))
    if scale < 1.0:
        size = (round(original_width * scale), round(original_height * scale))
        reference = cv2.resize(reference, size, interpolation=cv2.INTER_AREA)
        later = cv2.resize(later, size, interpolation=cv2.INTER_AREA)
    sx, sy = reference.shape[1] / original_width, reference.shape[0] / original_height
    first = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference
    second = cv2.cvtColor(later, cv2.COLOR_BGR2GRAY) if later.ndim == 3 else later
    height, width = first.shape
    mask = np.where(first > 35, 255, 0).astype(np.uint8)
    mask[:int(height * 0.12)] = 0
    mask[int(height * 0.90):] = 0
    corners = cv2.goodFeaturesToTrack(first, maxCorners=MAX_BACKGROUND_FEATURES, qualityLevel=0.008,
                                      minDistance=8, mask=mask)
    if corners is None or len(corners) < 50:
        return {"status": "unknown", "reason": "背景特征不足", "points": {}}
    tracked, forward, _ = cv2.calcOpticalFlowPyrLK(first, second, corners, None,
                                                    winSize=(21, 21), maxLevel=3)
    if tracked is None or forward is None:
        return {"status": "unknown", "reason": "背景跟踪失败", "points": {}}
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(second, first, tracked, None,
                                                winSize=(21, 21), maxLevel=3)
    if back is None or reverse is None:
        return {"status": "unknown", "reason": "背景反向跟踪失败", "points": {}}
    good = ((forward.ravel() > 0) & (reverse.ravel() > 0)
            & (np.linalg.norm(back.reshape(-1, 2) - corners.reshape(-1, 2), axis=1) < 1.5))
    source = corners.reshape(-1, 2)[good]
    target = tracked.reshape(-1, 2)[good]
    if len(source) < 50:
        return {"status": "unknown", "reason": "背景跟踪不足", "points": {}}
    homography, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    if homography is None or inliers is None or int(inliers.sum()) < 40:
        return {"status": "unknown", "reason": "静态背景运动模型不稳定", "points": {}}
    inlier_source = source[inliers.ravel().astype(bool)]
    inlier_target = target[inliers.ravel().astype(bool)]
    predicted_background = cv2.perspectiveTransform(
        inlier_source.reshape(-1, 1, 2), homography).reshape(-1, 2)
    background_error = np.linalg.norm(predicted_background - inlier_target, axis=1)
    tolerance = max(3.0, 3.0 * float(np.median(background_error)))
    texture_map = cv2.cornerMinEigenVal(first, blockSize=7)
    support_hull = cv2.convexHull(inlier_source.astype(np.float32))
    point_result = {}
    for label, xy in points.items():
        original_point = np.asarray(xy, dtype=np.float32)
        if (original_point.shape != (2,) or not np.isfinite(original_point).all()
                or not (0 <= original_point[0] < original_width and 0 <= original_point[1] < original_height)):
            point_result[label] = {"status": "unknown", "reason": "点位不在画面内"}
            continue
        point = (original_point * np.asarray([sx, sy], dtype=np.float32)).astype(np.float32)
        expected = cv2.perspectiveTransform(point.reshape(1, 1, 2), homography).reshape(2)
        if cv2.pointPolygonTest(support_hull, tuple(map(float, point)), False) < 0:
            point_result[label] = {"status": "unknown", "reason": "点位超出背景跟踪覆盖区域"}
            continue
        if float(np.min(np.linalg.norm(inlier_source - point, axis=1))) > 75:
            point_result[label] = {"status": "unknown", "reason": "点位附近缺少背景特征"}
            continue
        texture = float(texture_map[min(height - 1, round(point[1])), min(width - 1, round(point[0]))])
        if texture < 1e-4:
            point_result[label] = {"status": "unknown", "reason": "点位纹理不足",
                                   "background_prediction_pixel": (expected / [sx, sy]).tolist()}
            continue
        measured, ok, _ = cv2.calcOpticalFlowPyrLK(first, second, point.reshape(1, 1, 2), None,
                                                   winSize=(21, 21), maxLevel=3)
        if measured is None or ok is None or not bool(ok[0, 0]):
            point_result[label] = {"status": "unknown", "reason": "点位跟踪失败"}
            continue
        returned, backward_ok, _ = cv2.calcOpticalFlowPyrLK(
            second, first, measured, None, winSize=(21, 21), maxLevel=3)
        if (returned is None or backward_ok is None or not bool(backward_ok[0, 0])
                or np.linalg.norm(returned.reshape(2) - point) > 1.5):
            point_result[label] = {"status": "unknown", "reason": "点位双向跟踪不一致"}
            continue
        actual = measured.reshape(2)
        residual = float(np.linalg.norm(actual - expected))
        point_result[label] = {
            "status": "background_consistent" if residual <= tolerance else "background_deviation",
            "residual_px": residual / sx, "threshold_px": tolerance / sx,
            "tracked_pixel": (actual / [sx, sy]).tolist(),
            "background_prediction_pixel": (expected / [sx, sy]).tolist(),
        }
    return {"status": "analyzed", "background_inliers": int(inliers.sum()),
            "background_median_error_px": float(np.median(background_error)),
            "analysis_image_size": [width, height],
            "background_feature_cap": MAX_BACKGROUND_FEATURES,
            "points": point_result,
            "note": "背景一致不证明点在地面；偏离可能由视差、独立运动或错误跟踪造成。"}
