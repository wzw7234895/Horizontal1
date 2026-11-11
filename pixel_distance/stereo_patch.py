"""Match selected stereo pixels by rectified, bidirectional patch search."""

from __future__ import annotations

import cv2
import numpy as np


PATCH_SIZE = 21
ROW_TOLERANCE = 2
MIN_TEXTURE_STD = 9.0
MIN_SCORE = 0.82
MIN_UNIQUENESS = 0.06


def _search(image: np.ndarray, template: np.ndarray,
            row: int, support: np.ndarray) -> tuple[np.ndarray, float, float]:
    half = PATCH_SIZE // 2
    top = row - half - ROW_TOLERANCE
    bottom = row + half + ROW_TOLERANCE + 1
    if top < 0 or bottom > image.shape[0]:
        raise ValueError("选点的极线接近图像边界")
    scores = cv2.matchTemplate(image[top:bottom], template, cv2.TM_CCOEFF_NORMED)
    centers = support[top + half:bottom - half, half:image.shape[1] - half]
    scores[(centers < 0.99) | ~np.isfinite(scores)] = -1
    _, best, _, location = cv2.minMaxLoc(scores)
    if best < MIN_SCORE:
        raise ValueError(f"极线图像块相似度不足（{best:.2f}）")
    x, y = location
    scores[max(0, y - ROW_TOLERANCE):y + ROW_TOLERANCE + 1,
           max(0, x - PATCH_SIZE):x + PATCH_SIZE + 1] = -1
    second = float(np.max(scores))
    if best - second < MIN_UNIQUENESS:
        raise ValueError(f"极线上存在相似的多个候选（差值 {best - second:.2f}）")
    point = np.array([x + half, top + y + half], dtype=float)
    return point, float(best), second


def _patch(image: np.ndarray, valid: np.ndarray, point: np.ndarray) -> np.ndarray:
    x, y = np.rint(point).astype(int)
    half = PATCH_SIZE // 2
    if (x < half or y < half or x + half >= image.shape[1]
            or y + half >= image.shape[0]):
        raise ValueError("选点的图像块超出有效画面")
    patch = image[y - half:y + half + 1, x - half:x + half + 1]
    if not np.all(valid[y - half:y + half + 1, x - half:x + half + 1]):
        raise ValueError("选点图像块包含校正后的无效边界")
    if float(np.std(patch)) < MIN_TEXTURE_STD:
        raise ValueError("选点局部纹理不足")
    return patch


def match_rectified_patches(reference: np.ndarray, target: np.ndarray,
                            points: np.ndarray, reference_entry: dict,
                            target_entry: dict, allow_partial: bool = False) -> dict:
    """Return conservative pixel correspondences for a calibrated stereo pair."""
    if reference.shape[:2] != target.shape[:2]:
        raise ValueError("图像块匹配要求两路图像尺寸相同")
    height, width = reference.shape[:2]
    k1 = np.asarray(reference_entry["K"], dtype=np.float64)
    k2 = np.asarray(target_entry["K"], dtype=np.float64)
    d1 = np.asarray(reference_entry["D"], dtype=np.float64)
    d2 = np.asarray(target_entry["D"], dtype=np.float64)
    r1 = np.asarray(reference_entry["R"], dtype=np.float64)
    r2 = np.asarray(target_entry["R"], dtype=np.float64)
    t1 = np.asarray(reference_entry["t_m"], dtype=np.float64)
    t2 = np.asarray(target_entry["t_m"], dtype=np.float64)
    rotation = r2 @ r1.T
    translation = t2 - rotation @ t1
    if np.linalg.norm(translation) < 1e-4:
        raise ValueError("两相机基线过短")
    rect1, rect2, p1, p2, _, _, _ = cv2.stereoRectify(
        k1, d1, k2, d2, (width, height), rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    gray1 = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference
    gray2 = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target
    frames, masks = [], []
    for gray, k, d, rect, projection in ((gray1, k1, d1, rect1, p1),
                                          (gray2, k2, d2, rect2, p2)):
        map_x, map_y = cv2.initUndistortRectifyMap(
            k, d, rect, projection, (width, height), cv2.CV_32FC1)
        frames.append(cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR))
        valid = cv2.remap(np.full(gray.shape, 255, dtype=np.uint8),
                          map_x, map_y, cv2.INTER_NEAREST) > 0
        masks.append(valid)
    support = [cv2.boxFilter(mask.astype(np.float32), cv2.CV_32F,
                             (PATCH_SIZE, PATCH_SIZE), normalize=True)
               for mask in masks]
    matches, checks = [], []
    for point in np.asarray(points, dtype=np.float64):
        try:
            rectified = cv2.undistortPoints(point.reshape(1, 1, 2), k1, d1,
                                            R=rect1, P=p1[:, :3]).reshape(2)
            template = _patch(frames[0], masks[0], rectified)
            candidate, score, second = _search(frames[1], template,
                                                round(rectified[1]), support[1])
            reverse_template = _patch(frames[1], masks[1], candidate)
            returned, reverse_score, _ = _search(
                frames[0], reverse_template, round(candidate[1]), support[0])
            round_trip = float(np.linalg.norm(returned - np.rint(rectified)))
            if round_trip > 2.5:
                raise ValueError(f"双向匹配不一致（{round_trip:.1f} 像素）")
            ray = np.linalg.solve(p2[:, :3], [candidate[0], candidate[1], 1.0])
            original_ray = rect2.T @ ray
            if original_ray[2] <= 0:
                raise ValueError("匹配视线指向相机后方")
            pixel = cv2.projectPoints(original_ray.reshape(1, 3),
                                      np.zeros(3), np.zeros(3), k2, d2)[0].reshape(2)
            if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
                raise ValueError("匹配点超出目标原图")
            matches.append(pixel.tolist())
            checks.append({"status": "matched_candidate", "method": "rectified_patch",
                           "score": score, "second_score": second,
                           "reverse_score": reverse_score,
                           "round_trip_px": round_trip})
        except (ValueError, cv2.error) as exc:
            if not allow_partial:
                raise
            matches.append(None)
            checks.append({"status": "unknown", "reason": str(exc)})
    if not any(point is not None for point in matches):
        if not allow_partial:
            raise ValueError("两个选点都缺少可信的极线图像块匹配")
        return {"status": "unknown", "matcher": "patch", "pixel_points": matches,
                "point_checks": checks}
    return {"status": "matched_candidate" if all(point is not None for point in matches)
            else "partial_coverage", "matcher": "patch", "pixel_points": matches,
            "point_checks": checks}
