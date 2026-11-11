"""Sample the field of view to diagnose arbitrary two-point matchability."""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from .auto3d import _features, _prepare, match_view


def assess_coverage(images: dict[str, np.ndarray], reference_id: str,
                    config: dict, matcher: str = "sift",
                    columns: int = 9, rows: int = 6) -> dict:
    if reference_id not in images or len(images) < 2:
        raise ValueError("Reference and at least one other image are required")
    if columns < 2 or rows < 2:
        raise ValueError("Coverage grid must have at least two rows and columns")
    cameras = config["cameras"]
    if any(camera_id not in cameras for camera_id in images):
        raise ValueError("Unknown camera ID")
    reference = images[reference_id]
    height, width = reference.shape[:2]
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference
    points = []
    for y in np.linspace(0.15 * height, 0.85 * height, rows):
        for x in np.linspace(0.08 * width, 0.92 * width, columns):
            if gray[round(y), round(x)] > 35:
                points.append([float(x), float(y)])
    if len(points) < 2:
        raise ValueError("Reference image contains too few visible sample points")
    prepared = None
    if matcher in ("sift", "auto"):
        ref_gray, ref_scale = _prepare(reference)
        prepared = (ref_gray, ref_scale, _features(ref_gray))
    views = {}
    for camera_id, target in images.items():
        if camera_id == reference_id:
            continue
        try:
            candidate = match_view(reference, target, np.asarray(points),
                                   cameras[reference_id], cameras[camera_id],
                                   matcher, prepared, allow_partial=True)
            checks = candidate["point_checks"]
            matched = candidate["matched_point_count"]
            views[camera_id] = {
                "status": "analyzed", "matcher": candidate["matcher"],
                "epipolar_inliers": candidate["epipolar_inliers"],
                "matched_points": matched,
                "pair_match_upper_bound": matched * (matched - 1) / (len(points) * (len(points) - 1)),
                "failure_reasons": dict(Counter(check.get("reason", "") for check in checks
                                                if check["status"] == "unknown")),
                "point_checks": checks,
                "matched_pixels": candidate["pixel_points"],
            }
        except (ValueError, cv2.error) as exc:
            views[camera_id] = {"status": "unknown", "reason": str(exc),
                                "matched_points": 0, "pair_match_upper_bound": 0.0}
    return {"reference_camera": reference_id, "matcher": matcher,
            "sample_points": points, "sample_count": len(points), "views": views,
            "interpretation": "Grid coverage is a sample, not a success rate for all possible points. "
                              "Pair bound only counts two individual correspondences and does not prove triangulation."}
