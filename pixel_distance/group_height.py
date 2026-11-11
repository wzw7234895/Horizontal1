"""Triangulate one point per rigid camera group and compare vertical heights."""

from __future__ import annotations

import math

import numpy as np

from .pose_state import triangulate_one_with_state, validate_pose_frame


def aligned_height_difference(z_by_point: dict, groups: dict, alignment: dict) -> tuple[float, float]:
    if alignment is None or alignment.get("status") != "validated":
        raise ValueError("跨组比较高度须提供已验证的离线竖直高度基准")
    offsets = alignment.get("z_offsets_m", {})
    errors = alignment.get("point_height_error_bound_m", {})
    try:
        z = {label: float(z_by_point[label]) + float(offsets[groups[label]])
             for label in ("A", "B")}
        point_errors = [float(errors[label]) for label in ("A", "B")]
        alignment_error = float(alignment["alignment_error_bound_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("高度基准须包含两组 Z 偏移和已验证的误差上界") from exc
    if (not all(math.isfinite(v) for v in (*z.values(), *point_errors, alignment_error))
            or min(*point_errors, alignment_error) < 0):
        raise ValueError("高度偏移或误差上界无效")
    return z["B"] - z["A"], sum(point_errors) + alignment_error


def compare_group_points(observations: dict, images: dict, state: dict,
                         config: dict, alignment: dict | None = None) -> dict:
    """A and B each need two same-group observations; only aligned Z crosses groups."""
    if state.get("method") not in ("surveyed_common_world_poses", "surveyed_independent_group_poses") or state.get("scale_status") == "assumed_not_calibrated":
        raise ValueError("跨组高度比较需要现场标定的相机位姿状态")
    points = {}
    groups = {}
    diagnostics = {}
    for label in ("A", "B"):
        views = observations.get(label)
        if not isinstance(views, list) or len(views) != 2:
            raise ValueError(f"{label} 必须在同一安装组的两台相机中各有一个同名像素")
        names = [view["camera_id"] for view in views]
        if len(set(names)) != 2 or any(name not in config["cameras"] for name in names):
            raise ValueError(f"{label} 的相机编号重复或未知")
        rigs = {config["cameras"][name]["rig"] for name in names}
        if len(rigs) != 1:
            raise ValueError(f"{label} 的两路观测必须来自同一安装组")
        group = rigs.pop()
        pixels = []
        entries = []
        for view in views:
            name = view["camera_id"]
            if name not in images or name not in state.get("cameras", {}):
                raise ValueError(f"{name} 缺少本秒图像或对应位姿")
            image = images[name]
            entry = state["cameras"][name]
            validate_pose_frame(image, entry)
            pixel = np.asarray(view["pixel"], dtype=float)
            if (pixel.shape != (2,) or not np.isfinite(pixel).all()
                    or np.any(pixel < 0) or pixel[0] >= image.shape[1]
                    or pixel[1] >= image.shape[0]):
                raise ValueError(f"{name} 的 {label} 像素无效")
            pixels.append(pixel)
            entries.append(entry)
        estimate = triangulate_one_with_state(pixels[0], pixels[1], entries[0], entries[1])
        groups[label] = group
        points[label] = np.asarray(estimate["world_xyz_m"], dtype=float)
        diagnostics[label] = {"group": group, "cameras": names,
                              "ray_angle_deg": estimate["ray_angle_deg"],
                              "reprojection_error_px": estimate["reprojection_error_px"]}
    if groups["A"] == groups["B"]:
        delta = points["B"] - points["A"]
        return {"status": "surveyed_group_distance_requires_validation",
                "measurement_mode": "same_group_3d", "distance_m": float(np.linalg.norm(delta)),
                "height_difference_m": float(delta[2]),
                "point_xyz_m": {k: v.tolist() for k, v in points.items()},
                "point_checks": diagnostics}
    difference, bound = aligned_height_difference(
        {label: points[label][2] for label in ("A", "B")}, groups, alignment)
    z = {label: float(points[label][2]) + float(alignment["z_offsets_m"][groups[label]])
         for label in ("A", "B")}
    return {"status": "surveyed_cross_group_height_requires_validation",
            "measurement_mode": "cross_group_height_only", "distance_m": None,
            "height_difference_m": difference,
            "height_difference_interval_m": [difference - bound, difference + bound],
            "aligned_height_m": z, "point_checks": diagnostics,
            "note": "小车只水平平移；不输出跨组水平位置或三维距离。误差上界由现场校验输入。"}
