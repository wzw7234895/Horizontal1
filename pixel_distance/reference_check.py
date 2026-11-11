"""Offline comparison of a measured segment with an independent length reference."""

from __future__ import annotations

import math


def check_reference_length(measurement: dict, name: str,
                           length_range_m: tuple[float, float]) -> dict:
    lower, upper = map(float, length_range_m)
    if not name.strip():
        raise ValueError("需提供参照物名称")
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower <= upper):
        raise ValueError("参照物长度范围必须为正数且按从小到大填写")
    if measurement.get("status") in ("unknown", "diagnostic_only"):
        raise ValueError("需要独立产生的有效候选测距结果")
    if (measurement.get("scale_source") == "vehicle_length_prior"
            or measurement.get("pose_state_method") == "vehicle_length_diagnostic_only"):
        raise ValueError("该距离已由同一长度先验推导，不能再用它核对该先验")
    distance = float(measurement["distance_m"])
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("候选距离必须为有限正数")
    return {"status": "diagnostic_only", "reference_name": name.strip(),
            "reference_length_range_m": [lower, upper],
            "candidate_distance_m": distance,
            "candidate_status": measurement.get("status"),
            "comparison": "within_range" if lower <= distance <= upper else "outside_range",
            "distance_difference_range_m": [distance - upper, distance - lower],
            "conditional_scale_factor_range": [lower / distance, upper / distance],
            "limitations": ["This comparison does not alter runtime camera calibration or pose.",
                            "Scale ratios are diagnostic hints, not calibrated correction factors.",
                            "Verify point identity and measurement independence before interpreting the result."]}
