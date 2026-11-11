import pytest

from pixel_distance.reference_check import check_reference_length


def test_reference_check_accepts_any_independent_object():
    result = check_reference_length(
        {"status": "nominal_3d_candidate_not_calibrated", "distance_m": 3.0},
        "悬挂工件试件", (3.5, 4.0))
    assert result["status"] == "diagnostic_only"
    assert result["comparison"] == "outside_range"
    assert result["conditional_scale_factor_range"] == pytest.approx([3.5 / 3, 4 / 3])


def test_reference_check_rejects_circular_length_calibration():
    with pytest.raises(ValueError, match="同一长度先验"):
        check_reference_length({"status": "candidate", "distance_m": 4.5,
                                "pose_state_method": "vehicle_length_diagnostic_only"},
                               "运输车", (4, 5))
    with pytest.raises(ValueError, match="独立产生"):
        check_reference_length({"status": "unknown"}, "试件", (1, 2))
