import numpy as np
import pytest

from pixel_distance.coverage import assess_coverage


def test_coverage_reports_unknown_without_cross_view_features():
    image = np.full((360, 640, 3), 120, dtype=np.uint8)
    config = {"cameras": {"D01": {"horizontal_fov_deg": 60},
                          "D02": {"horizontal_fov_deg": 60}}}
    report = assess_coverage({"D01": image, "D02": image}, "D01", config,
                             columns=4, rows=3)
    assert report["sample_count"] == 12
    assert report["views"]["D02"]["status"] == "unknown"
    assert report["views"]["D02"]["matched_points"] == 0


def test_coverage_requires_two_reference_images():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="at least one other"):
        assess_coverage({"D01": image}, "D01", {"cameras": {"D01": {}}})
