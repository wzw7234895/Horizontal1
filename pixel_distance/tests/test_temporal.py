import numpy as np

from pixel_distance.temporal import motion_evidence


def test_background_consistency_does_not_claim_ground():
    rng = np.random.default_rng(3)
    image = rng.integers(50, 220, size=(480, 640), dtype=np.uint8)
    result = motion_evidence(image, image.copy(), {"A": (320, 240), "B": (420, 290)})
    assert result["status"] == "analyzed"
    assert result["points"]["A"]["status"] == "background_consistent"
    assert result["points"]["B"]["status"] == "background_consistent"
    assert "不证明点在地面" in result["note"]


def test_temporal_requires_same_image_dimensions():
    first = np.zeros((100, 120), dtype=np.uint8)
    second = np.zeros((110, 120), dtype=np.uint8)
    try:
        motion_evidence(first, second, {"A": (50, 50)})
    except ValueError as exc:
        assert "dimensions" in str(exc)
    else:
        raise AssertionError("Expected mismatched frame sizes to be rejected")


def test_temporal_bounds_full_hd_work_and_preserves_pixel_coordinates():
    rng = np.random.default_rng(4)
    image = rng.integers(50, 220, size=(1080, 1920), dtype=np.uint8)
    result = motion_evidence(image, image.copy(), {"A": (500, 400), "B": (1000, 600)})
    assert result["analysis_image_size"] == [960, 540]
    assert result["background_inliers"] <= 800
    assert result["points"]["A"]["status"] == "background_consistent"
    np.testing.assert_allclose(result["points"]["A"]["tracked_pixel"], [500, 400], atol=1)
