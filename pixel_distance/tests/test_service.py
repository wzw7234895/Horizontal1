import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from pixel_distance.geometry import load_json
from pixel_distance.service import DistanceService, create_app


def test_service_requires_four_valid_frames_and_in_bounds_points():
    service = DistanceService(load_json("pixel_distance/camera_presets.json"))
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    images = {camera: image for camera in ("D01", "D02", "D03", "D04")}
    with pytest.raises(ValueError, match="必须提供四路图像"):
        service.measure_images({"D04": image}, "D04", [1, 2], [3, 4])
    with pytest.raises(ValueError, match="超出参考图像范围"):
        service.measure_images(images, "D04", [160, 2], [3, 4])
    with pytest.raises(ValueError, match="uint8 BGR"):
        service.measure_images({**images, "D01": np.zeros((10, 10), dtype=np.uint8)},
                               "D04", [1, 2], [3, 4])


def test_service_returns_stable_result_and_distinguishes_candidate_from_verified(monkeypatch):
    monkeypatch.setattr("pixel_distance.service.measure_video_views",
                        lambda *_args, **_kwargs: {
                            "status": "nominal_3d_candidate_not_calibrated",
                            "measurement_mode": "same_group_nominal_3d",
                            "distance_m": 2.5, "height_difference_m": 0.3,
                            "metric_group": "trolley", "views": {"D02": {"status": "matched_candidate"}}})
    service = DistanceService(load_json("pixel_distance/camera_presets.json"))
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    result = service.measure_images({camera: image for camera in
                                     ("D01", "D02", "D03", "D04")},
                                    "D04", [10, 20], [100, 90],
                                    timestamp="2026-11-10T12:34:56")
    assert result["schema_version"] == "1.0"
    assert result["status"] == "nominal_3d_candidate_not_calibrated"
    assert result["distance_m"] == 2.5
    assert result["distance_available"] is True
    assert result["distance_determined"] is False
    assert result["height_difference_definition"] == "Z_B - Z_A"
    assert len(result["request_id"]) == 36


def test_config_update_rejects_cross_group_fixed_rig_assumption():
    config = load_json("pixel_distance/camera_presets.json")
    service = DistanceService(config)
    revision = service.update_config(config)
    assert revision == service.update_config(config)
    invalid = {**config, "all_cameras_rigid": True}
    with pytest.raises(ValueError, match="all_cameras_rigid=false"):
        service.update_config(invalid)
    assert service.get_config()["all_cameras_rigid"] is False


def test_http_endpoints_accept_four_images_and_validate_coordinates(monkeypatch):
    monkeypatch.setattr("pixel_distance.service.measure_video_views",
                        lambda *_args, **_kwargs: {
                            "status": "unknown", "measurement_mode": "same_group_unavailable",
                            "distance_m": None, "height_difference_m": None,
                            "reason": "no reliable correspondence", "views": {}})
    app = create_app(DistanceService(load_json("pixel_distance/camera_presets.json")))
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    cameras = client.get("/v1/cameras").json()
    assert set(cameras["cameras"]) == {"D01", "D02", "D03", "D04"}
    config = load_json("pixel_distance/camera_presets.json")
    updated = client.put("/v1/config", json=config)
    assert updated.status_code == 200
    invalid_config = {**config, "all_cameras_rigid": True}
    assert client.put("/v1/config", json=invalid_config).status_code == 422
    ok, encoded = cv2.imencode(".jpg", np.zeros((40, 60, 3), dtype=np.uint8))
    assert ok
    files = {camera: (f"{camera}.jpg", encoded.tobytes(), "image/jpeg")
             for camera in ("D01", "D02", "D03", "D04")}
    data = {"reference_camera": "D04", "point_a_x": "10", "point_a_y": "10",
            "point_b_x": "30", "point_b_y": "30", "timestamp": "2026-11-10T12:34:56"}
    result = client.post("/v1/measure/images", data=data, files=files)
    assert result.status_code == 200
    assert result.json()["distance_m"] is None
    assert result.json()["distance_determined"] is False
    bad = client.post("/v1/measure/images", data={**data, "point_b_x": "60"}, files=files)
    assert bad.status_code == 422
