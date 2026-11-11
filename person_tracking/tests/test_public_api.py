import json
from threading import Event
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


@pytest.fixture
def short_video(tmp_path):
    path = tmp_path / "api_clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (160, 100))
    assert writer.isOpened()
    for index in range(50):
        writer.write(np.full((100, 160, 3), index * 4, np.uint8))
    writer.release()
    return path


def test_public_api_streams_versioned_frame_records(short_video, tmp_path, monkeypatch):
    from ultralytics.engine.results import Boxes

    from person_tracking.api import run_person_tracking
    from person_tracking.common import PACKAGE_DIR

    config = tmp_path / "cameras.json"
    config.write_text(json.dumps({
        "project_root": str(PACKAGE_DIR.parent),
        "model": "person_tracking/weights/yolo11n.pt",
        "image_size": 640,
        "device": "cpu",
        "detection_conf": 0.1,
        "nms_iou": 0.6,
        "analysis_fps": 10,
        "lost_buffer_seconds": 2.5,
        "stability": {"confirm_hits": 3, "prediction_box_seconds": 2.0},
        "tracker": str(PACKAGE_DIR / "configs" / "botsort.yaml"),
        "cameras": [{"id": "D01", "name": "test", "mount": "test",
                     "source": str(short_video), "recording_start": "2026-01-01T00:00:00",
                     "time_offset_s": 0, "roi": []}],
    }), encoding="utf-8")

    class FakeModel:
        def predict(self, frames, **kwargs):
            return [SimpleNamespace(boxes=Boxes(
                np.array([[30, 10, 70, 90, 0.9, 0]], dtype=np.float32), frame.shape[:2]))
                    for frame in frames]

    import person_tracking.tracking as tracking
    monkeypatch.setattr(tracking, "load_detector", lambda *args, **kwargs: (FakeModel(), {}))

    records = []
    def enrich(frame, record):
        record["enriched_for_test"] = list(frame.shape)
        return record

    result = run_person_tracking(config=config, camera_ids="D01", output_dir=tmp_path / "out",
                                 analysis_fps=10, save_video=False, on_frame=records.append,
                                 frame_enricher=enrich,
                                 roi_overrides={"D01": [[0, 0], [1, 0], [1, 1], [0, 1]]})

    assert result["schema_version"] == "1.0"
    assert result["status"] == "completed"
    assert result["cameras"][0]["analysed_frames"] == len(records) == 20
    assert all(record["schema_version"] == "1.0" for record in records)
    assert all(record["camera_id"] == "D01" for record in records)
    assert records[-1]["detections"][0]["observed"] is True
    assert records[-1]["detections"][0]["track_id"] == 1
    assert records[-1]["detections"][0]["inside_roi"] is True
    assert records[-1]["enriched_for_test"] == [100, 160, 3]
    run_config = json.loads((tmp_path / "out" / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["roi_source"] == "api_override"
    assert run_config["cameras"][0]["roi"] == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    saved = [json.loads(line) for line in
             (tmp_path / "out" / "D01" / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    assert saved == records

    stop_event = Event()
    stopped_records = []

    def stop_after_four(record):
        stopped_records.append(record)
        if len(stopped_records) == 4:
            stop_event.set()

    stopped_result = run_person_tracking(
        config=config, camera_ids="D01", output_dir=tmp_path / "stopped_out",
        analysis_fps=10, save_video=False, on_frame=stop_after_four, stop_event=stop_event,
    )

    assert stopped_result["status"] == "stopped"
    assert len(stopped_records) == 4
    saved_summary = json.loads((tmp_path / "stopped_out" / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["status"] == "stopped"
