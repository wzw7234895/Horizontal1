import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from person_tracking.common import load_config, start_seconds
from person_tracking.video import FileReader, read_frame_at


@pytest.fixture
def short_video(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (160, 100))
    assert writer.isOpened()
    for i in range(50):
        writer.write(np.full((100, 160, 3), i * 4, np.uint8))
    writer.release()
    return path


@pytest.mark.parametrize("decode_mode", ["seek", "sequential"])
def test_frame_sampling_uses_source_times(short_video, decode_mode):
    reader = FileReader(str(short_video), analysis_fps=10, start=0.4, duration=1, decode_mode=decode_mode)
    items = []
    try:
        while (item := reader.next()) is not None:
            items.append(item)
    finally:
        reader.close()
    assert len(items) == 10
    indexes = [i[1] for i in items]
    assert indexes[0] == 10 and indexes[-1] < 35
    assert len(set(indexes)) == 10
    assert all(t == pytest.approx(frame / 25) for _, frame, t, _ in items)
    assert all(np.mean(image) == pytest.approx(index * 4, abs=3) for image, index, _, _ in items)


def test_separate_camera_clock_offsets():
    cfg = load_config()
    starts = [start_seconds(c, 0, "15:57:20") for c in cfg["cameras"]]
    assert starts == [2740, 2147, 1511, 3694]
    assert {c["id"] for c in cfg["cameras"] if c["mount"] == "小车"} == {"D02", "D04"}


def test_tracker_survives_short_occlusion_without_emitting_predictions():
    from ultralytics.engine.results import Boxes
    from person_tracking.tracking import make_tracker
    cfg = load_config()
    tracker = make_tracker(cfg, 10)
    frame = np.zeros((120, 160, 3), np.uint8)
    detection = Boxes(np.array([[30, 20, 55, 95, 0.9, 0]], dtype=np.float32), frame.shape[:2])
    empty = Boxes(np.empty((0, 6), dtype=np.float32), frame.shape[:2])
    track = tracker.update(detection, frame)
    first_id = int(track[0, 4])
    tracker.update(detection, frame)
    for _ in range(3):
        assert len(tracker.update(empty, frame)) == 0
    returned = tracker.update(detection, frame)
    assert int(returned[0, 4]) == first_id
    assert tracker.max_frames_lost == 25


def test_camera_trackers_have_independent_state():
    from ultralytics.engine.results import Boxes
    from person_tracking.tracking import make_tracker
    cfg = load_config()
    tracker_a = make_tracker(cfg, 10)
    frame = np.zeros((120, 160, 3), np.uint8)
    detection = Boxes(np.array([[30, 20, 55, 95, 0.9, 0]], dtype=np.float32), frame.shape[:2])
    first = tracker_a.update(detection, frame)
    tracker_b = make_tracker(cfg, 10)
    second = tracker_b.update(detection, frame)
    assert int(first[0, 4]) == int(second[0, 4]) == 1
    assert tracker_a.frame_id == tracker_b.frame_id == 1
    again = tracker_a.update(detection, frame)
    assert int(again[0, 4]) == 1
    assert tracker_a.frame_id == 2 and tracker_b.frame_id == 1


def test_loss_timeout_uses_elapsed_time_when_analysis_is_slow(short_video, tmp_path):
    from types import SimpleNamespace
    from ultralytics.engine.results import Boxes
    from person_tracking.tracking import CameraState
    cfg = load_config()
    camera = {**cfg["cameras"][0], "source": str(short_video),
              "roi": [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]]}
    args = SimpleNamespace(start=0, start_clock=None, duration=None, fps=10, out=tmp_path / "tracks",
                           no_video=True, decode_mode="seek")
    state = CameraState(camera, cfg, args, "test-run")
    frame = np.zeros((100, 160, 3), np.uint8)
    detected = SimpleNamespace(boxes=Boxes(np.array([[30, 20, 55, 95, 0.9, 0]], np.float32), frame.shape[:2]))
    empty = SimpleNamespace(boxes=Boxes(np.empty((0, 6), np.float32), frame.shape[:2]))
    try:
        for t, result in [(0, detected), (0.1, detected), (0.2, detected),
                          (0.8, empty), (1.6, empty), (2.0, empty),
                          (2.4, detected), (2.5, detected), (2.6, detected), (2.7, detected)]:
            state.update((frame.copy(), round(t*25), t, 0), result)
    finally:
        state.close("test_completed")
    rows = [json.loads(line) for line in (args.out / "D01/frames.jsonl").read_text(encoding="utf8").splitlines()]
    assert rows[0]["detections"][0]["track_id"] is None
    assert rows[1]["detections"][0]["track_id"] is None
    assert rows[2]["detections"][0]["track_id"] == 1
    assert rows[2]["detections"][0]["inside_roi"] is False
    assert rows[2]["detections"][0]["displayed"] is True
    assert rows[3]["detections"][0]["predicted"] is True
    assert rows[5]["detections"][0]["prediction_age_s"] == pytest.approx(1.8)
    assert rows[6]["detections"][0]["observed"] is True
    assert rows[-1]["detections"][0]["track_id"] == 1
    assert rows[0]["session_id"] == rows[-1]["session_id"]


def test_nested_person_detections_are_removed_before_tracking(short_video, tmp_path):
    from types import SimpleNamespace
    from ultralytics.engine.results import Boxes
    from person_tracking.tracking import CameraState

    cfg = load_config()
    camera = {**cfg["cameras"][0], "source": str(short_video)}
    args = SimpleNamespace(start=0, start_clock=None, duration=None, fps=10, out=tmp_path / "nested_tracks",
                           no_video=True, decode_mode="seek")
    state = CameraState(camera, cfg, args, "nested-test")
    frame = np.zeros((100, 160, 3), np.uint8)
    nested = SimpleNamespace(boxes=Boxes(np.array([
        [20, 10, 100, 90, 0.70, 0],
        [40, 20, 70, 75, 0.95, 0],
    ], dtype=np.float32), frame.shape[:2]))
    try:
        for frame_index in range(3):
            state.update((frame.copy(), frame_index, frame_index / 10, 0), nested)
    finally:
        summary = state.close("test_completed")

    rows = [json.loads(line) for line in (args.out / "D01/frames.jsonl").read_text(encoding="utf8").splitlines()]
    assert all(len(row["detections"]) == 1 for row in rows)
    assert all(row["raw_detection_count"] == 2 for row in rows)
    assert all(row["detections_suppressed_nested"] == 1 for row in rows)
    assert all(row["detections"][0]["xyxy"] == [20.0, 10.0, 100.0, 90.0] for row in rows)
    assert summary["raw_detections"] == 6
    assert summary["detections"] == 3
    assert summary["detections_suppressed_nested"] == 3
    assert summary["tracks_created"] == 1


def test_display_hysteresis_keeps_observed_boxes_without_drawing_missing_frames(short_video, tmp_path):
    from types import SimpleNamespace
    from ultralytics.engine.results import Boxes
    from person_tracking.tracking import CameraState

    cfg = load_config()
    cfg["stability"] = {"confirm_hits": 3, "display_min_confidence": 0.2,
                        "display_keep_confidence": 0.15}
    camera = {**cfg["cameras"][0], "source": str(short_video)}
    args = SimpleNamespace(start=0, start_clock=None, duration=None, fps=10,
                           out=tmp_path / "display_hysteresis", no_video=True, decode_mode="seek")
    state = CameraState(camera, cfg, args, "display-test")
    frame = np.zeros((100, 160, 3), np.uint8)
    scores = [0.9, 0.9, 0.21, 0.19, 0.16, 0.14, 0.18, 0.21, None, 0.9]
    rendered = []
    state.args.no_video = False
    state.render = lambda image, observations, polygon, timestamp: rendered.append(
        [obs for obs in observations if obs["displayed"]])
    try:
        for index, score in enumerate(scores):
            values = np.empty((0, 6), np.float32) if score is None else np.array(
                [[30, 20, 55, 95, score, 0]], np.float32)
            state.update((frame.copy(), index, index / 10, 0),
                         SimpleNamespace(boxes=Boxes(values, frame.shape[:2])))
    finally:
        state.close("test_completed")
    rows = [json.loads(line) for line in (args.out / "D01/frames.jsonl").read_text(encoding="utf8").splitlines()]
    assert [bool(items) for items in rendered] == [False, False, True, True, True, False, False, True, False, True]
    assert rows[8]["detections"] == []
    assert all(obs["xyxy"] == [30.0, 20.0, 55.0, 95.0] for row in rows for obs in row["detections"])
    assert {obs["track_id"] for row in rows[2:] for obs in row["detections"]} == {1}


def test_sequential_sample_frames_match_reference_and_reject_backwards(short_video):
    cap = cv2.VideoCapture(str(short_video))
    try:
        for index in (10, 23, 37):
            frame = read_frame_at(cap, index, "sequential")
            assert np.mean(frame) == pytest.approx(index * 4, abs=3)
            assert int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == index + 1
        with pytest.raises(ValueError, match="increasing"):
            read_frame_at(cap, 23, "sequential")
        with pytest.raises(OSError, match="stopped"):
            read_frame_at(cap, 70, "sequential")
    finally:
        cap.release()


class TimestampCapture:
    """A clip whose presentation times differ from its nominal frame rate."""

    def __init__(self, timestamps):
        self.timestamps = timestamps
        self.index = 0
        self.set_calls = []

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 25
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.timestamps)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.index
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.timestamps[max(0, self.index - 1)] * 1000
        return 0

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        self.index = next((i for i, t in enumerate(self.timestamps) if t >= value / 1000), len(self.timestamps))
        return True

    def grab(self):
        if self.index >= len(self.timestamps):
            return False
        self.index += 1
        return True

    def retrieve(self):
        return True, np.full((100, 160, 3), (self.index - 1) * 4, np.uint8)

    def read(self):
        return self.retrieve() if self.grab() else (False, None)

    def release(self):
        pass


def test_pts_define_requested_interval_even_beyond_nominal_duration(monkeypatch):
    cap = TimestampCapture([0, 0.05, 0.1, 0.15, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.1])
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: cap)
    reader = FileReader("variable.mp4", start=0.7, duration=0.4, analysis_fps=10)
    items = []
    while (item := reader.next()) is not None:
        items.append(item)
    assert [item[1] for item in items] == [4, 6, 8, 10]
    assert [item[2] for item in items] == pytest.approx([0.7, 0.8, 0.9, 1.0])
    assert cap.set_calls == []
    assert reader.source_frame_basis == "decoded_frame_count"
    assert reader.timestamp_fallback_frames == 0
    assert reader.finish_reason == "requested_duration"


def test_pts_gap_skips_empty_intervals_instead_of_burst_sampling(monkeypatch):
    cap = TimestampCapture([0, 0.8, 0.84, 0.88, 0.92, 1.0])
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: cap)
    reader = FileReader("gap.mp4", analysis_fps=10)
    times = []
    while (item := reader.next()) is not None:
        times.append(item[2])
    assert times == pytest.approx([0, 0.8, 0.92, 1])


def test_invalid_timestamps_have_explicit_fallback(monkeypatch):
    cap = TimestampCapture([0, 0, 0.08, 0.04, float("nan"), 0.2])
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: cap)
    reader = FileReader("broken_pts.mp4", analysis_fps=25)
    times, bases = [], []
    while (item := reader.next()) is not None:
        times.append(item[2])
        bases.append(reader.time_basis)
    assert times == pytest.approx([0, 0.04, 0.08, 0.12, 0.16, 0.2])
    assert bases == ["file_pts", "nominal_fps_fallback", "file_pts", "nominal_fps_fallback",
                     "nominal_fps_fallback", "file_pts"]
    assert reader.timestamp_fallback_frames == 3


def test_sampling_manifest_records_decoded_time_and_group(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from person_tracking.dataset import run_sampling
    timestamps = [0, 0.05, 0.1, 0.15, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.1]
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: TimestampCapture(timestamps))
    cfg = load_config()
    cfg.pop("root")
    cfg["cameras"] = [cfg["cameras"][0]]
    cfg["cameras"][0]["recording_start"] = "2026-10-20T15:00:00"
    cfg["project_root"] = "."
    config = tmp_path / "cameras.json"
    config.write_text(json.dumps(cfg), encoding="utf-8")
    args = SimpleNamespace(interval=0.1, group_seconds=600, guard_seconds=0, duration=0.4,
                           batch_size=2, start=0.7, start_clock=None, out=tmp_path / "samples",
                           config=config, cameras=None, no_proposals=True, decode_mode="sequential", seed=42)
    run_sampling(args)
    rows = [json.loads(line) for line in (args.out / "samples.jsonl").read_text(encoding="utf8").splitlines()]
    assert [r["source_frame"] for r in rows] == [4, 6, 8, 10]
    assert [r["video_time_s"] for r in rows] == pytest.approx([0.7, 0.8, 0.9, 1])
    assert all(r["time_basis"] == "file_pts" and r["source_frame_basis"] == "decoded_frame_count" for r in rows)
    assert len({r["split_group"] for r in rows}) == 1
    assert all(r["review_status"] == "pending" and r["split"] == "train" for r in rows)
    assert json.loads((args.out / "summary.json").read_text())["training_started"] is False
