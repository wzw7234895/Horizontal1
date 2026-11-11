from datetime import datetime
from types import SimpleNamespace

from person_tracking.experiments.complete_person_v5 import (
    fingerprint_export,
    load_reference_rows,
    resolve_reference_dataset,
    sha256,
)
from person_tracking.experiments.prepare_person_hard_negative_review import select_candidates


def row(frame, timestamp, detections):
    return {
        "source_frame": frame,
        "recording_time_estimate": timestamp,
        "detections": detections,
    }


def test_candidate_mining_prefers_unassociated_tracks_and_samples_empty_frames():
    rows = [
        row(100, "2026-10-20T15:02:01", [{"confidence": 0.95, "track_id": 8}]),
        row(200, "2026-10-20T15:02:08", [{"confidence": 0.30, "track_id": None}]),
        row(300, "2026-10-20T15:03:01", []),
        row(400, "2026-10-20T15:12:10", [{"confidence": 0.99, "track_id": None}]),
    ]
    train_group = {"2026-10-20T15:00:00"}

    selected = select_candidates(rows, train_group, [], {8: 100})

    assert [item[0]["source_frame"] for item in selected] == [200, 300]
    assert selected[0][1]["candidate_type"] == "detector_candidate_review"
    assert selected[0][1]["suspected_unassociated_detection"] is True
    assert selected[1][1]["candidate_type"] == "no_detection_review"


def test_candidate_mining_skips_val_groups_and_near_existing_training_samples():
    rows = [
        row(100, "2026-10-20T15:02:01", [{"confidence": 0.90, "track_id": 4}]),
        row(200, "2026-10-20T15:12:10", [{"confidence": 0.99, "track_id": None}]),
    ]
    existing = [datetime.fromisoformat("2026-10-20T15:02:00")]

    selected = select_candidates(rows, {"2026-10-20T15:00:00"}, existing, {4: 30})

    assert selected == []


def test_reference_dataset_resolves_from_locked_base_summary_when_manifest_omits_it(tmp_path):
    reference = tmp_path / "v4_export"
    base = SimpleNamespace(summary={"split_reference_dataset": str(reference)})

    assert resolve_reference_dataset({"base_dataset": "base"}, base) == reference.resolve()


def test_explicit_reference_dataset_takes_precedence(tmp_path):
    manifest_reference = tmp_path / "manifest_export"
    base = SimpleNamespace(summary={"split_reference_dataset": str(tmp_path / "other")})

    assert resolve_reference_dataset({"base_exported_dataset": str(manifest_reference)}, base) == manifest_reference.resolve()


def test_frozen_review_manifest_resolves_test_image_and_label_paths(tmp_path):
    (tmp_path / "reviewed_labels").mkdir()
    (tmp_path / "samples.jsonl").write_text(
        '{"id":"sample_1","image":"images/sample_1.jpg","split":"test"}\n',
        encoding="utf-8",
    )

    rows = load_reference_rows(tmp_path)

    assert rows["sample_1"] == {
        "image": "images/sample_1.jpg",
        "label": "reviewed_labels/sample_1.txt",
        "split": "test",
    }


def test_export_fingerprints_cover_every_image_and_label(tmp_path):
    (tmp_path / "images/train").mkdir(parents=True)
    (tmp_path / "labels/train").mkdir(parents=True)
    (tmp_path / "images/train/sample_1.jpg").write_bytes(b"image")
    (tmp_path / "labels/train/sample_1.txt").write_bytes(b"label")
    rows = {"sample_1": {"id": "sample_1", "camera_id": "D01", "split": "train",
                         "image": "images/train/sample_1.jpg", "label": "labels/train/sample_1.txt"}}

    fingerprints = fingerprint_export(tmp_path, rows)

    assert fingerprints == [{
        "id": "sample_1", "camera_id": "D01", "split": "train",
        "image_sha256": sha256(tmp_path / "images/train/sample_1.jpg"),
        "label_sha256": sha256(tmp_path / "labels/train/sample_1.txt"),
    }]


