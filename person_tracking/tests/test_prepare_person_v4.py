import json
import shutil

import numpy as np
import pytest

from person_tracking.common import read_jsonl, write_json
from person_tracking.dataset import save_jpeg, write_labels
from person_tracking.experiments import prepare_person_v4
from person_tracking.review import ReviewStore


def rewrite_rows(store):
    (store.root / "samples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in store.rows), encoding="utf-8")


@pytest.fixture
def completed_review(tmp_path):
    reference = tmp_path / "reference"
    for folder in ("images", "reviewed_labels", "proposals/labels"):
        (reference / folder).mkdir(parents=True)
    rows = []
    for i, split in enumerate(("train", "train", "val", "val", "test", "test")):
        sid = f"sample_{i}"
        row = {"id": sid, "camera_id": "D01", "split": split, "split_group": split,
               "width": 160, "height": 100, "image": f"images/{sid}.jpg",
               "proposal_label": f"proposals/labels/{sid}.txt", "proposal_count": 1,
               "recording_time_estimate": "2026-10-20T15:01:30", "annotation_locked": False}
        save_jpeg(reference / row["image"], np.full((100, 160, 3), i * 20, dtype=np.uint8))
        write_labels(reference / row["proposal_label"], [[0.5, 0.5, 0.2, 0.6]])
        rows.append(row)
    (reference / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    write_json(reference / "summary.json", {"status": "completed", "test_set_sealed": True})
    base = ReviewStore(reference)
    for i in range(6):
        base.save(i, "reviewed", [[0.5, 0.5, 0.2, 0.6]])
        base.rows[i]["annotation_locked"] = base.rows[i]["split"] == "test"
    rewrite_rows(base)
    review = tmp_path / "review"
    shutil.copytree(reference, review)
    provenance = review / "provenance"
    provenance.mkdir()
    for name in ("samples.jsonl", "reviews.json"):
        shutil.copy2(reference / name, provenance / ("reference_" + name))
    hashes = [{"id": r["id"], "split": r["split"], "camera_id": r["camera_id"],
               "image_sha256": prepare_person_v4.sha256(reference / r["image"]),
               "label_sha256": prepare_person_v4.sha256(base.label_path(r))} for r in rows]
    write_json(provenance / "initial_fingerprints.json", hashes)
    write_json(review / "summary.json", {"status": "completed", "test_set_sealed": True,
        "require_complete_review": True, "split_reference_dataset": str(reference)})
    return review


def test_export_preserves_negatives_and_fixed_test(completed_review, tmp_path):
    store = ReviewStore(completed_review)
    store.save(1, "reviewed", [], confirm_empty=True)
    exported, audit_dir = tmp_path / "export", tmp_path / "audit"
    audit = prepare_person_v4.prepare_dataset(completed_review, audit_dir, exported)
    assert audit["status"] == "passed"
    assert audit["by_split"]["train"]["negative_images"] == 1
    assert (exported / "labels/train/sample_1.txt").read_bytes() == b""
    assert audit["test_images_and_labels_unchanged"] == 2
    assert audit["test_split_evaluated"] is False
    assert audit["training_started"] is False
    assert audit["split_reselected"] is False
    assert audit["corrected_labels_since_review_preparation"] == [{"id": "sample_1", "split": "train"}]
    exported_rows = read_jsonl(exported / "exported_samples.jsonl")
    assert [r["split"] for r in exported_rows] == [r["split"] for r in store.rows]
    before = store.label_path(store.rows[0]).read_bytes()
    (exported / "labels/train/sample_0.txt").write_bytes(b"export changed")
    assert store.label_path(store.rows[0]).read_bytes() == before


def test_pending_review_cannot_create_export(completed_review, tmp_path):
    store = ReviewStore(completed_review)
    store.reviews["sample_0"]["status"] = "pending"
    write_json(store.review_path, store.reviews)
    with pytest.raises(ValueError, match="Incomplete annotation"):
        prepare_person_v4.prepare_dataset(completed_review, tmp_path / "audit", tmp_path / "export")
    assert not (tmp_path / "audit").exists()
    assert not (tmp_path / "export").exists()


def test_changed_test_box_is_rejected_even_if_review_count_matches(completed_review, tmp_path):
    store = ReviewStore(completed_review)
    write_labels(store.label_path(store.rows[4]), [[0.4, 0.5, 0.2, 0.6]])
    with pytest.raises(ValueError, match="Sealed test annotation changed"):
        prepare_person_v4.prepare_dataset(completed_review, tmp_path / "audit", tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_validation_image_cannot_be_moved_into_training(completed_review, tmp_path):
    store = ReviewStore(completed_review)
    store.rows[2]["split"] = "train"
    store.rows[2]["split_group"] = "train"
    rewrite_rows(store)
    with pytest.raises(ValueError, match="metadata or split changed"):
        prepare_person_v4.prepare_dataset(completed_review, tmp_path / "audit", tmp_path / "export")


def test_excluded_training_image_is_recorded_and_omitted(completed_review, tmp_path):
    store = ReviewStore(completed_review)
    store.save(1, "excluded")
    exported = tmp_path / "export"
    audit = prepare_person_v4.prepare_dataset(completed_review, tmp_path / "audit", exported)
    assert audit["excluded_images"] == [{"id": "sample_1", "split": "train"}]
    assert not (exported / "images/train/sample_1.jpg").exists()
    assert len(read_jsonl(exported / "exported_samples.jsonl")) == 5


def test_annotation_change_during_export_prevents_passed_audit(completed_review, tmp_path, monkeypatch):
    original_export = prepare_person_v4.export_dataset

    def concurrent_export(source, destination):
        result = original_export(source, destination)
        store = ReviewStore(source)
        store.save(0, "reviewed", [[0.4, 0.5, 0.2, 0.6]])
        return result

    monkeypatch.setattr(prepare_person_v4, "export_dataset", concurrent_export)
    exported, audit_dir = tmp_path / "export", tmp_path / "audit"
    with pytest.raises(RuntimeError, match="changed during audit/export"):
        prepare_person_v4.prepare_dataset(completed_review, audit_dir, exported)
    assert not (exported / "annotation_audit.json").exists()
    assert not (audit_dir / "audit.json").exists()
