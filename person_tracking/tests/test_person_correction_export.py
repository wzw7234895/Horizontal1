import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from person_tracking.common import read_jsonl, write_json
from person_tracking.dataset import read_labels, save_jpeg, write_labels
from person_tracking.experiments.complete_person_correction import complete_correction
from person_tracking.experiments.complete_person_v5 import sha256
from person_tracking.experiments.train_person_v2 import validate_initialization
from person_tracking.review import ReviewStore


@pytest.fixture
def corrected(tmp_path):
    source, correction = tmp_path / "original", tmp_path / "correction"
    (source / "images").mkdir(parents=True)
    (source / "reviewed_labels").mkdir()
    rows, reviews = [], {}
    for index, split in enumerate(("train", "train", "val", "val", "test", "test", "train", "train")):
        sample_id = f"sample_{index}"
        empty = index in (1, 3, 5)
        row = {"id": sample_id, "camera_id": "D01", "split": split, "split_group": f"group_{index}",
               "image": f"images/{sample_id}.jpg", "source": "video.mp4", "source_frame": index,
               "recording_time_estimate": f"2026-10-20T15:{index:02d}:30", "width": 20, "height": 16,
               "proposal_count": int(not empty), "proposal_label": f"reviewed_labels/{sample_id}.txt"}
        save_jpeg(source / row["image"], np.full((16, 20, 3), 20 + index * 25, np.uint8))
        write_labels(source / "reviewed_labels" / f"{sample_id}.txt", [] if empty else [[0.5, 0.5, 0.3, 0.6]])
        rows.append(row)
        reviews[sample_id] = {"status": "reviewed", "box_count": int(not empty),
                              "confirmed_empty": empty, "reviewed_at": "2026-11-06T12:00:00",
                              "review_source": "local_review_interface"}
    (source / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    write_json(source / "reviews.json", reviews)
    write_json(source / "summary.json", {"status": "completed"})
    shutil.copytree(source, correction)
    backup = correction / "provenance/original"
    backup.mkdir(parents=True)
    initial = [{"path": relative, "sha256": sha256(source / relative)} for row in rows
               for relative in (row["image"], f"reviewed_labels/{row['id']}.txt")]
    write_json(backup / "original_image_label_fingerprints.json", initial)
    write_json(correction / "annotation_correction.json", {
        "review_scope": "previous_positive_images_only", "source_dataset": str(source),
        "source_metadata_sha256": {name: sha256(source / name) for name in ("samples.jsonl", "reviews.json", "summary.json")},
        "backup": str(backup), "inherited_confirmed_empty_images": 3})
    write_json(correction / "summary.json", {"status": "completed", "split_selection_status": "pending_reselection"})
    store = ReviewStore(correction)
    store.save(0, "reviewed", [[0.5, 0.5, 0.4, 0.8]])
    store.save(7, "excluded", [])
    initialization = tmp_path / "public.pt"
    initialization.write_bytes(b"clean public initialization")
    return correction, source, initialization


def complete_fixture(corrected, tmp_path):
    correction, _, initialization = corrected
    return complete_correction(correction, tmp_path / "snapshot", tmp_path / "audit", tmp_path / "export", initialization)


def test_correction_freezes_labels_preserves_splits_and_inherits_empty_reviews(corrected, tmp_path):
    correction, source, initialization = corrected
    before = {path.relative_to(correction): sha256(path) for path in correction.rglob("*") if path.is_file()}
    result = complete_fixture(corrected, tmp_path)
    assert result["status"] == "passed"
    assert result["historical_split_membership_preserved"] is True
    assert result["inherited_confirmed_empty_images"] == 3
    assert result["excluded_images"] == 1
    assert [item["id"] for item in result["changed_label_files"]] == ["sample_0"]
    exported = read_jsonl(tmp_path / "export/exported_samples.jsonl")
    assert len(exported) == 7
    old_splits = {row["id"]: row["split"] for row in read_jsonl(source / "samples.jsonl")}
    assert all(row["split"] == old_splits[row["id"]] for row in exported)
    assert not (tmp_path / "export/images/train/sample_7.jpg").exists()
    assert read_labels(tmp_path / "export/labels/train/sample_0.txt") == [[0.5, 0.5, 0.4, 0.8]]
    assert all(row["annotation_locked"] for row in ReviewStore(tmp_path / "snapshot").rows)
    for sample_id in ("sample_1", "sample_3", "sample_5"):
        assert ReviewStore(tmp_path / "snapshot").reviews[sample_id]["reviewed_at"] == "2026-11-06T12:00:00"
    assert all(sha256(correction / relative) == digest for relative, digest in before.items())
    validate_initialization(result, initialization)
    bad = tmp_path / "old_finetuned.pt"
    bad.write_bytes(b"old labels were used for this checkpoint")
    with pytest.raises(ValueError, match="clean public initialization"):
        validate_initialization(result, bad)


@pytest.mark.parametrize("problem", ["pending", "changed_empty", "split", "image", "count", "source_label"])
def test_invalid_or_unfinished_correction_never_creates_export(corrected, tmp_path, problem):
    correction, source, _ = corrected
    if problem == "pending":
        reviews = json.loads((correction / "reviews.json").read_text())
        reviews["sample_0"]["status"] = "pending"
        write_json(correction / "reviews.json", reviews)
    elif problem == "changed_empty":
        ReviewStore(correction).save(1, "reviewed", [[0.5, 0.5, 0.2, 0.4]])
    elif problem == "split":
        rows = read_jsonl(correction / "samples.jsonl")
        rows[0]["split"] = "test"
        (correction / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif problem == "image":
        (correction / "images/sample_0.jpg").write_bytes(b"modified image")
    elif problem == "count":
        write_labels(correction / "reviewed_labels/sample_0.txt", [])
    else:
        write_labels(source / "reviewed_labels/sample_0.txt", [[0.5, 0.5, 0.1, 0.1]])
    with pytest.raises(ValueError):
        complete_fixture(corrected, tmp_path)
    assert not any((tmp_path / name).exists() for name in ("snapshot", "audit", "export"))
