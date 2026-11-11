import json
from pathlib import Path

import numpy as np
import pytest

from person_tracking.common import read_jsonl, write_json
from person_tracking.dataset import save_jpeg, write_labels
from person_tracking.experiments.complete_d03_plainclothes import complete
from person_tracking.experiments.complete_person_v5 import sha256
from person_tracking.experiments.train_person_v2 import validate_initialization


def make_review(root, entries):
    (root / "images").mkdir(parents=True)
    (root / "reviewed_labels").mkdir()
    rows, reviews = [], {}
    for i, (sid, camera, split, group, positive) in enumerate(entries):
        save_jpeg(root / f"images/{sid}.jpg", np.full((16, 20, 3), 10 + i * 20, np.uint8))
        write_labels(root / f"reviewed_labels/{sid}.txt", [[0.5, 0.5, 0.5, 0.7]] if positive else [])
        rows.append({"id": sid, "camera_id": camera, "split": split, "split_group": group,
            "image": f"images/{sid}.jpg", "proposal_label": f"reviewed_labels/{sid}.txt",
            "width": 20, "height": 16, "proposal_count": int(positive),
            "source": "fixture.mp4", "recording_time_estimate": group,
            "image_sha256": sha256(root / f"images/{sid}.jpg")})
        reviews[sid] = {"status": "reviewed", "box_count": int(positive), "confirmed_empty": not positive,
                        "reviewed_at": "2026-11-07T20:00:00", "review_source": "unit_test_fixture"}
    (root / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    write_json(root / "reviews.json", reviews)
    write_json(root / "summary.json", {"status": "completed"})


@pytest.fixture
def specialist(tmp_path):
    base, new = tmp_path / "base", tmp_path / "review"
    make_review(base, [
        ("old_train", "D03", "train", "15:50", True),
        ("retire_d03", "D03", "val", "16:20", True),
        ("retire_d02", "D02", "val", "16:20", False),
        ("keep_val", "D01", "val", "15:10", True),
        ("keep_test", "D03", "test", "16:00", True),
    ])
    make_review(new, [("new_person", "D03", "train", "16:20", True),
                      ("new_empty", "D03", "train", "16:20", False)])
    write_json(new / "sampling_plan.json", {"source_dataset": str(base),
        "source_metadata_sha256": {name: sha256(base / name) for name in ("samples.jsonl", "reviews.json", "summary.json")}})
    weights = tmp_path / "v9_best.pt"
    weights.write_bytes(b"bound v9 checkpoint fixture")
    return new, base, weights


def execute(specialist, tmp_path):
    new, _, weights = specialist
    return complete(new, tmp_path / "snapshot", tmp_path / "audit", tmp_path / "export", weights)


def test_only_new_images_train_and_all_matching_camera_groups_leave_validation(specialist, tmp_path):
    new, base, weights = specialist
    before = {str(path): sha256(path) for root in (new, base) for path in root.rglob("*") if path.is_file()}
    result = execute(specialist, tmp_path)
    rows = read_jsonl(tmp_path / "export/exported_samples.jsonl")
    assert {r["id"] for r in rows if r["split"] == "train"} == {"new_person", "new_empty"}
    assert {r["id"] for r in rows if r["split"] == "val"} == {"keep_val"}
    assert {r["id"] for r in rows if r["split"] == "test"} == {"keep_test"}
    assert result["retired_validation_images"] == 2
    assert result["test_membership_and_hashes_preserved"]
    assert result["review_status"]["pending"] == 0
    assert sha256(tmp_path / "export/labels/test/keep_test.txt") == sha256(base / "reviewed_labels/keep_test.txt")
    assert all(sha256(Path(path)) == digest for path, digest in before.items())
    validate_initialization(result, weights)
    wrong = tmp_path / "wrong.pt"
    wrong.write_bytes(b"other weights")
    with pytest.raises(ValueError, match="checkpoint bound"):
        validate_initialization(result, wrong)


@pytest.mark.parametrize("problem", ["pending", "label_count", "test_group", "changed_image"])
def test_incomplete_or_conflicting_specialist_data_cannot_create_an_export(specialist, tmp_path, problem):
    new, _, _ = specialist
    if problem == "pending":
        reviews = json.loads((new / "reviews.json").read_text())
        del reviews["new_person"]
        write_json(new / "reviews.json", reviews)
    elif problem == "label_count":
        write_labels(new / "reviewed_labels/new_person.txt", [])
    elif problem == "test_group":
        rows = read_jsonl(new / "samples.jsonl")
        rows[0]["split_group"] = "16:00"
        (new / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        (new / "images/new_person.jpg").write_bytes(b"modified after review")
    with pytest.raises(ValueError):
        execute(specialist, tmp_path)
    assert not any((tmp_path / name).exists() for name in ("snapshot", "audit", "export"))
