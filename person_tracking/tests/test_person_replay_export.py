import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from person_tracking.common import read_jsonl, write_json
from person_tracking.dataset import save_jpeg, write_labels
from person_tracking.experiments.complete_person_v5 import fingerprint_export, sha256
from person_tracking.experiments.prepare_person_replay import prepare
from person_tracking.experiments.train_person_v2 import validate_export_contract


def register_export(root, rows):
    (root / "exported_samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf8")
    write_json(root / "data.yaml", {"path": str(root), "train": "images/train", "val": "images/val", "test": "images/test", "names": ["person"]})
    write_json(root / "summary.json", {"status": "completed"})
    fingerprints = root / "fingerprints.json"
    write_json(fingerprints, fingerprint_export(root, {r["id"]: r for r in rows}))
    write_json(root / "annotation_audit.json", {"status": "passed", "review_status": {"pending": 0},
        "exported_dataset": str(root), "fingerprints_file": str(fingerprints), "fingerprints_sha256": sha256(fingerprints)})


def add_sample(root, sid, camera, split, group, positive, gray):
    for folder in ("images", "labels"):
        (root / folder / split).mkdir(parents=True, exist_ok=True)
    image, label = f"images/{split}/{sid}.jpg", f"labels/{split}/{sid}.txt"
    save_jpeg(root / image, np.full((20, 30, 3), gray, np.uint8))
    write_labels(root / label, [[.5, .5, .2, .5]] if positive else [])
    return {"id": sid, "camera_id": camera, "split": split, "split_group": group,
        "image": image, "label": label, "width": 30, "height": 20, "review_status": "reviewed",
        "review": {"status": "reviewed", "box_count": int(positive), "confirmed_empty": not positive}}


@pytest.fixture
def replay_sources(tmp_path):
    base, specialist = tmp_path / "base", tmp_path / "specialist"
    rows = [add_sample(base, *entry) for entry in [
        ("old_person", "D01", "train", "2026-10-20T15:00:00", True, 10),
        ("old_empty", "D02", "train", "2026-10-20T15:00:00", False, 30),
        ("retired", "D03", "val", "2026-10-20T16:20:00", True, 50),
        ("keep_val", "D04", "val", "2026-10-20T15:10:00", True, 70),
        ("keep_test", "D03", "test", "2026-10-20T16:00:00", True, 90),
    ]]
    register_export(base, rows)
    new = [add_sample(specialist, *entry) for entry in [
        ("new_person", "D03", "train", "2026-10-20T16:20:00", True, 110),
        ("new_empty", "D03", "train", "2026-10-20T16:20:00", False, 130),
    ]]
    for row in rows[-2:]:
        for key in ("image", "label"):
            (specialist / row[key]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base / row[key], specialist / row[key])
        new.append(row)
    register_export(specialist, new)
    weights = tmp_path / "v9_best.pt"
    weights.write_bytes(b"synthetic bound checkpoint; never field results")
    return base, specialist, weights


def build(sources, tmp_path):
    base, specialist, weights = sources
    return prepare(base, specialist, tmp_path / "export", tmp_path / "audit", weights, 3)


def test_replay_preserves_unique_annotations_and_only_repeats_new_training(replay_sources, tmp_path):
    base, specialist, _ = replay_sources
    before = {str(p): sha256(p) for root in (base, specialist) for p in root.rglob("*") if p.is_file()}
    result = build(replay_sources, tmp_path)
    rows = read_jsonl(tmp_path / "export/exported_samples.jsonl")
    train = [r for r in rows if r["split"] == "train"]
    assert len(train) == 8
    assert {r["oversampling_original_id"] for r in train} == {"old_person", "old_empty", "new_person", "new_empty"}
    assert len([r for r in train if r["replay_origin"] == "old_replay"]) == 2
    assert len([r for r in train if r["replay_origin"] == "new_d03"]) == 6
    assert result["export_summary"]["training_unique_images"] == 4
    assert result["export_summary"]["unique_source_images"] == 6
    assert {r["id"] for r in rows if r["split"] == "val"} == {"keep_val"}
    assert {r["id"] for r in rows if r["split"] == "test"} == {"keep_test"}
    assert "retired" not in {r["id"] for r in rows}
    for row in rows:
        for key in ("image", "label"):
            assert sha256(tmp_path / "export" / row[key]) == sha256(Path(row["source_export"]) / row["source_export_" + key])
    assert all(sha256(Path(path)) == expected for path, expected in before.items())


@pytest.mark.parametrize("conflict", ["source_changed", "test_changed", "temporal_leak", "image_leak"])
def test_replay_refuses_corruption_and_cross_partition_leakage(replay_sources, tmp_path, conflict):
    base, specialist, _ = replay_sources
    rows = read_jsonl(specialist / "exported_samples.jsonl")
    if conflict == "source_changed":
        (specialist / rows[0]["label"]).write_text("", encoding="utf8")
    elif conflict == "test_changed":
        # Even an internally consistent specialist export cannot replace sealed test truth.
        write_labels(specialist / "labels/test/keep_test.txt", [[.5, .5, .3, .5]])
        register_export(specialist, rows)
    elif conflict == "temporal_leak":
        rows[0]["split_group"] = "2026-10-20T15:10:00"
        register_export(specialist, rows)
    else:
        shutil.copy2(base / "images/train/old_person.jpg", specialist / "images/val/keep_val.jpg")
        register_export(specialist, rows)
    with pytest.raises(ValueError):
        build(replay_sources, tmp_path)
    assert not (tmp_path / "export").exists()
    assert not (tmp_path / "audit").exists()


@pytest.mark.parametrize("change", ["route", "relative_root", "manifest", "extra_image", "extra_label"])
def test_export_contract_rejects_unregistered_training_inputs(replay_sources, tmp_path, change):
    audit = build(replay_sources, tmp_path)
    root = tmp_path / "export"
    assert validate_export_contract(root, audit)["partition_membership"] == {"train": 8, "val": 1, "test": 1}
    if change == "route":
        config = json.loads((root / "data.yaml").read_text())
        config["train"] = "images/val"
        write_json(root / "data.yaml", config)
    elif change == "relative_root":
        config = json.loads((root / "data.yaml").read_text())
        config["path"] = "."
        write_json(root / "data.yaml", config)
        # A legacy audit without YAML binding must still reject ambiguous routing.
        audit.pop("export_metadata_sha256")
    elif change == "manifest":
        (root / "exported_samples.jsonl").write_text("", encoding="utf8")
    elif change == "extra_image":
        shutil.copy2(root / "images/train/old_person.jpg", root / "images/train/unregistered.jpg")
    else:
        (root / "labels/train/unregistered.txt").write_text("", encoding="utf8")
    with pytest.raises(ValueError):
        validate_export_contract(root, audit)


@pytest.mark.parametrize("conflict", ["reencoded_source_frame", "evaluation_only", "excluded_dev_frame"])
def test_replay_excludes_evaluation_and_reencoded_frame_leakage(replay_sources, tmp_path, conflict):
    base, specialist, weights = replay_sources
    rows = read_jsonl(specialist / "exported_samples.jsonl")
    excluded = []
    if conflict == "reencoded_source_frame":
        # Different JPEG hashes and even a wrong temporal group cannot hide the same camera/source frame.
        for row in (rows[0], rows[-2]):
            row.update(source="same_source.mp4", source_frame=123, camera_id="D03")
    elif conflict == "evaluation_only":
        rows[0]["evaluation_only"] = True
    else:
        rows[0].update(source="same_source.mp4", source_frame=123)
        excluded_root = tmp_path / "development"
        excluded_root.mkdir()
        (excluded_root / "samples.jsonl").write_text(json.dumps({
            "id": "other_encoded_image", "source": "same_source.mp4", "camera_id": "D03", "source_frame": 123,
            "image_sha256": "different content hash; same source frame"}) + "\n", encoding="utf8")
        excluded = [excluded_root]
    register_export(specialist, rows)
    with pytest.raises(ValueError):
        prepare(base, specialist, tmp_path / "export", tmp_path / "audit", weights, 3, excluded)
    assert not (tmp_path / "export").exists()
