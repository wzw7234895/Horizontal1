"""Copy reviewed annotations into a correction version with old test locks removed."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

from person_tracking.common import read_jsonl, write_json
from person_tracking.dataset import read_labels
from person_tracking.review import ReviewStore


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.out.resolve()
    if output.exists():
        raise FileExistsError(output)
    if source == output or source in output.parents:
        raise ValueError("Correction directory must be separate from its source")
    store = ReviewStore(source)
    review_hash = sha256(store.review_path)
    rows = store.rows
    tests = [r for r in rows if r["split"] == "test"]
    if not tests:
        raise ValueError("Source has no old test samples")
    shutil.copytree(source, output)
    now = datetime.now().isoformat()
    unlocked = []
    for row in rows:
        if row["split"] == "test":
            row["annotation_locked"] = False
            row["previous_test_membership"] = True
            row["test_membership_status"] = "invalidated_pending_reselection"
            row["annotation_unsealed_at"] = now
            unlocked.append(row["id"])
    (output / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf8")
    summary = json.loads((output / "summary.json").read_text(encoding="utf8"))
    summary.update(split_selection_status="pending_reselection", test_set_sealed=False,
        test_set_unsealed_at=now, correction_source_dataset=str(source),
        correction_reason="User reported incorrect annotations, including in the old test split; correct labels and select splits again.",
        annotation_lock_count=sum(bool(r.get("annotation_locked")) for r in rows))
    write_json(output / "summary.json", summary)
    fingerprints = []
    for row in rows:
        for relative in (row["image"], f"reviewed_labels/{row['id']}.txt"):
            expected = sha256(source / relative)
            if sha256(output / relative) != expected:
                raise ValueError(f"Copy mismatch: {relative}")
            fingerprints.append({"path": relative, "sha256": expected})
        review = store.reviews[row["id"]]
        if review["status"] == "reviewed":
            boxes = read_labels(output / f"reviewed_labels/{row['id']}.txt")
            if len(boxes) != review["box_count"] or (not boxes and not review["confirmed_empty"]):
                raise ValueError(f"Review and label count disagree: {row['id']}")
    if review_hash != sha256(store.review_path) or review_hash != sha256(output / "reviews.json"):
        raise RuntimeError("Source annotations changed while copying; repeat with a new destination")
    record = {"status": "unsealed_for_correction", "created_at": now, "source_dataset": str(source),
        "correction_dataset": str(output), "old_test_images_unlocked": len(unlocked),
        "old_test_sample_ids": unlocked, "all_images_and_annotations_copied_without_change": True,
        "source_reviews_sha256": review_hash, "split_selection_status": "pending_reselection",
        "old_split_labels_are_for_navigation_only": True, "training_remains_paused": True,
        "note": "Correct labels before selecting train/val/test again; old test membership is no longer a sealed evaluation split."}
    write_json(output / "split_reselection.json", record)
    provenance = output / "provenance/test_reopen"
    provenance.mkdir(parents=True)
    for name in ("samples.jsonl", "summary.json", "reviews.json"):
        shutil.copy2(source / name, provenance / ("source_" + name))
    write_json(provenance / "initial_fingerprints.json", fingerprints)
    correction = ReviewStore(output)
    assert all(not item["annotation_locked"] for item in correction.items() if item["split"] == "test")
    print(json.dumps({**record, "review_status": correction.status()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
