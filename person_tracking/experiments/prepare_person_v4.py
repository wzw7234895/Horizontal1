"""Audit completed negative reviews and export with the existing v3 split fixed."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

from person_tracking.common import read_jsonl, write_json
from person_tracking.dataset import read_image, read_labels
from person_tracking.review import ReviewStore, export_dataset


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_dataset(dataset, output, exported):
    source, output, exported = (Path(p).resolve() for p in (dataset, output, exported))
    if output == exported or output.exists() or exported.exists():
        raise FileExistsError("Choose separate new audit and export directories")
    store = ReviewStore(source)
    status = store.status()
    if status["pending"] or status["reviewed"] + status["excluded"] != status["total"]:
        raise ValueError(f"Incomplete annotation: {status['pending']} pending")
    if not store.summary.get("test_set_sealed"):
        raise ValueError("The reference test split must remain sealed")
    reference = ReviewStore(store.summary["split_reference_dataset"])
    if not reference.summary.get("test_set_sealed"):
        raise ValueError("The reference dataset is not sealed")
    provenance = source / "provenance"
    reference_rows = {r["id"]: r for r in read_jsonl(provenance / "reference_samples.jsonl")}
    live_reference_rows = {r["id"]: r for r in reference.rows}
    initial = {r["id"]: r for r in read_json(provenance / "initial_fingerprints.json")}
    reference_reviews = read_json(provenance / "reference_reviews.json")
    ids = {r["id"] for r in store.rows}
    if ids != set(reference_rows) or ids != set(live_reference_rows) or ids != set(initial):
        raise ValueError("Sample membership changed since the review was prepared")
    if set(store.reviews) != ids:
        raise ValueError("Review records do not cover exactly the expected samples")
    guarded_paths = [source / name for name in ("summary.json", "samples.jsonl", "reviews.json")]
    guarded_paths += [provenance / name for name in
                      ("reference_samples.jsonl", "initial_fingerprints.json", "reference_reviews.json")]
    guarded_paths += [reference.root / name for name in ("samples.jsonl", "reviews.json", "summary.json")]
    original_hashes = {path: sha256(path) for path in guarded_paths}
    fingerprints, excluded, changed = [], [], []
    image_ids, group_splits = defaultdict(list), {}
    by_split, by_camera_split = defaultdict(Counter), defaultdict(Counter)
    test_ids, small_boxes = set(), 0
    identity_fields = ("split", "split_group", "camera_id", "width", "height")
    for row in store.rows:
        sid, split = row["id"], row["split"]
        saved, base = reference_rows[sid], live_reference_rows[sid]
        if split not in ("train", "val", "test") or any(
                row.get(field) != saved.get(field) or saved.get(field) != base.get(field)
                for field in identity_fields):
            raise ValueError(f"Fixed sample metadata or split changed: {sid}")
        if initial[sid]["split"] != split or initial[sid]["camera_id"] != row["camera_id"]:
            raise ValueError(f"Initial fingerprint metadata disagrees: {sid}")
        previous = group_splits.setdefault(row["split_group"], split)
        if previous != split:
            raise ValueError("One temporal group appears in multiple splits")
        if bool(row.get("annotation_locked")) != (split == "test"):
            raise ValueError(f"Annotation lock disagrees with split: {sid}")
        review = store.reviews[sid]
        image_path, label_path = source / row["image"], store.label_path(row)
        image_hash = sha256(image_path)
        if image_hash != initial[sid]["image_sha256"] or image_hash != sha256(reference.root / base["image"]):
            raise ValueError(f"Source image changed: {sid}")
        original_hashes[image_path] = image_hash
        original_hashes[label_path] = sha256(label_path)
        image_ids[image_hash].append(sid)
        if split == "test":
            test_ids.add(sid)
            if review != reference_reviews[sid] or review != reference.reviews[sid] or review["status"] != "reviewed":
                raise ValueError(f"Sealed test review changed: {sid}")
            if original_hashes[label_path] != initial[sid]["label_sha256"] or original_hashes[label_path] != sha256(reference.label_path(base)):
                raise ValueError(f"Sealed test annotation changed: {sid}")
        elif review["status"] == "excluded":
            excluded.append({"id": sid, "split": split})
            continue
        if review["status"] != "reviewed":
            raise ValueError(f"Unexpected annotation status: {sid}")
        boxes = read_labels(label_path)
        if len(boxes) != review.get("box_count") or review.get("confirmed_empty") is not (not boxes):
            raise ValueError(f"Review/label disagreement or unconfirmed negative: {sid}")
        h, w = read_image(image_path).shape[:2]
        if (w, h) != (row["width"], row["height"]):
            raise ValueError(f"Image dimensions disagree: {sid}")
        small_boxes += sum(min(b[2] * w, b[3] * h) < 3 for b in boxes)
        counts = {"images": 1, "positive_images": int(bool(boxes)), "negative_images": int(not boxes), "boxes": len(boxes)}
        by_split[split].update(counts)
        by_camera_split[f"{row['camera_id']}_{split}"].update(counts)
        if original_hashes[label_path] != initial[sid]["label_sha256"]:
            changed.append({"id": sid, "split": split})
        fingerprints.append({"id": sid, "split": split, "camera_id": row["camera_id"],
            "image": row["image"], "image_sha256": image_hash,
            "label": label_path.relative_to(source).as_posix(), "label_sha256": original_hashes[label_path]})
    duplicates = [values for values in image_ids.values() if len(values) > 1]
    if duplicates:
        raise ValueError(f"Duplicate source images: {duplicates}")
    if test_ids != {sid for sid, row in reference_rows.items() if row["split"] == "test"}:
        raise ValueError("Sealed test membership changed")
    if any(by_split[split]["positive_images"] == 0 for split in ("train", "val", "test")):
        raise ValueError("Every split must retain a reviewed positive image")

    def assert_stable():
        for path, expected in original_hashes.items():
            if sha256(path) != expected:
                raise RuntimeError(f"Annotations or source changed during audit/export: {path}")

    assert_stable()
    output.mkdir(parents=True)
    for name in ("reviews.json", "samples.jsonl", "summary.json"):
        shutil.copy2(source / name, output / ("source_" + name))
    export_summary = export_dataset(source, exported)
    exported_rows = {r["id"]: r for r in read_jsonl(exported / "exported_samples.jsonl")}
    if set(exported_rows) != {r["id"] for r in fingerprints}:
        raise ValueError("Exported membership disagrees with the audited selection")
    for item in fingerprints:
        copied = exported_rows[item["id"]]
        for kind in ("image", "label"):
            if sha256(exported / copied[kind]) != item[f"{kind}_sha256"]:
                raise ValueError(f"Export fingerprint mismatch: {item['id']}")
    assert_stable()
    fingerprint_path = output / "fingerprints.json"
    write_json(fingerprint_path, fingerprints)
    audit = {"status": "passed", "audited_at": datetime.now().isoformat(), "source_dataset": str(source),
        "exported_dataset": str(exported), "review_status": status, "by_split": dict(by_split),
        "by_camera_split": dict(by_camera_split), "fixed_split_reference": str(reference.root),
        "group_splits": group_splits, "split_reselected": False, "excluded_images": excluded,
        "corrected_labels_since_review_preparation": changed, "all_images_decoded": len(fingerprints),
        "unique_image_sha256": len(image_ids), "temporal_group_cross_split_conflicts": 0,
        "test_images_and_labels_unchanged": len(test_ids), "test_set_sealed": True,
        "test_split_evaluated": False, "training_started": False,
        "boxes_shorter_than_3_pixels": small_boxes, "export_summary": export_summary,
        "source_reviews_sha256": original_hashes[store.review_path],
        "fingerprints_file": str(fingerprint_path), "fingerprints_sha256": sha256(fingerprint_path),
        "limitations": ["Structural integrity and completed reviews do not certify semantic label accuracy.",
            "The same-day validation set is for development; this is not a cross-day field acceptance result.",
            "Compare models on the same corrected validation labels, not on historical metrics with older labels."]}
    write_json(output / "audit.json", audit)
    write_json(exported / "annotation_audit.json", audit)
    return audit


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_dataset(args.dataset, args.out, args.export)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
