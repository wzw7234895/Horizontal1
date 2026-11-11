"""Audit completed person annotations and export an immutable training snapshot."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

from person_tracking.common import PACKAGE_DIR, read_jsonl, write_json
from person_tracking.dataset import read_image, read_labels
from person_tracking.review import ReviewStore, export_dataset


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    args = parser.parse_args()
    source, output, exported = args.dataset.resolve(), args.out.resolve(), args.export.resolve()
    if output.exists() or exported.exists():
        raise FileExistsError("Audit and export destinations must both be new directories")
    store = ReviewStore(source)
    status = store.status()
    if status["pending"] or status["reviewed"] + status["excluded"] != status["total"]:
        raise ValueError(f"Incomplete annotation: {status}")
    reference = PACKAGE_DIR / "data/person_reviewed_v1"
    old_rows = {r["id"]: r for r in read_jsonl(reference / "exported_samples.jsonl")}
    old_hashes = json.loads((PACKAGE_DIR / "runs/annotation_check_20261105/label_sha256.json").read_text(encoding="utf8"))
    review_hash = sha256(store.review_path)
    group_splits, image_ids = {}, defaultdict(list)
    by_split, by_camera = defaultdict(Counter), defaultdict(Counter)
    fingerprints, changed_inherited_labels, test_errors = [], [], []
    small_boxes = 0
    for index, row in enumerate(store.rows):
        previous = group_splits.setdefault(row["split_group"], row["split"])
        if previous != row["split"]:
            raise ValueError(f"Temporal group crosses splits: {row['split_group']}")
        review = store.reviews[row["id"]]
        if review["status"] == "excluded":
            if row["split"] == "test":
                test_errors.append(f"Excluded sealed test: {row['id']}")
            continue
        image_path, label_path = source / row["image"], store.label_path(row)
        boxes = read_labels(label_path)
        if len(boxes) != review["box_count"] or bool(review["confirmed_empty"]) != (not boxes):
            raise ValueError(f"Review/label disagreement: {row['id']}")
        image = read_image(image_path)
        h, w = image.shape[:2]
        if (w, h) != (row["width"], row["height"]):
            raise ValueError(f"Image dimensions disagree: {row['id']}")
        ih, lh = sha256(image_path), sha256(label_path)
        image_ids[ih].append((row["id"], row["split"]))
        for box in boxes:
            small_boxes += min(box[2] * w, box[3] * h) < 3
        counts = {"images": 1, "positive_images": int(bool(boxes)), "negative_images": int(not boxes), "boxes": len(boxes)}
        by_split[row["split"]].update(counts)
        by_camera[row["camera_id"]].update(counts)
        inherited = old_rows.get(row["id"])
        if inherited:
            if row["split"] != inherited["split"] or ih != sha256(reference / inherited["image"]):
                raise ValueError(f"Inherited image or split changed: {row['id']}")
            if lh != sha256(reference / inherited["label"]):
                changed_inherited_labels.append({"id": row["id"], "split": row["split"]})
        if row["split"] == "test":
            if not inherited or not row.get("annotation_locked") or lh != old_hashes[row["id"]]:
                test_errors.append(row["id"])
        fingerprints.append({"id": row["id"], "split": row["split"], "camera_id": row["camera_id"],
            "image": row["image"], "image_sha256": ih, "label": str(label_path.relative_to(source)), "label_sha256": lh})
        if (index + 1) % 100 == 0:
            print(f"Audited {index + 1}/{len(store.rows)} images", flush=True)
    duplicates = [rows for rows in image_ids.values() if len(rows) > 1]
    if duplicates or test_errors:
        raise ValueError(f"Duplicate images={duplicates}; sealed test errors={test_errors}")
    test_ids = {r["id"] for r in fingerprints if r["split"] == "test"}
    if test_ids != {k for k, r in old_rows.items() if r["split"] == "test"}:
        raise ValueError("Sealed test membership changed")
    if sha256(store.review_path) != review_hash:
        raise ValueError("Annotations changed during audit; repeat after saving all edits")
    for item in fingerprints:
        if sha256(source / item["label"]) != item["label_sha256"]:
            raise ValueError(f"Label changed during audit: {item['id']}")
    output.mkdir(parents=True)
    shutil.copy2(store.review_path, output / "reviews_snapshot.json")
    shutil.copy2(source / "samples.jsonl", output / "samples_snapshot.jsonl")
    export_summary = export_dataset(source, exported)
    export_rows = {r["id"]: r for r in read_jsonl(exported / "exported_samples.jsonl")}
    for item in fingerprints:
        saved = export_rows[item["id"]]
        if sha256(exported / saved["image"]) != item["image_sha256"] or sha256(exported / saved["label"]) != item["label_sha256"]:
            raise ValueError(f"Export fingerprint mismatch: {item['id']}")
    if sha256(store.review_path) != review_hash:
        raise ValueError("Annotations changed during export; do not train this snapshot")
    fingerprint_path = output / "fingerprints.json"
    write_json(fingerprint_path, fingerprints)
    audit = {"status": "passed", "audited_at": datetime.now().isoformat(), "source_dataset": str(source),
        "exported_dataset": str(exported), "review_status": status, "by_split": dict(by_split), "by_camera": dict(by_camera),
        "all_images_decoded": len(fingerprints), "unique_image_sha256": len(image_ids),
        "image_duplicates": duplicates, "temporal_group_cross_split_conflicts": 0,
        "inherited_labels_changed": changed_inherited_labels, "test_images_and_labels_unchanged": len(test_ids),
        "test_split_evaluated": False, "training_started": False, "boxes_shorter_than_3_pixels": small_boxes,
        "reviews_sha256": review_hash, "fingerprints_file": str(fingerprint_path),
        "fingerprints_sha256": sha256(fingerprint_path), "export_summary": export_summary,
        "limitations": ["Structural audit and human review completion do not independently certify semantic annotation accuracy.",
            "Same-day temporally correlated footage; no cross-day generalization claim."]}
    write_json(output / "audit.json", audit)
    shutil.copy2(output / "audit.json", exported / "annotation_audit.json")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
