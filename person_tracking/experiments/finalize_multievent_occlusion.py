"""Freeze completed v12 labels and merge reviewed additions into a new export.

This command audits the standalone review directory, freezes the reviewed
images without excluded samples, and creates a new YOLO export by appending
the reviewed additions to the v11 training partition. The v11 validation and
test partitions are copied byte-for-byte and are never used for gradients.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

from person_tracking.common import read_jsonl, write_json
from person_tracking.dataset import read_image, read_labels, write_labels
from person_tracking.review import ReviewStore


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path):
    return read_jsonl(path)


def copy_checked(source: Path, destination: Path, expected: str | None = None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    actual = sha256(destination)
    if expected and actual != expected:
        raise RuntimeError(f"Copied file fingerprint changed: {source}")
    return actual


def audit_review(review_root: Path):
    store = ReviewStore(review_root)
    status = store.status()
    if status["pending"]:
        raise ValueError(f"There are still {status['pending']} pending images")
    if status["reviewed"] + status["excluded"] != status["total"]:
        raise ValueError("Every review image must be reviewed or explicitly excluded")
    if status["total"] != len(store.rows):
        raise ValueError("Review status total disagrees with manifest")
    kept, excluded = [], []
    boxes_by_camera = Counter()
    split_groups = {}
    image_hashes = set()
    label_hashes = {}
    for row in store.rows:
        review = store.reviews.get(row["id"], {})
        source = (review_root / row["image"]).resolve()
        if not source.is_relative_to(review_root) or not source.is_file():
            raise ValueError(f"Invalid or missing source image: {row['id']}")
        image_hash = sha256(source)
        if row.get("image_sha256") and image_hash != row["image_sha256"]:
            raise ValueError(f"Source image changed: {row['id']}")
        image = read_image(source)
        if (image.shape[1], image.shape[0]) != (row["width"], row["height"]):
            raise ValueError(f"Image dimensions disagree: {row['id']}")
        if row["split"] != "train":
            raise ValueError(f"Supplemental sample is outside train: {row['id']}")
        old = split_groups.setdefault(row["split_group"], row["split"])
        if old != row["split"]:
            raise ValueError(f"Supplemental temporal group crosses split: {row['split_group']}")
        if review["status"] == "excluded":
            excluded.append({**row, "review": review})
            continue
        if review["status"] != "reviewed":
            raise ValueError(f"Unexpected review status: {row['id']}")
        label = store.label_path(row)
        if not label.is_file():
            raise ValueError(f"Missing reviewed label: {row['id']}")
        boxes = read_labels(label)
        if len(boxes) != review.get("box_count"):
            raise ValueError(f"Review box count disagrees: {row['id']}")
        if bool(review.get("confirmed_empty")) != (not boxes):
            raise ValueError(f"Review empty confirmation disagrees: {row['id']}")
        if image_hash in image_hashes:
            raise ValueError(f"Duplicate supplemental image: {row['id']}")
        image_hashes.add(image_hash)
        label_hash = sha256(label)
        label_hashes[row["id"]] = label_hash
        boxes_by_camera[row["camera_id"]] += len(boxes)
        kept.append({"row": row, "review": review, "boxes": boxes,
                     "image_sha256": image_hash, "label_sha256": label_hash})
    return store, kept, excluded, {"status": status, "split_groups": split_groups,
                                   "boxes_by_camera": dict(boxes_by_camera),
                                   "image_count": len(kept), "excluded_count": len(excluded)}


def make_snapshot(review_root, snapshot, kept, excluded, review_status, audit_meta):
    if snapshot.exists():
        raise FileExistsError(f"Choose a new snapshot directory: {snapshot}")
    (snapshot / "images").mkdir(parents=True)
    (snapshot / "reviewed_labels").mkdir(parents=True)
    frozen_rows, frozen_reviews = [], {}
    for item in kept:
        row, review = item["row"], item["review"]
        image_rel = f"images/{row['id']}.jpg"
        label_rel = f"reviewed_labels/{row['id']}.txt"
        copy_checked(review_root / row["image"], snapshot / image_rel, item["image_sha256"])
        copy_checked(review_root / "reviewed_labels" / f"{row['id']}.txt",
                     snapshot / label_rel, item["label_sha256"])
        frozen_rows.append({**row, "image": image_rel, "proposal_label": label_rel,
                            "label": label_rel, "proposal_count": review["box_count"],
                            "review_status": "reviewed", "annotation_locked": True,
                            "source_review_dataset": str(review_root),
                            "source_review_image": row["image"],
                            "source_review_label": f"reviewed_labels/{row['id']}.txt",
                            "image_sha256": item["image_sha256"],
                            "label_sha256": item["label_sha256"]})
        frozen_reviews[row["id"]] = review
    (snapshot / "samples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in frozen_rows), encoding="utf-8")
    write_json(snapshot / "reviews.json", frozen_reviews)
    write_json(snapshot / "summary.json", {
        "status": "completed", "images": len(frozen_rows),
        "reviewed_images": len(frozen_rows), "excluded_images": len(excluded),
        "by_split": {"train": len(frozen_rows)},
        "split_selection_status": "frozen_train_candidates",
        "require_complete_review": True, "test_set_sealed": False,
        "source_review_dataset": str(review_root), "training_started": False,
        "review_status": review_status,
        "audit_note": "Supplement-only freeze; not an independent validation set."})
    write_json(snapshot / "excluded_samples.json", excluded)
    return frozen_rows


def merge_export(base_root, snapshot, export_root, base_rows, new_rows, audit_meta):
    if export_root.exists():
        raise FileExistsError(f"Choose a new export directory: {export_root}")
    # IDs and image contents must be disjoint before any copy occurs.
    base_ids = {row["id"] for row in base_rows}
    new_ids = {row["id"] for row in new_rows}
    if base_ids & new_ids:
        raise ValueError("Supplemental IDs collide with v11 IDs")
    base_hashes = {}
    for row in base_rows:
        source = base_root / row["image"]
        if not source.is_file():
            raise FileNotFoundError(source)
        base_hashes.setdefault(sha256(source), []).append(row["id"])
    if any(sha256(snapshot / row["image"]) in base_hashes for row in new_rows):
        raise ValueError("Supplemental image is already present in the v11 export")
    for split in ("train", "val", "test"):
        (export_root / "images" / split).mkdir(parents=True)
        (export_root / "labels" / split).mkdir(parents=True)
    fingerprints = []
    merged = []
    for source_root, source_rows in ((base_root, base_rows), (snapshot, new_rows)):
        for row in source_rows:
            split = row["split"]
            source_image = source_root / row["image"]
            if source_root == base_root:
                source_label = source_root / row["label"]
            else:
                source_label = source_root / row["label"]
            if not source_label.is_file():
                # The v11 manifest always has label; the fallback keeps this
                # helper explicit if an older frozen row only carries proposal_label.
                source_label = source_root / row["proposal_label"]
            target_image = export_root / "images" / split / f"{row['id']}.jpg"
            target_label = export_root / "labels" / split / f"{row['id']}.txt"
            image_hash = copy_checked(source_image, target_image)
            label_hash = copy_checked(source_label, target_label)
            merged.append({**row, "source_sample_image": row["image"],
                           "image": f"images/{split}/{row['id']}.jpg",
                           "label": f"labels/{split}/{row['id']}.txt",
                           "proposal_label": f"labels/{split}/{row['id']}.txt",
                           "review_status": "reviewed", "annotation_locked": True,
                           "image_sha256": image_hash, "label_sha256": label_hash,
                           "review": row.get("review", {})})
            fingerprints.append({"id": row["id"], "camera_id": row["camera_id"],
                                 "split": split, "image_sha256": image_hash,
                                 "label_sha256": label_hash})
    counts = Counter(row["split"] for row in merged)
    image_hash_by_split = defaultdict(set)
    image_hash_all = set()
    for row in merged:
        image_hash = sha256(export_root / row["image"])
        image_hash_all.add(image_hash)
        image_hash_by_split[row["split"]].add(image_hash)
    positive = Counter()
    boxes = Counter()
    for row in merged:
        label = export_root / row["label"]
        b = read_labels(label)
        positive[row["split"]] += bool(b)
        boxes[row["split"]] += len(b)
    if any(positive[s] == 0 for s in ("train", "val", "test")):
        raise ValueError("Every split must retain at least one positive image")
    data = {"path": str(export_root), "train": "images/train", "val": "images/val",
            "test": "images/test", "names": ["person"]}
    import yaml
    (export_root / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (export_root / "exported_samples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged), encoding="utf-8")
    summary = {"status": "completed", "reviewed_images_exported": len(merged),
               "unique_source_images": len(image_hash_all),
               "training_unique_images": len(image_hash_by_split["train"]),
               "unique_images_by_split": {split: len(image_hash_by_split[split])
                                          for split in ("train", "val", "test")},
               "by_split": dict(counts), "positive_images_by_split": dict(positive),
               "boxes_by_split": dict(boxes), "v11_base_images": len(base_rows),
               "supplemental_reviewed_images": len(new_rows),
               "supplemental_excluded_images": audit_meta["excluded_count"],
               "test_set_sealed": True, "test_split_evaluated": False,
               "training_started": False,
               "data_policy": "v11 train plus reviewed v12 additions; v11 val/test copied unchanged"}
    write_json(export_root / "summary.json", summary)
    return merged, fingerprints, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()
    review_root, base_root, snapshot, export_root, audit_root = (
        p.resolve() for p in (args.review, args.base, args.snapshot, args.export, args.audit_out))
    if any(a == b or a in b.parents or b in a.parents
           for i, a in enumerate((review_root, base_root, snapshot, export_root, audit_root))
           for b in (review_root, base_root, snapshot, export_root, audit_root)[i + 1:]):
        raise ValueError("Review, base and output directories must not overlap")
    if any(p.exists() for p in (snapshot, export_root, audit_root)):
        raise FileExistsError("Choose new snapshot, export and audit directories")
    base_rows = rows(base_root / "exported_samples.jsonl")
    if len({r["id"] for r in base_rows}) != len(base_rows):
        raise ValueError("v11 manifest contains duplicate IDs")
    store, kept, excluded, audit_meta = audit_review(review_root)
    new_rows = make_snapshot(review_root, snapshot, kept, excluded, audit_meta["status"], audit_meta)
    merged, fingerprints, export_summary = merge_export(base_root, snapshot, export_root,
                                                         base_rows, new_rows, audit_meta)
    audit_root.mkdir(parents=True)
    fp_path = audit_root / "fingerprints.json"
    write_json(fp_path, fingerprints)
    base_test = {r["id"]: r for r in base_rows if r["split"] == "test"}
    exported_by_id = {r["id"]: r for r in merged}
    test_unchanged = True
    for sid, row in base_test.items():
        old_image, new_image = base_root / row["image"], export_root / exported_by_id[sid]["image"]
        old_label, new_label = base_root / row["label"], export_root / exported_by_id[sid]["label"]
        test_unchanged &= sha256(old_image) == sha256(new_image) and sha256(old_label) == sha256(new_label)
    if not test_unchanged:
        raise RuntimeError("v11 test image/label fingerprints changed")
    audit = {
        "status": "passed", "audited_at": datetime.now().astimezone().isoformat(),
        "review_dataset": str(review_root), "frozen_snapshot": str(snapshot),
        "base_export": str(base_root), "exported_dataset": str(export_root),
        "source_review_status": audit_meta["status"],
        "review_status": audit_meta["status"],
        "reviewed_additions": len(kept), "excluded_additions": len(excluded),
        "reviewed_boxes_by_camera": audit_meta["boxes_by_camera"],
        "merged_summary": export_summary, "test_membership_and_fingerprints_unchanged": test_unchanged,
        "test_split_evaluated": False, "training_started": False,
        "fingerprints_file": str(fp_path), "fingerprints_sha256": sha256(fp_path),
        "input_review_metadata": {name: sha256(review_root / name) for name in
                                   ("samples.jsonl", "reviews.json", "summary.json", "sampling_plan.json")},
        "limitations": [
            "Labels were manually reviewed, but semantic correctness still requires a second human spot check.",
            "The added images are from the same date and source recordings; this is a training augmentation, not independent field validation.",
            "v11 validation/test remain unchanged, but new training data share same-day temporal operation groups with existing training scenes.",
            "Occlusion, edge truncation and overhead tags are selection hints; this review UI did not collect per-person occlusion percentages.",
        ]
    }
    write_json(audit_root / "annotation_audit.json", audit)
    write_json(export_root / "annotation_audit.json", audit)
    print(json.dumps({"status": audit["status"], "reviewed_additions": len(kept),
                      "excluded_additions": len(excluded), "merged": export_summary,
                      "test_unchanged": test_unchanged, "export": str(export_root)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
