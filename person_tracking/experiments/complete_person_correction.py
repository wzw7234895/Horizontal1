"""Freeze completed label corrections while preserving historical split membership."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import shutil

from person_tracking.common import PACKAGE_DIR, read_jsonl, write_json
from person_tracking.dataset import read_image, read_labels
from person_tracking.experiments.complete_person_v5 import fingerprint_export, sha256
from person_tracking.review import ReviewStore, export_dataset


def complete_correction(dataset, snapshot, audit_output, export_output, initialization=None):
    review_root, snapshot, audit_output, export_output = (
        Path(path).resolve() for path in (dataset, snapshot, audit_output, export_output))
    paths = (review_root, snapshot, audit_output, export_output)
    if any(a == b or a in b.parents or b in a.parents
           for index, a in enumerate(paths) for b in paths[index + 1:]):
        raise ValueError("Source, snapshot, audit and export must be separate, non-overlapping directories")
    if any(path.exists() for path in paths[1:]):
        raise FileExistsError("Choose new snapshot, audit and export directories")
    store = ReviewStore(review_root)
    status = store.status()
    if status["pending"]:
        raise ValueError(f"There are still {status['pending']} pending images; finish manual correction first")
    if status["reviewed"] + status["excluded"] != status["total"]:
        raise ValueError("Every image must be reviewed or explicitly excluded")
    record_path = review_root / "annotation_correction.json"
    record_hash = sha256(record_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("review_scope") != "previous_positive_images_only":
        raise ValueError("Expected positive-label correction with inherited confirmed empty images")
    source = ReviewStore(record["source_dataset"])
    if any(source.root == path or source.root in path.parents or path in source.root.parents
           for path in paths):
        raise ValueError("Correction and outputs must not overlap the original source dataset")
    for name, expected in record["source_metadata_sha256"].items():
        if sha256(source.root / name) != expected:
            raise ValueError(f"The original source changed: {name}")
    initial = json.loads((Path(record["backup"]) / "original_image_label_fingerprints.json").read_text(encoding="utf-8"))
    initial_hashes = {item["path"]: item["sha256"] for item in initial}
    rows_by_id = {row["id"]: row for row in store.rows}
    source_by_id = {row["id"]: row for row in source.rows}
    if set(rows_by_id) != set(source_by_id):
        raise ValueError("Correction sample membership differs from the original source")
    source_hashes = {name: sha256(review_root / name)
                     for name in ("samples.jsonl", "reviews.json", "summary.json")}
    initialization = Path(initialization or PACKAGE_DIR / "weights/yolo11n.pt").resolve()
    if not initialization.is_file():
        raise FileNotFoundError(f"Clean initialization checkpoint not found: {initialization}")

    fingerprints, changed, excluded, frozen_rows = [], [], [], []
    groups, image_splits = {}, defaultdict(set)
    counts, camera_counts = defaultdict(Counter), defaultdict(Counter)
    inherited_empty = 0
    identity_fields = ("camera_id", "split", "split_group", "image", "source", "source_frame",
                       "recording_time_estimate", "width", "height")
    for row in store.rows:
        sample_id = row["id"]
        original = source_by_id[sample_id]
        if any(row.get(key) != original.get(key) for key in identity_fields):
            raise ValueError(f"Sample identity or historical split changed: {sample_id}")
        split = row["split"]
        if split not in ("train", "val", "test") or groups.setdefault(row["split_group"], split) != split:
            raise ValueError(f"Temporal group spans multiple splits: {row['split_group']}")
        image = (review_root / row["image"]).resolve()
        original_image = (source.root / original["image"]).resolve()
        label = store.label_path(row)
        original_label = source.label_path(original)
        if not image.is_relative_to(review_root) or not original_image.is_relative_to(source.root):
            raise ValueError(f"Image path is outside its dataset: {sample_id}")
        image_hash, label_hash = sha256(image), sha256(label)
        original_label_hash = sha256(original_label)
        expected_label = initial_hashes.get(f"reviewed_labels/{sample_id}.txt")
        if sha256(original_image) != initial_hashes.get(original["image"]) or original_label_hash != expected_label:
            raise ValueError(f"The original source image or label changed: {sample_id}")
        if image_hash != initial_hashes.get(row["image"]):
            raise ValueError(f"Image changed during label-only correction: {sample_id}")
        review = store.reviews[sample_id]
        original_review = source.reviews[sample_id]
        old_empty = (original_review["status"] == "reviewed" and
                     original_review.get("confirmed_empty") and original_review["box_count"] == 0)
        if old_empty:
            if (review["status"] != "reviewed" or not review.get("confirmed_empty")
                    or review["box_count"] != 0 or label_hash != original_label_hash
                    or review.get("reviewed_at") != original_review.get("reviewed_at")):
                raise ValueError(f"Inherited confirmed empty sample changed: {sample_id}")
            inherited_empty += 1
        if label_hash != original_label_hash:
            changed.append({"id": sample_id, "split": split, "status": review["status"],
                            "old_label_sha256": original_label_hash, "label_sha256": label_hash})
        if review["status"] == "excluded":
            excluded.append({**row, "exclusion_review": review})
            continue
        boxes = read_labels(label)
        if len(boxes) != review["box_count"] or bool(review.get("confirmed_empty")) != (not boxes):
            raise ValueError(f"Review and label disagree: {sample_id}")
        decoded = read_image(image)
        if (decoded.shape[1], decoded.shape[0]) != (row["width"], row["height"]):
            raise ValueError(f"Image dimensions disagree: {sample_id}")
        image_splits[image_hash].add(split)
        delta = {"images": 1, "positive_images": int(bool(boxes)),
                 "negative_images": int(not boxes), "boxes": len(boxes)}
        counts[split].update(delta)
        camera_counts[f"{row['camera_id']}/{split}"].update(delta)
        fingerprints.append({"id": sample_id, "camera_id": row["camera_id"], "split": split,
                             "image_sha256": image_hash, "label_sha256": label_hash})
        frozen_rows.append({**row, "annotation_locked": True, "review_status": "reviewed",
                            "test_membership_status": "sealed_after_correction" if split == "test" else "not_test_after_correction",
                            "source_correction_dataset": str(review_root)})
    if inherited_empty != record["inherited_confirmed_empty_images"]:
        raise ValueError("Inherited confirmed empty sample count changed")
    if any(len(splits) > 1 for splits in image_splits.values()):
        raise ValueError("Identical image contents appear in multiple splits")
    if any(counts[split]["positive_images"] == 0 for split in ("train", "val", "test")):
        raise ValueError("Need at least one reviewed positive image in each preserved split")

    def unchanged():
        if sha256(record_path) != record_hash or any(
                sha256(review_root / name) != expected for name, expected in source_hashes.items()):
            raise RuntimeError("Correction changed during audit; repeat with new output directories")
        if any(sha256(source.root / name) != expected
               for name, expected in record["source_metadata_sha256"].items()):
            raise RuntimeError("Original source changed during audit")

    unchanged()
    now = datetime.now().astimezone().isoformat()
    audit_output.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    write_json(audit_output / "status.json", {"status": "building", "started_at": now})
    write_json(snapshot / "summary.json", {"status": "building", "training_started": False})
    for row, fingerprint in zip(frozen_rows, fingerprints):
        target_image, target_label = snapshot / row["image"], snapshot / "reviewed_labels" / f"{row['id']}.txt"
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(review_root / row["image"], target_image)
        shutil.copy2(store.label_path(row), target_label)
        if (sha256(target_image) != fingerprint["image_sha256"] or
                sha256(target_label) != fingerprint["label_sha256"]):
            raise RuntimeError(f"Correction changed during copy: {row['id']}")
    (snapshot / "samples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in frozen_rows), encoding="utf-8")
    write_json(snapshot / "reviews.json", {row["id"]: store.reviews[row["id"]] for row in frozen_rows})
    write_json(snapshot / "summary.json", {
        "status": "completed", "images": len(frozen_rows), "require_complete_review": True,
        "split_selection_status": "frozen_corrected_preserved_membership", "test_set_sealed": True,
        "by_split": {split: value["images"] for split, value in counts.items()},
        "source_dataset": str(review_root), "frozen_at": now, "training_started": False})
    export_summary = export_dataset(snapshot, export_output)
    exported_rows = {row["id"]: row for row in read_jsonl(export_output / "exported_samples.jsonl")}
    if fingerprint_export(export_output, exported_rows) != sorted(fingerprints, key=lambda row: row["id"]):
        raise RuntimeError("Exported image or label fingerprints disagree with the audited snapshot")
    unchanged()
    for name in source_hashes:
        shutil.copy2(review_root / name, audit_output / ("correction_" + name))
    fingerprints_path = audit_output / "fingerprints.json"
    write_json(fingerprints_path, sorted(fingerprints, key=lambda row: row["id"]))
    write_json(audit_output / "excluded_samples.json", excluded)
    audit = {
        "status": "passed", "audited_at": now, "source_review_dataset": str(review_root),
        "frozen_review_snapshot": str(snapshot), "exported_dataset": str(export_output),
        "review_status": ReviewStore(snapshot).status(), "export_summary": export_summary,
        "source_review_status": status, "source_metadata_sha256": source_hashes,
        "inherited_confirmed_empty_images": inherited_empty, "excluded_images": len(excluded),
        "changed_label_files": changed, "by_split": dict(counts), "by_camera_split": dict(camera_counts),
        "historical_split_membership_preserved": True, "temporal_group_cross_split_conflicts": 0,
        "test_membership_policy": "Preserve original temporal groups; omit explicitly excluded images",
        "test_split_evaluated": False, "training_started": False,
        "fresh_public_pretrained_initialization_required": True,
        "required_initialization": {"path": str(initialization), "sha256": sha256(initialization)},
        "fingerprints_file": str(fingerprints_path), "fingerprints_sha256": sha256(fingerprints_path),
        "limitations": [
            "Structural audit and completed manual reviews do not certify semantic label correctness.",
            "Existing confirmed empty images were inherited without new manual review, as requested.",
            "Exclusions and corrected val labels change the evaluation scope; re-evaluate baselines on this snapshot.",
            "The existing same-day data remains a development collection, not an independent field acceptance test."]}
    write_json(audit_output / "audit.json", audit)
    write_json(export_output / "annotation_audit.json", audit)
    write_json(audit_output / "status.json", {"status": "passed", "completed_at": now})
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--export-out", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, help="Clean public YOLO11n checkpoint to bind to the new audit")
    args = parser.parse_args()
    result = complete_correction(args.dataset, args.snapshot, args.audit_out, args.export_out, args.initialization)
    print(json.dumps({key: result[key] for key in (
        "status", "exported_dataset", "inherited_confirmed_empty_images", "excluded_images", "by_split")},
        ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
