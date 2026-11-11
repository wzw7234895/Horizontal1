"""Freeze reviewed D03 specialist samples and retire their old validation group."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import shutil

from person_tracking.common import read_jsonl, write_json
from person_tracking.dataset import read_image, read_labels
from person_tracking.experiments.complete_person_v5 import fingerprint_export, sha256
from person_tracking.review import ReviewStore, export_dataset


def checked_rows(store):
    status = store.status()
    if status["pending"] or status["reviewed"] + status["excluded"] != status["total"]:
        raise ValueError("Finish all manual annotations before freezing specialist data")
    result = []
    for row in store.rows:
        review = store.reviews[row["id"]]
        if review["status"] == "excluded":
            continue
        labels = read_labels(store.label_path(row))
        if len(labels) != review["box_count"] or bool(review.get("confirmed_empty")) != (not labels):
            raise ValueError(f"Review and label disagree: {row['id']}")
        image = (store.root / row["image"]).resolve()
        if not image.is_relative_to(store.root) or not image.is_file():
            raise ValueError(f"Invalid source image path: {row['id']}")
        if row.get("image_sha256") and sha256(image) != row["image_sha256"]:
            raise ValueError(f"Reviewed source image changed: {row['id']}")
        result.append((row, labels, review))
    return result


def complete(dataset, snapshot, audit_output, export_output, initialization):
    dataset, snapshot, audit_output, export_output, initialization = [
        Path(path).resolve() for path in (dataset, snapshot, audit_output, export_output, initialization)]
    if not initialization.is_file():
        raise FileNotFoundError(initialization)
    store = ReviewStore(dataset)
    plan = json.loads((dataset / "sampling_plan.json").read_text(encoding="utf-8"))
    base = ReviewStore(plan["source_dataset"])
    all_paths = (dataset, base.root, snapshot, audit_output, export_output)
    if any(a == b or a in b.parents or b in a.parents
           for i, a in enumerate(all_paths) for b in all_paths[i + 1:]):
        raise ValueError("Source, frozen snapshot, audit and export must be non-overlapping")
    if any(path.exists() for path in (snapshot, audit_output, export_output)):
        raise FileExistsError("Choose new snapshot, audit and export directories")
    for name, expected in plan["source_metadata_sha256"].items():
        if sha256(base.root / name) != expected:
            raise ValueError(f"Original frozen dataset metadata changed: {name}")
    source_hashes = {name: sha256(dataset / name) for name in ("samples.jsonl", "reviews.json", "summary.json", "sampling_plan.json")}
    candidate = checked_rows(store)
    original = checked_rows(base)
    if not candidate or not any(labels for _, labels, _ in candidate):
        raise ValueError("Specialist training needs manually confirmed people")
    if any(row["camera_id"] != "D03" or row["split"] != "train" for row, _, _ in candidate):
        raise ValueError("Specialist samples must be D03 training candidates")
    groups = {row["split_group"] for row, _, _ in candidate}
    retired = [row for row, _, _ in original if row["split_group"] in groups]
    if not retired or any(row["split"] != "val" for row in retired):
        raise ValueError("Candidate groups must match only the original validation partition")
    selected = [(store, row, labels, review, "train") for row, labels, review in candidate]
    selected += [(base, row, labels, review, row["split"]) for row, labels, review in original
                 if row["split"] in ("val", "test") and row["split_group"] not in groups]
    ids, temporal, image_membership = set(), {}, defaultdict(set)
    counts, fingerprints_before, test_hashes = defaultdict(Counter), {}, {}
    for origin, row, labels, _, split in selected:
        if row["id"] in ids:
            raise ValueError("Duplicate specialist sample ID")
        ids.add(row["id"])
        if temporal.setdefault(row["split_group"], split) != split:
            raise ValueError("A shared temporal group crosses train/val/test")
        image_path, label_path = origin.root / row["image"], origin.label_path(row)
        image_hash, label_hash = sha256(image_path), sha256(label_path)
        image_membership[image_hash].add(split)
        fingerprints_before[row["id"]] = (image_hash, label_hash)
        if split == "test":
            # Preserve and hash test files, without decoding them or evaluating a model.
            test_hashes[row["id"]] = (image_hash, label_hash)
        else:
            image = read_image(image_path)
            if image.shape[:2] != (row["height"], row["width"]):
                raise ValueError(f"Image dimensions disagree: {row['id']}")
        counts[split].update(images=1, positive_images=int(bool(labels)),
                             negative_images=int(not labels), boxes=len(labels))
    if any(len(splits) > 1 for splits in image_membership.values()):
        raise ValueError("Identical images cross partitions")
    if any(counts[split]["positive_images"] == 0 for split in ("train", "val", "test")):
        raise ValueError("Every retained partition needs a positive sample")
    expected_test_ids = {row["id"] for row, _, _ in original if row["split"] == "test"}
    if set(test_hashes) != expected_test_ids:
        raise ValueError("Original test membership changed")
    for folder in ("images", "reviewed_labels"):
        (snapshot / folder).mkdir(parents=True)
    frozen_rows, frozen_reviews = [], {}
    for origin, row, _, review, split in selected:
        image_path, label_path = origin.root / row["image"], origin.label_path(row)
        image_relative = f"images/{row['id']}.jpg"
        label_relative = f"reviewed_labels/{row['id']}.txt"
        shutil.copy2(image_path, snapshot / image_relative)
        shutil.copy2(label_path, snapshot / label_relative)
        expected_image, expected_label = fingerprints_before[row["id"]]
        if sha256(snapshot / image_relative) != expected_image or sha256(snapshot / label_relative) != expected_label:
            raise ValueError("Source changed while freezing labels")
        frozen_rows.append({**row, "split": split, "image": image_relative,
            "proposal_label": label_relative, "proposal_count": review["box_count"],
            "review_status": "reviewed", "annotation_locked": True,
            "split_assignment_status": "resolved_specialist_train_or_retained_validation_test",
            "source_review_dataset": str(origin.root), "source_review_image": row["image"]})
        frozen_reviews[row["id"]] = review
    with (snapshot / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for row in frozen_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(snapshot / "reviews.json", frozen_reviews)
    write_json(snapshot / "summary.json", {
        "status": "completed", "images": len(frozen_rows), "require_complete_review": True,
        "split_selection_status": "resolved_specialist_old_validation_group_retired",
        "test_set_sealed": True, "by_split": {key: value["images"] for key, value in counts.items()},
        "source_dataset": str(dataset), "base_dataset": str(base.root), "training_started": False})
    for name, before in source_hashes.items():
        if sha256(dataset / name) != before:
            raise ValueError("Manual annotations changed during freeze")
    summary = export_dataset(snapshot, export_output)
    exported = {row["id"]: row for row in read_jsonl(export_output / "exported_samples.jsonl")}
    fingerprints = fingerprint_export(export_output, exported)
    for record in fingerprints:
        if (record["image_sha256"], record["label_sha256"]) != fingerprints_before[record["id"]]:
            raise ValueError("Export differs from the frozen source")
    write_json(audit_output / "fingerprints.json", fingerprints)
    write_json(audit_output / "retired_validation_samples.json", retired)
    audit = {
        "status": "passed", "audited_at": datetime.now().astimezone().isoformat(),
        "source_review_dataset": str(dataset), "base_frozen_dataset": str(base.root),
        "frozen_review_snapshot": str(snapshot), "exported_dataset": str(export_output),
        "review_status": ReviewStore(snapshot).status(), "export_summary": summary,
        "by_split": {key: dict(value) for key, value in counts.items()},
        "specialist_training_groups": sorted(groups), "retired_validation_images": len(retired),
        "retired_validation_samples": str(audit_output / "retired_validation_samples.json"),
        "original_train_images_used_for_new_gradients": 0,
        "temporal_group_cross_split_conflicts": 0, "test_membership_and_hashes_preserved": True,
        "source_review_metadata_sha256": source_hashes, "base_metadata_sha256": plan["source_metadata_sha256"],
        "required_checkpoint_initialization": {"path": str(initialization), "sha256": sha256(initialization)},
        "fresh_public_pretrained_initialization_required": False,
        "fingerprints_file": str(audit_output / "fingerprints.json"),
        "fingerprints_sha256": sha256(audit_output / "fingerprints.json"),
        "training_scope": "D03 specialist fine-tuning on newly reviewed person images only; other retained-camera val guards retention. One person class; no workwear classification or tracker learning.",
        "test_split_evaluated": False, "training_started": False,
        "limitations": [
            "The original shared validation group containing specialist samples was retired; scores cannot be compared with the old 115-image validation numbers.",
            "The remaining validation has no D03 images and cannot establish non-workwear person recall; use separate D03 development diagnostics and new footage for acceptance.",
            "Fine-tuning only the new images can forget old scenes; evaluate the previous training collection as a retention diagnostic, not independent validation.",
            "Structural checks and completed human labels do not independently certify semantic label correctness.",
        ],
    }
    write_json(audit_output / "annotation_audit.json", audit)
    write_json(export_output / "annotation_audit.json", audit)
    print(json.dumps({key: audit[key] for key in ("status", "exported_dataset", "by_split", "retired_validation_images", "limitations")}, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("dataset", "snapshot", "audit-out", "export-out", "initialization"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    complete(args.dataset, args.snapshot, args.audit_out, args.export_out, args.initialization)
