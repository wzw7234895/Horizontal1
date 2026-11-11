"""Freeze corrected annotations, reselect temporal groups, audit, and export a new version."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
from itertools import product
import json
from pathlib import Path
import shutil

from person_tracking.common import read_jsonl, write_json
from person_tracking.dataset import read_image, read_labels
from person_tracking.review import ReviewStore, export_dataset


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_grouped_splits(rows, seed=20261106):
    """Optimize 70/15/15 image counts with whole groups and camera coverage.

    Uses IDs, groups, camera IDs and image counts only; never labels or predictions.
    A seed-dependent hash resolves ties, without trying multiple seeds against metrics.
    """
    groups = sorted({r["split_group"] for r in rows})
    cameras = {r["camera_id"] for r in rows}
    totals = Counter(r["split_group"] for r in rows)
    coverage = {g: {r["camera_id"] for r in rows if r["split_group"] == g} for g in groups}
    roles = ("train", "val", "test")
    best_key = best_assignment = None
    for assignment in product(roles, repeat=len(groups)):
        count = Counter()
        seen = {role: set() for role in roles}
        for group, role in zip(groups, assignment):
            count[role] += totals[group]
            seen[role].update(coverage[group])
        if any(seen[role] != cameras for role in roles):
            continue
        n = len(rows)
        # Integer score avoids floating tie differences for a 70/15/15 target.
        error = sum((20 * count[role] - target * n) ** 2 for role, target in zip(roles, (14, 3, 3)))
        tie = hashlib.sha256(f"{seed}|{'|'.join(assignment)}".encode()).hexdigest()
        key = error, tie
        if best_key is None or key < best_key:
            best_key, best_assignment = key, assignment
    if best_assignment is None:
        raise ValueError("Cannot form three whole-group splits each covering all cameras")
    return dict(zip(groups, best_assignment))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()
    source, snapshot, exported, output = [p.resolve() for p in (args.source, args.snapshot, args.export, args.out)]
    if len({source, snapshot, exported, output}) != 4 or any(p.exists() for p in (snapshot, exported, output)):
        raise FileExistsError("Choose separate new snapshot, export and audit destinations")
    store = ReviewStore(source)
    source_status = store.status()
    if source_status["pending"]:
        raise ValueError("Pending manual annotations remain")
    review_sha, manifest_sha = sha256(store.review_path), sha256(source / "samples.jsonl")
    selected = [r for r in store.rows if store.reviews[r["id"]]["status"] == "reviewed"]
    mapping = select_grouped_splits(selected, args.seed)
    print("Selected whole-group split", json.dumps(mapping), flush=True)
    initial = json.loads((source / "provenance/test_reopen/initial_fingerprints.json").read_text(encoding="utf8"))
    initial_hash = {r["path"]: r["sha256"] for r in initial}
    changes, fingerprints, image_ids = [], [], defaultdict(list)
    by_split, by_camera_split = defaultdict(Counter), defaultdict(Counter)
    errors, short_boxes = [], 0
    for i, row in enumerate(selected):
        image_path, label_path = source / row["image"], store.label_path(row)
        review, boxes = store.reviews[row["id"]], read_labels(label_path)
        if len(boxes) != review["box_count"] or bool(review["confirmed_empty"]) != (not boxes):
            raise ValueError(f"Review/label disagreement: {row['id']}")
        image = read_image(image_path)
        h, w = image.shape[:2]
        if (w, h) != (row["width"], row["height"]):
            raise ValueError(f"Image dimensions disagree: {row['id']}")
        ih, lh = sha256(image_path), sha256(label_path)
        image_ids[ih].append(row["id"])
        split = mapping[row["split_group"]]
        count = {"images": 1, "positive_images": int(bool(boxes)), "negative_images": int(not boxes), "boxes": len(boxes)}
        by_split[split].update(count)
        by_camera_split[f"{row['camera_id']}_{split}"].update(count)
        relative_label = label_path.relative_to(source).as_posix()
        if initial_hash.get(relative_label) != lh:
            changes.append({"id": row["id"], "old_split": row["split"], "new_split": split})
        if initial_hash.get(row["image"]) != ih:
            errors.append(f"Source image changed: {row['id']}")
        short_boxes += sum(min(b[2] * w, b[3] * h) < 3 for b in boxes)
        fingerprints.append({"id": row["id"], "camera_id": row["camera_id"], "split": split,
            "split_group": row["split_group"], "image": row["image"], "label": relative_label,
            "image_sha256": ih, "label_sha256": lh})
        if (i + 1) % 100 == 0:
            print(f"Audited {i + 1}/{len(selected)} corrected images", flush=True)
    duplicate_ids = [ids for ids in image_ids.values() if len(ids) > 1]
    if errors or duplicate_ids or any(by_split[s]["positive_images"] == 0 for s in ("train", "val", "test")):
        raise ValueError(f"Invalid image/audit/split: {errors}, duplicates={duplicate_ids}")
    if sha256(store.review_path) != review_sha or sha256(source / "samples.jsonl") != manifest_sha:
        raise RuntimeError("Source annotations changed during audit; repeat with new destinations")
    snapshot.mkdir(parents=True)
    output.mkdir(parents=True)
    for name in ("reviews.json",):
        shutil.copy2(source / name, snapshot / name)
    for directory in ("images", "reviewed_labels"):
        (snapshot / directory).mkdir()
    for item in fingerprints:
        shutil.copy2(source / item["image"], snapshot / item["image"])
        shutil.copy2(source / item["label"], snapshot / item["label"])
    now = datetime.now().isoformat()
    frozen_rows = []
    for row in selected:
        new_split = mapping[row["split_group"]]
        frozen_rows.append({**row, "previous_split": row["split"], "split": new_split,
            "annotation_locked": new_split == "test", "test_membership_status": "sealed_v3" if new_split == "test" else "not_test_v3",
            "review_status": "reviewed", "source_correction_dataset": str(source)})
    (snapshot / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in frozen_rows), encoding="utf8")
    summary = {"status": "completed", "images": len(selected), "by_split": {k: v["images"] for k, v in by_split.items()},
        "reviewed_images": len(selected), "pending_images": 0, "split_selection_status": "frozen_v3",
        "test_set_sealed": True, "frozen_at": now, "source_dataset": str(source), "training_started": False}
    write_json(snapshot / "summary.json", summary)
    split_record = {"status": "frozen_v3", "seed": args.seed, "selected_at": now, "group_splits": mapping,
        "target_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "selection_inputs": "Whole temporal groups, sample counts and camera coverage only; no labels, scores, metrics or predictions.",
        "tie_policy": "Seed-dependent SHA-256 of assignment; seed fixed before training.",
        "source_dataset": str(source), "fresh_public_pretrained_initialization_required": True,
        "test_reused_for_selection": False,
        "limitations": ["All samples come from one day; exposure synchronization is unverified.",
            "Part of this sample pool was model-assisted during earlier sampling. Reselection does not create a new independent collection.",
            "Historical finetuned models may have trained on the newly selected holdout groups; do not compare them as unseen-test baselines."]}
    write_json(snapshot / "split_reselection.json", split_record)
    export_summary = export_dataset(snapshot, exported)
    exported_rows = {r["id"]: r for r in read_jsonl(exported / "exported_samples.jsonl")}
    for item in fingerprints:
        copied = exported_rows[item["id"]]
        for kind in ("image", "label"):
            if sha256(exported / copied[kind]) != item[f"{kind}_sha256"]:
                raise ValueError(f"Export mismatch: {item['id']}")
        if sha256(source / item["label"]) != item["label_sha256"]:
            raise RuntimeError(f"Source annotation changed during copy: {item['id']}")
    if sha256(store.review_path) != review_sha:
        raise RuntimeError("Source reviews changed during export")
    write_json(output / "fingerprints.json", fingerprints)
    for name in ("reviews.json", "samples.jsonl"):
        shutil.copy2(source / name, output / ("source_" + name))
    write_json(output / "split_reselection.json", split_record)
    audit = {"status": "passed", "audited_at": now, "source_dataset": str(source), "frozen_review_snapshot": str(snapshot),
        "exported_dataset": str(exported), "review_status": ReviewStore(snapshot).status(),
        "source_reviews_sha256": review_sha, "by_split": dict(by_split), "by_camera_split": dict(by_camera_split),
        "corrected_labels": changes, "corrected_label_count": len(changes), "all_images_decoded": len(selected),
        "unique_image_sha256": len(image_ids), "temporal_group_cross_split_conflicts": 0,
        "test_set_sealed": True, "test_split_evaluated": False, "training_started": False,
        "boxes_shorter_than_3_pixels": short_boxes, "fingerprints_file": str(output / "fingerprints.json"),
        "fingerprints_sha256": sha256(output / "fingerprints.json"), "export_summary": export_summary,
        "limitations": ["Structural validity and manual completion do not certify semantic accuracy.", *split_record["limitations"]]}
    write_json(output / "audit.json", audit)
    shutil.copy2(output / "audit.json", exported / "annotation_audit.json")
    print(json.dumps({"status": "passed", "corrected_labels": len(changes), "by_split": dict(by_split),
        "by_camera_split": dict(by_camera_split), "snapshot": str(snapshot), "exported": str(exported)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
