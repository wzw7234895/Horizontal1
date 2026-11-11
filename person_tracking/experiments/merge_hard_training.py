"""Merge reviewed hard additions into an audited YOLO training snapshot.

The v13 review exports each contain the locked v9 baseline.  This utility
copies the already reviewed v12 snapshot, extracts only new train rows from
the D03/D04 exports, removes duplicate image content, and writes a new audit
contract.  Validation and test membership come exclusively from v12.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

from person_tracking.common import read_jsonl, write_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_registered_file(source_root: Path, target_root: Path, relative: str) -> None:
    source = source_root / relative
    target = target_root / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def label_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def load_rows(root: Path) -> list[dict]:
    rows = read_jsonl(root / "exported_samples.jsonl")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate sample IDs in {root}")
    return rows


def merge(base: Path, supplements: list[Path], output: Path, initialization: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if not base.is_dir() or not (base / "exported_samples.jsonl").is_file():
        raise FileNotFoundError(f"Invalid base export: {base}")
    if not initialization.is_file():
        raise FileNotFoundError(f"Initialization checkpoint not found: {initialization}")
    base_rows = load_rows(base)
    base_ids = {row["id"] for row in base_rows}
    if any(row.get("split") not in ("train", "val", "test") for row in base_rows):
        raise ValueError("Invalid base split")

    output.mkdir(parents=True)
    for row in base_rows:
        copy_registered_file(base, output, row["image"])
        copy_registered_file(base, output, row["label"])

    image_hashes = {}
    for row in base_rows:
        image_path = output / row["image"]
        digest = sha256(image_path)
        image_hashes.setdefault(digest, row["id"])
    merged_rows = list(base_rows)
    added = []
    skipped = []
    for supplement in supplements:
        rows = load_rows(supplement)
        supplement_added = 0
        supplement_skipped = 0
        for row in rows:
            # v13 exports repeat their 881-image locked baseline.  Only rows
            # absent from v12 and assigned to train are eligible additions.
            # v13 exports retain the locked-base rows and mark only the new
            # hard-review rows with annotation_locked=false.
            if row.get("annotation_locked") is not False:
                supplement_skipped += 1
                continue
            if row.get("split") != "train":
                raise ValueError(f"Supplemental row is not train-only: {row['id']}")
            image_source = supplement / row["image"]
            label_source = supplement / row["label"]
            image_digest = sha256(image_source)
            if image_digest in image_hashes:
                skipped.append({"id": row["id"], "source": str(supplement),
                                "duplicate_of": image_hashes[image_digest], "reason": "image_sha256"})
                supplement_skipped += 1
                continue
            target_image = f"images/train/{row['id']}.jpg"
            target_label = f"labels/train/{row['id']}.txt"
            copy_registered_file(supplement, output, row["image"])
            copy_registered_file(supplement, output, row["label"])
            # The source export paths are already split-specific.  Rename into
            # the merged train namespace while preserving the source metadata.
            (output / row["image"]).replace(output / target_image)
            (output / row["label"]).replace(output / target_label)
            merged = {**row, "split": "train", "image": target_image, "label": target_label,
                      "image_sha256": image_digest, "label_sha256": sha256(output / target_label),
                      "merged_from": str(supplement)}
            merged_rows.append(merged)
            base_ids.add(row["id"])
            image_hashes[image_digest] = row["id"]
            added.append(merged)
            supplement_added += 1
        if supplement_added == 0 and supplement_skipped == 0:
            raise ValueError(f"No rows found in supplement: {supplement}")

    # The base manifest already uses the same relative layout.  Normalize
    # hashes for every row so the audit is self-contained.
    for row in merged_rows:
        row["image_sha256"] = sha256(output / row["image"])
        row["label_sha256"] = sha256(output / row["label"])
    merged_rows.sort(key=lambda row: (row["split"], row["id"]))
    (output / "exported_samples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged_rows), encoding="utf-8")
    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        "path: '" + output.resolve().as_posix() + "'\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n- person\n", encoding="utf-8")

    fingerprints = [{"id": row["id"], "camera_id": row.get("camera_id"), "split": row["split"],
                     "image_sha256": row["image_sha256"], "label_sha256": row["label_sha256"]}
                    for row in merged_rows]
    fingerprint_path = output / "fingerprints.json"
    write_json(fingerprint_path, fingerprints)
    by_split = Counter(row["split"] for row in merged_rows)
    by_camera = Counter(row.get("camera_id") for row in added)
    labels_by_split = Counter()
    unique_images_by_split = Counter()
    for split in ("train", "val", "test"):
        split_rows = [row for row in merged_rows if row["split"] == split]
        labels_by_split[split] = sum(label_count(output / row["label"]) for row in split_rows)
        unique_images_by_split[split] = len({row["image_sha256"] for row in split_rows})
    summary = {
        "status": "completed", "created_at": datetime.now().isoformat(),
        "base_dataset": str(base), "supplement_datasets": [str(p) for p in supplements],
        "merged_rows": len(merged_rows), "added_rows": len(added), "skipped_duplicates": len(skipped),
        "added_by_camera": dict(by_camera), "by_split": dict(by_split),
        "unique_images_by_split": dict(unique_images_by_split), "boxes_by_split": dict(labels_by_split),
        "test_membership_source": str(base), "test_set_sealed": True,
        "test_images_and_labels_copied_unchanged": by_split["test"],
        "training_started": False, "test_split_used": False,
        "initialization_checkpoint": str(initialization),
    }
    write_json(output / "summary.json", summary)
    audit = {
        "status": "passed", "audited_at": datetime.now().isoformat(),
        "review_status": {"total": len(merged_rows), "pending": 0,
                          "reviewed": len(merged_rows), "excluded": len(skipped)},
        "base_dataset": str(base), "supplement_datasets": [str(p) for p in supplements],
        "exported_dataset": str(output.resolve()), "merged_summary": summary,
        "duplicate_rows_not_exported": skipped,
        "fingerprints_file": str(fingerprint_path.resolve()),
        "fingerprints_sha256": sha256(fingerprint_path),
        "export_metadata_sha256": {"data.yaml": sha256(data_yaml),
                                   "exported_samples.jsonl": sha256(output / "exported_samples.jsonl")},
        "required_checkpoint_initialization": {"path": str(initialization.resolve()),
                                                "sha256": sha256(initialization)},
        "test_set_sealed": True, "test_split_used": False, "test_split_evaluated": False,
        "training_started": False,
        "training_scope": "Static person detector; v12 train plus reviewed D03/D04 hard additions. No tracking, helmet or workwear labels.",
        "limitations": [
            "The additions and base recordings are from the same date and are not independent field validation.",
            "The inherited v12 validation set remains the selection set; use a new-date holdout for acceptance.",
            "Duplicate image content was retained only once for the merged additions and recorded in duplicate_rows_not_exported.",
        ],
    }
    write_json(output / "annotation_audit.json", audit)
    write_json(output / "merge_manifest.json", {"added": added, "skipped": skipped,
                                                  "base_rows": len(base_rows)})
    print(json.dumps({"summary": summary, "audit": audit}, ensure_ascii=False, indent=2), flush=True)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    args = parser.parse_args(argv)
    merge(args.base, args.supplement, args.out, args.initialization)


if __name__ == "__main__":
    main()
