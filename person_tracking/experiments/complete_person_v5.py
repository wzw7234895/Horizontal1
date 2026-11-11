"""Audit completed hard-negative reviews and export a new fixed-split dataset."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

from person_tracking.common import read_jsonl, write_json
from person_tracking.dataset import read_image, read_labels
from person_tracking.review import ReviewStore, export_dataset


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_reference_dataset(provenance, base):
    reference = provenance.get("base_exported_dataset") or base.summary.get("split_reference_dataset")
    if not reference:
        raise ValueError("The locked base dataset does not identify its exported reference")
    return Path(reference).resolve()


def load_reference_rows(reference_root):
    reference_root = Path(reference_root)
    exported_manifest = reference_root / "exported_samples.jsonl"
    if exported_manifest.is_file():
        rows = read_jsonl(exported_manifest)
        return {row["id"]: {"image": row["image"], "label": row["label"],
                             "split": row["split"]} for row in rows}

    frozen_manifest = reference_root / "samples.jsonl"
    if frozen_manifest.is_file() and (reference_root / "reviewed_labels").is_dir():
        rows = read_jsonl(frozen_manifest)
        return {row["id"]: {"image": row["image"],
                             "label": f"reviewed_labels/{row['id']}.txt",
                             "split": row["split"]} for row in rows}
    raise FileNotFoundError(f"No supported frozen sample manifest in {reference_root}")


def fingerprint_export(export_root, exported_rows):
    export_root = Path(export_root)
    fingerprints = []
    for row in sorted(exported_rows.values(), key=lambda item: item["id"]):
        fingerprints.append({
            "id": row["id"], "camera_id": row["camera_id"], "split": row["split"],
            "image_sha256": sha256(export_root / row["image"]),
            "label_sha256": sha256(export_root / row["label"]),
        })
    return fingerprints


def complete_dataset(dataset, audit_output, export_output):
    review_root, audit_output, export_output = (Path(p).resolve()
                                                for p in (dataset, audit_output, export_output))
    if audit_output == export_output or audit_output.exists() or export_output.exists():
        raise FileExistsError("Choose separate new audit and export directories")
    store = ReviewStore(review_root)
    status = store.status()
    provenance = json.loads((review_root / "provenance.json").read_text(encoding="utf-8"))
    base = ReviewStore(provenance["base_dataset"])
    reference_root = resolve_reference_dataset(provenance, base)
    reference_rows = load_reference_rows(reference_root)
    rows_by_id = {row["id"]: row for row in store.rows}
    if status["pending"]:
        raise ValueError(f"There are still {status['pending']} pending images")
    if status["reviewed"] + status["excluded"] != status["total"]:
        raise ValueError("Every supplemental image must be reviewed or excluded")
    if sha256(base.root / "samples.jsonl") != provenance["base_samples_sha256"] or \
       sha256(base.review_path) != provenance["base_reviews_sha256"]:
        raise ValueError("The source review version changed after candidate preparation")
    base_ids = {row["id"] for row in base.rows}
    if not base_ids <= set(rows_by_id) or base_ids != set(reference_rows):
        raise ValueError("The locked base dataset membership changed")
    if set(base_ids) != {item["id"] for item in provenance["base_image_label_fingerprints"]}:
        raise ValueError("The base fingerprint manifest is incomplete")
    base_fingerprints = {item["id"]: item for item in provenance["base_image_label_fingerprints"]}
    group_splits = {row["split_group"]: row["split"] for row in base.rows}
    base_test_ids = {row["id"] for row in base.rows if row["split"] == "test"}
    if base_test_ids != set(provenance["base_test_ids"]):
        raise ValueError("The sealed base test membership changed")

    for row in base.rows:
        copied = rows_by_id[row["id"]]
        if not copied.get("annotation_locked") or copied["split"] != row["split"]:
            raise ValueError(f"A base annotation was unlocked or reassigned: {row['id']}")
        fingerprint = base_fingerprints[row["id"]]
        image_path = review_root / copied["image"]
        label_path = review_root / "reviewed_labels" / f"{row['id']}.txt"
        if sha256(image_path) != fingerprint["image_sha256"] or sha256(label_path) != fingerprint["label_sha256"]:
            raise ValueError(f"A locked base image or annotation changed: {row['id']}")
        if copied["split"] == "test":
            exported = reference_rows[row["id"]]
            if sha256(image_path) != sha256(reference_root / exported["image"]):
                raise ValueError(f"A sealed base test image changed: {row['id']}")
            if sha256(label_path) != sha256(reference_root / exported["label"]):
                raise ValueError(f"A sealed base test label changed: {row['id']}")

    additions = [row for row in store.rows if row["id"] not in base_ids]
    if not additions:
        raise ValueError("No supplemental samples were prepared")
    candidate_cameras = set(provenance.get("candidate_cameras", ["D01"]))
    image_hashes = {sha256(review_root / row["image"]) for row in base.rows}
    for row in additions:
        if row["camera_id"] not in candidate_cameras or row["split"] != "train" or row["split_group"] not in group_splits:
            raise ValueError(f"Supplemental sample is outside the fixed camera train groups: {row['id']}")
        if group_splits[row["split_group"]] != "train":
            raise ValueError(f"Supplemental sample crosses a held-out split boundary: {row['id']}")
        if row.get("annotation_locked"):
            raise ValueError(f"Supplemental sample is unexpectedly locked: {row['id']}")
        image_path = review_root / row["image"]
        digest = sha256(image_path)
        if digest in image_hashes:
            raise ValueError(f"Duplicate image in supplemental review: {row['id']}")
        image_hashes.add(digest)
        review = store.reviews.get(row["id"], {})
        if review.get("status") == "reviewed":
            labels = read_labels(store.label_path(row))
            if len(labels) != review.get("box_count") or bool(review.get("confirmed_empty")) != (not labels):
                raise ValueError(f"Review/label mismatch: {row['id']}")
        image = read_image(image_path)
        if image.shape[1] != row["width"] or image.shape[0] != row["height"]:
            raise ValueError(f"Image dimensions changed: {row['id']}")

    audit_output.mkdir(parents=True)
    export_summary = export_dataset(review_root, export_output)
    exported_rows = {row["id"]: row for row in read_jsonl(export_output / "exported_samples.jsonl")}
    if not base_test_ids <= set(exported_rows):
        raise ValueError("The exported dataset omitted a sealed test sample")
    for sample_id in base_test_ids:
        old, new = reference_rows[sample_id], exported_rows[sample_id]
        for kind in ("image", "label"):
            if sha256(reference_root / old[kind]) != sha256(export_output / new[kind]):
                raise ValueError(f"Exported sealed test {kind} changed: {sample_id}")
    exported_additions = [row for row in exported_rows.values() if row["id"] not in base_ids]
    if any(row["camera_id"] not in candidate_cameras or row["split"] != "train" for row in exported_additions):
        raise ValueError("An exported supplemental sample entered an unselected camera or held-out split")

    fingerprints = fingerprint_export(export_output, exported_rows)
    fingerprints_path = audit_output / "fingerprints.json"
    write_json(fingerprints_path, fingerprints)
    audit = {
        "status": "passed", "audited_at": datetime.now().isoformat(),
        "source_review_dataset": str(review_root), "base_dataset": str(base.root),
        "base_reference_dataset": str(reference_root), "exported_dataset": str(export_output),
        "review_status": status,
        "base_images_locked_and_unchanged": len(base_ids),
        "supplemental_reviewed": sum(store.reviews.get(row["id"], {}).get("status") == "reviewed"
                                      for row in additions),
        "supplemental_excluded": sum(store.reviews.get(row["id"], {}).get("status") == "excluded"
                                      for row in additions),
        "supplemental_exported_train_only": len(exported_additions),
        "supplemental_cameras": sorted(candidate_cameras),
        "test_images_and_labels_unchanged": len(base_test_ids),
        "fingerprints_file": str(fingerprints_path.resolve()),
        "fingerprints_sha256": sha256(fingerprints_path),
        "test_split_evaluated": False, "training_started": False,
        "export_summary": export_summary,
        "limitations": [
            "Automatic person boxes and candidate rankings were only review hints; exported labels come from human review.",
            "The same-day validation set remains for development and did not include the supplemental train images.",
            "A new training run and same-validation comparison are required before replacing the current checkpoint."
        ]
    }
    write_json(audit_output / "audit.json", audit)
    write_json(export_output / "annotation_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    args = parser.parse_args()
    complete_dataset(args.dataset, args.out, args.export)


if __name__ == "__main__":
    main()
