"""Build an audited old/new replay export with explicit within-train oversampling."""
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


def load_verified_export(root):
    root = Path(root).resolve()
    audit = json.loads((root / "annotation_audit.json").read_text(encoding="utf8"))
    if audit["status"] != "passed" or audit["review_status"]["pending"]:
        raise ValueError(f"Source annotation export is incomplete: {root}")
    if Path(audit["exported_dataset"]).resolve() != root:
        raise ValueError("Source audit belongs to another export")
    registry = Path(audit["fingerprints_file"])
    if sha256(registry) != audit["fingerprints_sha256"]:
        raise ValueError("Source fingerprint registry changed")
    rows = read_jsonl(root / "exported_samples.jsonl")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Source manifest contains duplicate IDs")
    expected = {row["id"]: row for row in json.loads(registry.read_text(encoding="utf8"))}
    if set(expected) != {row["id"] for row in rows}:
        raise ValueError("Source manifest and fingerprints differ")
    for row in rows:
        if row["split"] not in ("train", "val", "test") or row["split"] != expected[row["id"]]["split"]:
            raise ValueError("Source partition changed")
        for key in ("image", "label"):
            path = (root / row[key]).resolve()
            if not path.is_relative_to(root) or sha256(path) != expected[row["id"]][key + "_sha256"]:
                raise ValueError(f"Source file changed: {row['id']} {key}")
        review = row.get("review", {})
        if review.get("status") != "reviewed" or row.get("review_status") != "reviewed":
            raise ValueError("Only inherited human-reviewed samples are accepted")
    metadata = {name: sha256(root / name) for name in ("annotation_audit.json", "exported_samples.jsonl", "summary.json", "data.yaml")}
    return rows, expected, metadata


def frame_identity(row):
    if not row.get("source") or "source_frame" not in row:
        return None
    return str(Path(row["source"]).resolve()), row["camera_id"], row["source_frame"]


def exclusion_registry(datasets):
    image_hashes, frames, evidence = set(), set(), {}
    for dataset in datasets:
        root = Path(dataset).resolve()
        manifest = root / "samples.jsonl"
        if not manifest.is_file():
            manifest = root / "exported_samples.jsonl"
        if (root / "freeze.json").is_file():
            frozen = json.loads((root / "freeze.json").read_text(encoding="utf8"))
            for name, expected in frozen["files_sha256"].items():
                if sha256(root / name) != expected:
                    raise ValueError("Excluded evaluation metadata changed")
            evidence[str(root / "freeze.json")] = sha256(root / "freeze.json")
        evidence[str(manifest)] = sha256(manifest)
        for row in read_jsonl(manifest):
            image_hashes.add(row.get("image_sha256") or sha256(root / row["image"]))
            identity = frame_identity(row)
            if identity is not None:
                frames.add(identity)
    return image_hashes, frames, evidence


def prepare(base, specialist, output, audit_output, initialization, new_repeat=3, excluded_datasets=()):
    base, specialist, output, audit_output, initialization = [
        Path(p).resolve() for p in (base, specialist, output, audit_output, initialization)]
    roots = (base, specialist, output, audit_output)
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(roots) for b in roots[i+1:]):
        raise ValueError("Sources, audit, and export must be non-overlapping")
    if output.exists() or audit_output.exists():
        raise FileExistsError("Choose new output and audit directories")
    if type(new_repeat) is not int or new_repeat < 1:
        raise ValueError("new_repeat must be a positive integer")
    if not initialization.is_file():
        raise FileNotFoundError(initialization)
    old_rows, old_fingerprints, old_metadata = load_verified_export(base)
    new_rows, new_fingerprints, new_metadata = load_verified_export(specialist)
    excluded_images, excluded_frames, excluded_evidence = exclusion_registry(excluded_datasets)
    old_train = [row for row in old_rows if row["split"] == "train"]
    new_train = [row for row in new_rows if row["split"] == "train"]
    if not old_train or not new_train or any(row["camera_id"] != "D03" for row in new_train):
        raise ValueError("Need original replay train and reviewed D03 train samples")
    if {r["id"] for r in old_train} & {r["id"] for r in new_train}:
        raise ValueError("Old and new training IDs overlap")
    # Retain v10 val/test exactly. The retired old val group is never replayed.
    retained = [row for row in new_rows if row["split"] in ("val", "test")]
    original_test = {r["id"]: old_fingerprints[r["id"]] for r in old_rows if r["split"] == "test"}
    retained_test = {r["id"]: new_fingerprints[r["id"]] for r in retained if r["split"] == "test"}
    if original_test != retained_test:
        raise ValueError("Original test membership or hashes changed")
    selection = [(base, row, old_fingerprints[row["id"]], "old_replay", 0) for row in old_train]
    selection += [(specialist, row, new_fingerprints[row["id"]], "new_d03", copy_index)
                  for row in new_train for copy_index in range(new_repeat)]
    selection += [(specialist, row, new_fingerprints[row["id"]], "retained_" + row["split"], 0) for row in retained]
    ids, temporal, hash_splits, frame_splits = set(), {}, defaultdict(set), {}
    counts, origins, cameras = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    records = []
    for source, row, fingerprint, origin, copy_index in selection:
        sid = row["id"] if copy_index == 0 else row["id"] + f"__replay{copy_index+1:02d}"
        split = row["split"]
        if sid in ids:
            raise ValueError("Replay alias ID collision")
        ids.add(sid)
        if temporal.setdefault(row["split_group"], split) != split:
            raise ValueError("A shared temporal group crosses train/val/test")
        hash_splits[fingerprint["image_sha256"]].add(split)
        identity = frame_identity(row)
        if identity is not None and frame_splits.setdefault(identity, split) != split:
            raise ValueError("A source frame crosses train/val/test")
        if split == "train" and (row.get("evaluation_only") or fingerprint["image_sha256"] in excluded_images or identity in excluded_frames):
            raise ValueError("Evaluation-only or excluded development frames cannot enter training")
        review = row["review"]
        if split == "test":
            box_count = review["box_count"]
        else:
            labels = read_labels(source / row["label"])
            box_count = len(labels)
            if copy_index == 0:
                image = read_image(source / row["image"])
                if image.shape[:2] != (row["height"], row["width"]):
                    raise ValueError("Reviewed image dimensions changed")
            if box_count != review["box_count"]:
                raise ValueError("Inherited label and review counts differ")
        if bool(review.get("confirmed_empty")) != (box_count == 0):
            raise ValueError("Inherited empty annotation is not explicitly confirmed")
        summary = dict(images=1, positive_images=int(box_count > 0), negative_images=int(box_count == 0), boxes=box_count)
        counts[split].update(summary)
        origins[origin].update(summary)
        cameras[split + "/" + row["camera_id"]].update(summary)
        records.append((source, row, fingerprint, origin, copy_index, sid))
    if any(len(splits) != 1 for splits in hash_splits.values()):
        raise ValueError("Identical image content crosses partitions")
    for split in ("train", "val", "test"):
        if not counts[split]["positive_images"]:
            raise ValueError("Every retained partition needs a positive sample")
    for folder in ("images", "labels"):
        for split in ("train", "val", "test"):
            (output / folder / split).mkdir(parents=True)
    exported, allocation = [], []
    for source, row, fingerprint, origin, copy_index, sid in records:
        image_relative, label_relative = f"images/{row['split']}/{sid}.jpg", f"labels/{row['split']}/{sid}.txt"
        for key, relative in (("image", image_relative), ("label", label_relative)):
            shutil.copy2(source / row[key], output / relative)
            if sha256(output / relative) != fingerprint[key + "_sha256"]:
                raise ValueError("Source changed while copying replay export")
        exported.append({**row, "id": sid, "image": image_relative, "label": label_relative,
                         "replay_origin": origin, "oversampling_original_id": row["id"],
                         "oversampling_copy_index": copy_index, "source_export": str(source),
                         "source_export_image": row["image"], "source_export_label": row["label"]})
        allocation.append({"id": sid, "original_id": row["id"], "split": row["split"],
                           "origin": origin, "copy_index": copy_index, "image_sha256": fingerprint["image_sha256"],
                           "label_sha256": fingerprint["label_sha256"]})
    for source, metadata in ((base, old_metadata), (specialist, new_metadata)):
        if any(sha256(source / name) != expected for name, expected in metadata.items()):
            raise ValueError("Source metadata changed during replay export")
    (output / "exported_samples.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in exported), encoding="utf8")
    # JSON is valid YAML, so no additional serialization dependency is needed.
    write_json(output / "data.yaml", {"path": str(output), "train": "images/train", "val": "images/val", "test": "images/test", "names": ["person"]})
    unique_training = len(old_train) + len(new_train)
    summary = {"status": "completed", "reviewed_images_exported": len(exported),
        "unique_source_images": unique_training + len(retained), "training_unique_images": unique_training,
        "by_split": {key: value["images"] for key, value in counts.items()},
        "positive_images_by_split": {key: value["positive_images"] for key, value in counts.items()},
        "new_d03_unique_images": len(new_train), "old_replay_unique_images": len(old_train),
        "new_d03_repeat": new_repeat, "new_d03_training_exposure_fraction": len(new_train) * new_repeat / counts["train"]["images"],
        "oversampling_is_not_new_independent_data": True, "test_set_sealed": True, "training_started": False}
    write_json(output / "summary.json", summary)
    audit_output.mkdir(parents=True)
    fingerprints = fingerprint_export(output, {row["id"]: row for row in exported})
    write_json(audit_output / "fingerprints.json", fingerprints)
    write_json(audit_output / "sampling_allocation.json", allocation)
    audit = {"status": "passed", "audited_at": datetime.now().astimezone().isoformat(),
        "exported_dataset": str(output), "base_export": str(base), "specialist_export": str(specialist),
        "review_status": {"total": summary["unique_source_images"], "reviewed": summary["unique_source_images"],
                          "pending": 0, "excluded": 0, "test_set_sealed": True,
                          "annotation_provenance": "Inherited frozen human reviews; oversampling aliases are not new manual annotations"},
        "export_summary": summary, "by_split": {k: dict(v) for k, v in counts.items()},
        "by_origin": {k: dict(v) for k, v in origins.items()}, "by_camera": {k: dict(v) for k, v in cameras.items()},
        "source_metadata_sha256": {str(base): old_metadata, str(specialist): new_metadata},
        "export_metadata_sha256": {name: sha256(output / name) for name in ("data.yaml", "exported_samples.jsonl")},
        "excluded_evaluation_evidence_sha256": excluded_evidence,
        "required_checkpoint_initialization": {"path": str(initialization), "sha256": sha256(initialization)},
        "fresh_public_pretrained_initialization_required": False,
        "fingerprints_file": str(audit_output / "fingerprints.json"), "fingerprints_sha256": sha256(audit_output / "fingerprints.json"),
        "sampling_allocation_file": str(audit_output / "sampling_allocation.json"),
        "sampling_allocation_sha256": sha256(audit_output / "sampling_allocation.json"),
        "temporal_group_cross_split_conflicts": 0, "identical_image_cross_split_conflicts": 0,
        "source_frame_cross_split_conflicts": 0, "excluded_evaluation_training_conflicts": 0,
        "test_membership_and_hashes_preserved": True, "retained_validation_membership_and_hashes_preserved": True,
        "retired_old_validation_images_used_for_training": 0,
        "training_scope": "Balanced person fine-tuning: all original training images replayed once, newly reviewed D03 training images explicitly oversampled; original v10 val/test unchanged. Single person class, no tracker/workwear classifier training.",
        "test_split_evaluated": False, "training_started": False,
        "limitations": ["Repeated aliases increase augmented training exposure, not unique images or independent persons.",
                       "Retained validation has no D03 and cannot establish independent non-workwear generalization.",
                       "Any new/old training fit evaluation is diagnostic only; occlusion development truth stays outside training."]}
    write_json(audit_output / "annotation_audit.json", audit)
    write_json(output / "annotation_audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "specialist", "out", "audit-out", "initialization"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--new-repeat", type=int, default=3)
    parser.add_argument("--exclude-dataset", type=Path, action="append", default=[])
    args = parser.parse_args()
    result = prepare(args.base, args.specialist, args.out, args.audit_out, args.initialization, args.new_repeat, args.exclude_dataset)
    print(json.dumps({key: result[key] for key in ("status", "by_split", "by_origin", "export_summary")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
