"""Reopen train/val annotations, preserving current edits and the sealed v3 test."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

from person_tracking.common import write_json
from person_tracking.dataset import read_labels
from person_tracking.review import ReviewStore


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--validation-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source, frozen = ReviewStore(args.source), ReviewStore(args.splits)
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new review directory: {output}")
    if source.status()["pending"] or not frozen.summary.get("test_set_sealed"):
        raise ValueError("Need completed source annotations and a sealed reference split")
    source_rows = {r["id"]: r for r in source.rows}
    if set(source_rows) != {r["id"] for r in frozen.rows}:
        raise ValueError("Source and reference sample IDs must agree")
    review_hash = sha256(source.review_path)
    manifest_hash = sha256(source.root / "samples.jsonl")
    validation = json.loads((args.validation_results / "per_image.json").read_text(encoding="utf-8"))
    attention_ids = {r["id"] for r in validation if r["fp"] or r["fn"]}
    rows, reviews, fingerprints, preserved_edits = [], {}, [], []
    for folder in ("images", "reviewed_labels", "proposals/labels", "provenance"):
        (output / folder).mkdir(parents=True, exist_ok=False)
    for original in frozen.rows:
        sid, split = original["id"], original["split"]
        current = source_rows[sid]
        sealed = split == "test"
        label_source = frozen.label_path(original) if sealed else source.label_path(current)
        label_bytes = label_source.read_bytes()
        boxes = read_labels(label_source)
        if not sealed and source.reviews[sid]["status"] != "reviewed":
            raise ValueError(f"Unexpected excluded annotation: {sid}")
        if not sealed and (len(boxes) != source.reviews[sid]["box_count"] or
                           bool(source.reviews[sid]["confirmed_empty"]) != (not boxes)):
            raise ValueError(f"Review/label mismatch: {sid}")
        changed = not sealed and label_bytes != frozen.label_path(original).read_bytes()
        if changed:
            preserved_edits.append(sid)
        if sealed:
            priority, reason = 9, "测试集已封存，保留原标签"
        elif changed:
            priority, reason = 0, "已继承你最近的修改，请检查是否完整保留真实人员"
        elif sid in attention_ids:
            priority, reason = 1, "验证中出现误报或漏检，请核对真实人员、背景误标和框边界"
        elif not boxes and original.get("proposal_count", 0):
            priority, reason = 2, "反例候选：旧模型曾给出人员框；确认全图是否确实无人"
        elif not boxes:
            priority, reason = 3, "无人候选：检查远处、遮挡处是否有人；全图无人才能确认反例"
        else:
            priority, reason = 4, "保留所有真实人员，删除非人员框，并检查遗漏和边界"
        row = {**original, "image": f"images/{sid}.jpg", "proposal_label": f"proposals/labels/{sid}.txt",
               "review_status": "reviewed" if sealed else "pending", "annotation_locked": sealed,
               "review_priority": priority, "review_reason": reason,
               "source_correction_dataset": str(source.root), "split_reference_dataset": str(frozen.root)}
        image_source = frozen.root / original["image"]
        shutil.copy2(image_source, output / row["image"])
        for relative in (f"reviewed_labels/{sid}.txt", row["proposal_label"]):
            (output / relative).write_bytes(label_bytes)
        inherited_review = frozen.reviews[sid] if sealed else source.reviews[sid]
        reviews[sid] = dict(inherited_review) if sealed else {
            "status": "pending", "box_count": len(boxes), "confirmed_empty": False,
            "previous_review": inherited_review, "review_source": "negative_review_reopened"}
        fingerprints.append({"id": sid, "split": split, "camera_id": row["camera_id"],
            "image": row["image"], "image_sha256": sha256(output / row["image"]),
            "label": f"reviewed_labels/{sid}.txt", "label_sha256": sha256(output / f"reviewed_labels/{sid}.txt")})
        if sha256(image_source) != fingerprints[-1]["image_sha256"]:
            raise ValueError(f"Image copy mismatch: {sid}")
        rows.append(row)
    if sha256(source.review_path) != review_hash or sha256(source.root / "samples.jsonl") != manifest_hash:
        raise RuntimeError("Source annotations changed during copying; repeat in a new directory")
    label_hashes = {r["id"]: r["label_sha256"] for r in fingerprints}
    for row in rows:
        if row["split"] != "test" and sha256(source.label_path(source_rows[row["id"]])) != label_hashes[row["id"]]:
            raise RuntimeError(f"Source label changed during copying: {row['id']}")
    rows.sort(key=lambda r: (r["review_priority"], r["camera_id"], r["id"]))
    (output / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    write_json(output / "reviews.json", reviews)
    editable_count = sum(r["split"] != "test" for r in rows)
    summary = {"status": "completed", "images": len(rows), "by_split": dict(Counter(r["split"] for r in rows)),
        "split_selection_status": "fixed_v3_membership_for_v4_review", "test_set_sealed": True,
        "require_complete_review": True, "created_at": datetime.now().isoformat(),
        "source_dataset": str(source.root), "split_reference_dataset": str(frozen.root),
        "preserved_recent_corrections": preserved_edits, "training_started": False,
        "review_instructions": f"本轮复核训练/验证集 {editable_count} 张，已有人员框已保留。全图无人：确认无人并保存；有人：只框真实人员，灭火器、标识、设备不框为人员。同图有人时不要点确认无人。最近 {len(preserved_edits)} 处修改已继承。完成后通知我重训。"}
    write_json(output / "summary.json", summary)
    write_json(output / "provenance/initial_fingerprints.json", fingerprints)
    shutil.copy2(source.review_path, output / "provenance/source_reviews.json")
    shutil.copy2(frozen.root / "samples.jsonl", output / "provenance/reference_samples.jsonl")
    shutil.copy2(frozen.review_path, output / "provenance/reference_reviews.json")
    write_json(output / "review_plan.json", {**summary, "priority_counts": dict(Counter(str(r["review_priority"]) for r in rows)),
        "source_reviews_sha256": review_hash, "test_split_evaluated": False,
        "negative_label_format": "No new class: empty person label file for genuinely person-free images; background is implicit in mixed images.",
        "next_step": "Complete manual review, audit/export with fixed v3 membership, then retrain from public YOLO11n and validate; no training launched."})
    print(json.dumps({"dataset": str(output), "review_status": ReviewStore(output).status(),
                      "preserved_recent_corrections": preserved_edits}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
