"""Extend the reviewed person dataset without resplitting or training a model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import cv2
import numpy as np

from person_tracking.common import PACKAGE_DIR, load_config, read_jsonl, recording_time, write_json
from person_tracking.dataset import group_for_time, make_contact_sheets, read_image, read_labels, save_jpeg, write_labels
from person_tracking.detector import load_detector
from person_tracking.review import ReviewStore
from person_tracking.video import FileReader


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def box_iou(a, b):
    def corners(box):
        x, y, w, h = box
        return x-w/2, y-h/2, x+w/2, y+h/2
    ax1, ay1, ax2, ay2 = corners(a)
    bx1, by1, bx2, by2 = corners(b)
    intersection = max(0, min(ax2, bx2)-max(ax1, bx1)) * max(0, min(ay2, by2)-max(ay1, by1))
    return intersection / (a[2]*a[3]+b[2]*b[3]-intersection) if intersection else 0.0


def merge_proposals(predictions, threshold=0.5, limit=12):
    kept = []
    for prediction in sorted(predictions, key=lambda p: p["confidence"], reverse=True):
        if all(box_iou(prediction["box"], previous["box"]) < threshold for previous in kept):
            kept.append(prediction)
    return kept[:limit]


def eligible_group(camera, pts, frame, fps, groups, old_times, guard=90, spacing=5):
    """Require PTS and legacy frame clocks to agree; never extend test groups."""
    nominal = frame / fps
    pts_group = group_for_time(recording_time(camera, pts), guard=guard)
    legacy_group = group_for_time(recording_time(camera, nominal), guard=guard)
    if pts_group is None or pts_group != legacy_group or groups.get(pts_group) not in ("train", "val"):
        return None
    if old_times and min(abs(pts-t) for t in old_times) < spacing - 1e-6:
        return None
    if old_times and min(abs(nominal-t) for t in old_times) < spacing - 1e-6:
        return None
    return pts_group


def source_snapshot(store):
    files = {}
    for row in store.rows:
        for rel in (row["image"], row["proposal_label"], f"reviewed_labels/{row['id']}.txt"):
            files[rel] = sha256(store.root / rel)
    return {"files": files, "samples_sha256": sha256(store.root/"samples.jsonl"),
            "reviews_sha256": sha256(store.review_path), "images": len(store.rows)}


def scan(args):
    store = ReviewStore(args.source)
    if store.status()["reviewed"] != len(store.rows):
        raise ValueError("Source dataset must be fully reviewed before extension")
    if args.out.exists():
        raise FileExistsError(args.out)
    cfg = load_config()
    plan = json.loads((store.root/"sampling_plan.json").read_text(encoding="utf-8"))
    groups = plan["group_splits"]
    old_times, positive_train_times = defaultdict(list), defaultdict(list)
    for row in store.rows:
        old_times[row["camera_id"]].append(row["video_time_s"])
        if row["split"] == "train" and read_labels(store.label_path(row)):
            positive_train_times[row["camera_id"]].append(row["video_time_s"])
    snapshot = source_snapshot(store)
    known_hashes = {snapshot["files"][row["image"]] for row in store.rows}
    args.out.mkdir(parents=True)
    for folder in ("images", "proposals/labels", "contact_sheets"):
        (args.out/folder).mkdir(parents=True)
    write_json(args.out/"source_snapshot.json", snapshot)
    model_specs = [("finetuned", args.weights.resolve()), ("pretrained", Path(cfg["model"]))]
    detectors = []
    for label, path in model_specs:
        model, options = load_detector({**cfg, "model": str(path)}, args.device, 1280)
        options.update(conf=0.05 if label == "finetuned" else 0.10, max_det=20)
        detectors.append((label, model, options))
    scan_config = {"source_dataset": str(store.root), "interval_s": args.interval,
                   "guard_seconds": args.guard, "minimum_old_frame_spacing_s": args.old_spacing,
                   "group_splits": groups, "decode_mode": "sequential", "models": [
                       {"name": label, "path": str(path), "sha256": sha256(path)} for label, path in model_specs],
                   "test_frames_saved_or_inferred": 0, "training_started": False,
                   "note": "Candidates and detector disagreement are not ground truth; val selection will be model-blind."}
    write_json(args.out/"scan_config.json", scan_config)
    rows, stats, reader_summaries = [], Counter(), []
    started = time.perf_counter()
    with (args.out/"candidates.jsonl").open("w", encoding="utf-8") as manifest:
        for camera in cfg["cameras"]:
            reader = FileReader(camera["source"], analysis_fps=1/args.interval, decode_mode="sequential")
            try:
                while (item := reader.next()) is not None:
                    image, frame, pts, _ = item
                    stats["sampled_times"] += 1
                    group = eligible_group(camera, pts, frame, reader.fps, groups,
                                           old_times[camera["id"]], args.guard, args.old_spacing)
                    if group is None or reader.time_basis != "file_pts":
                        stats["skipped_time_or_spacing"] += 1
                        continue
                    wall = recording_time(camera, pts)
                    stem = f"{camera['id']}_{wall.strftime('%Y%m%dT%H%M%S')}_f{frame:09d}_v2"
                    image_path = args.out/"images"/(stem+".jpg")
                    save_jpeg(image_path, image)
                    image_hash = sha256(image_path)
                    if image_hash in known_hashes:
                        image_path.unlink()
                        stats["exact_image_duplicates_removed"] += 1
                        continue
                    known_hashes.add(image_hash)
                    raw_predictions = []
                    by_model = {}
                    for label, model, options in detectors:
                        result = model.predict(image, **options)[0]
                        predictions = []
                        for box, confidence in zip(result.boxes.xywhn.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                            box = [float(v) for v in box]
                            predictions.append({"box": box, "confidence": float(confidence), "model": label})
                        by_model[label] = predictions
                        raw_predictions.extend(predictions)
                    merged = merge_proposals(raw_predictions)
                    disagreement = any(all(box_iou(p["box"], q["box"]) < 0.3 for q in by_model[other])
                                       for label, other in (("finetuned", "pretrained"), ("pretrained", "finetuned"))
                                       for p in by_model[label])
                    near_person = (groups[group] == "train" and any(abs(pts-t) <= 20
                                   for t in positive_train_times[camera["id"]]))
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    row = {"id": stem, "camera_id": camera["id"], "source": camera["source"],
                           "source_frame": frame, "source_fps": reader.fps, "video_time_s": pts,
                           "time_basis": reader.time_basis, "source_frame_basis": reader.source_frame_basis,
                           "legacy_nominal_time_s": frame/reader.fps, "recording_time_estimate": wall.isoformat(),
                           "split_group": group, "split": groups[group], "image": f"images/{stem}.jpg",
                           "proposal_label": f"proposals/labels/{stem}.txt", "width": image.shape[1],
                           "height": image.shape[0], "review_status": "pending", "proposal_generated": True,
                           "proposal_count": len(merged), "proposal_confidences": [p["confidence"] for p in merged],
                           "proposal_sources": merged, "detector_disagreement": disagreement,
                           "near_reviewed_train_person": near_person, "image_sha256": image_hash,
                           "blur_laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                           "mean_brightness": float(gray.mean())}
                    write_labels(args.out/row["proposal_label"], [p["box"] for p in merged])
                    manifest.write(json.dumps(row, ensure_ascii=False)+"\n")
                    manifest.flush()
                    rows.append(row)
                    if len(rows) % 25 == 0:
                        print(f"Candidate pool: {len(rows)} images; {camera['id']} PTS {pts:.1f}s; elapsed {time.perf_counter()-started:.0f}s", flush=True)
            finally:
                reader_summaries.append({"camera_id": camera["id"], "finish_reason": reader.finish_reason,
                                         "timestamp_fallback_frames": reader.timestamp_fallback_frames})
                reader.close()
    summary = {"status": "completed", "candidates": len(rows), "by_camera_split": dict(Counter(
        r["camera_id"]+"_"+r["split"] for r in rows)), "model_person_candidates": sum(r["proposal_count"]>0 for r in rows),
        "statistics": dict(stats), "reader_summaries": reader_summaries,
        "elapsed_seconds": time.perf_counter()-started, "training_started": False}
    write_json(args.out/"summary.json", summary)
    make_contact_sheets(args.out, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def spaced(row, selected, spacing):
    return all(abs(row["video_time_s"]-other["video_time_s"]) >= spacing-1e-6 for other in selected)


def uniform_select(rows, number, fixed=(), spacing=9.5):
    """Select by time coverage only; never consult proposals or human labels."""
    available = sorted(rows, key=lambda r: r["video_time_s"])
    selected = []
    if number == 0:
        return selected
    if not available:
        raise ValueError("Not enough eligible uniformly sampled frames")
    targets = np.linspace(0, len(available)-1, number)
    for target in targets:
        ordered = sorted(enumerate(available), key=lambda pair: (abs(pair[0]-target), pair[0]))
        row = next((r for _, r in ordered if r not in selected and spaced(r, [*fixed, *selected], spacing)), None)
        if row is None:
            raise ValueError(f"Cannot select {number} sufficiently spaced coverage frames")
        selected.append(row)
    return selected


def select_additions(rows, source_rows, target_total, val_per_camera=18, spacing=9.5, camera_totals=None):
    cameras = sorted({r["camera_id"] for r in source_rows})
    if target_total <= len(source_rows):
        raise ValueError("Target must exceed the source count")
    if camera_totals is None:
        if target_total % len(cameras):
            raise ValueError("Balanced target must be divisible by the number of cameras")
        camera_totals = {camera: target_total//len(cameras) for camera in cameras}
    if set(camera_totals) != set(cameras) or sum(camera_totals.values()) != target_total:
        raise ValueError("Camera totals must cover all cameras and sum to the target")
    selected = []
    rng = random.Random(20261106)
    for camera in cameras:
        original = sum(r["camera_id"] == camera for r in source_rows)
        count = camera_totals[camera]-original
        if count < 0:
            raise ValueError("Camera totals must retain every inherited source sample")
        candidates = [r for r in rows if r["camera_id"] == camera]
        if any(r["split"] not in ("train", "val") for r in candidates):
            raise ValueError("Extension candidates must not contain test samples")
        val_count = min(val_per_camera, count)
        validation = uniform_select([r for r in candidates if r["split"] == "val"], val_count, spacing=spacing)
        selected.extend({**r, "selection_category": "val_uniform_model_blind"} for r in validation)
        train_rows = [r for r in candidates if r["split"] == "train"]
        train_count = count-val_count
        blind_count = math.ceil(train_count*0.30)
        blind = uniform_select(train_rows, blind_count, spacing=spacing)
        training = list(blind)
        # Prefer disagreement, potential persons and reviewed train-positive neighborhoods.
        ranked = []
        for r in train_rows:
            if not (r.get("near_reviewed_train_person") or r.get("detector_disagreement") or r["proposal_count"]):
                continue
            score = (3*bool(r.get("near_reviewed_train_person")) + 2*bool(r.get("detector_disagreement"))
                     + 2*bool(r["proposal_count"]) + 0.5*min(r["proposal_count"], 3) + rng.random())
            ranked.append((score, r))
        buckets = defaultdict(list)
        for score, r in sorted(ranked, key=lambda item: item[0], reverse=True):
            buckets[r["split_group"]].append(r)
        while len(training) < train_count:
            changed = False
            for group in sorted(buckets):
                while buckets[group]:
                    row = buckets[group].pop(0)
                    if row not in training and spaced(row, training, spacing):
                        training.append(row)
                        changed = True
                        break
                if len(training) == train_count:
                    break
            if not changed:
                break
        blind_ids = {r["id"] for r in blind}
        hard_ids = {r["id"] for r in training if r["id"] not in blind_ids}
        if len(training) < train_count:
            training.extend(uniform_select([r for r in train_rows if r not in training],
                                           train_count-len(training), fixed=training, spacing=spacing))
        selected.extend({**r, "selection_category": "train_uniform_coverage" if r["id"] in blind_ids
                         else "train_person_or_disagreement" if r["id"] in hard_ids
                         else "train_uniform_gap_fill"} for r in training)
    return sorted(selected, key=lambda r: (r["camera_id"], r["video_time_s"]))


def build(args):
    store = ReviewStore(args.source)
    pool_config = json.loads((args.pool/"scan_config.json").read_text(encoding="utf-8"))
    snapshot = json.loads((args.pool/"source_snapshot.json").read_text(encoding="utf-8"))
    if source_snapshot(store) != snapshot:
        raise ValueError("Source dataset changed after scan; do not overwrite or silently lose reviews")
    candidates = read_jsonl(args.pool/"candidates.jsonl")
    camera_totals = dict(part.split("=", 1) for part in args.camera_totals) if args.camera_totals else None
    if camera_totals is not None:
        camera_totals = {camera: int(count) for camera, count in camera_totals.items()}
    additions = select_additions(candidates, store.rows, args.target_total, args.val_per_camera,
                                camera_totals=camera_totals)
    if args.out.exists():
        raise FileExistsError(args.out)
    groups = pool_config["group_splits"]
    for row in additions:
        if groups.get(row["split_group"]) != row["split"] or row["split"] == "test":
            raise ValueError("Candidate split differs from the source group mapping")
    args.out.mkdir(parents=True)
    for folder in ("images", "proposals/labels", "reviewed_labels", "contact_sheets", "provenance"):
        (args.out/folder).mkdir(parents=True)
    all_rows = []
    for row in store.rows:
        inherited = {**row, "origin_dataset": str(store.root), "inherited_review": True,
                     "annotation_locked": row["split"] == "test"}
        inherited.setdefault("time_basis", "filename_plus_frame_index_legacy_seek")
        inherited.setdefault("source_frame_basis", "opencv_position_estimate_after_seek")
        for rel in (row["image"], row["proposal_label"], f"reviewed_labels/{row['id']}.txt"):
            shutil.copy2(store.root/rel, args.out/rel)
        all_rows.append(inherited)
    shutil.copy2(store.review_path, args.out/"reviews.json")
    for row in additions:
        for rel in (row["image"], row["proposal_label"]):
            shutil.copy2(args.pool/rel, args.out/rel)
        all_rows.append({**row, "inherited_review": False, "annotation_locked": False})
    for name in ("samples.jsonl", "sampling_plan.json", "summary.json"):
        shutil.copy2(store.root/name, args.out/"provenance"/("source_"+name))
    shutil.copy2(args.pool/"source_snapshot.json", args.out/"provenance/source_snapshot.json")
    for name, records in (("samples.jsonl", all_rows), ("new_samples.jsonl", additions)):
        (args.out/name).write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records), encoding="utf-8")
    plan = {"mode": "extend_without_resplitting", "source_dataset": str(store.root),
            "candidate_pool": str(args.pool.resolve()), "group_splits": groups,
            "inherited_images": len(store.rows), "added_images": len(additions),
            "target_total": args.target_total, "new_test_images": 0,
            "camera_totals": dict(Counter(r["camera_id"] for r in all_rows)),
            "new_time_basis": "filename_plus_video_pts", "inherited_time_basis": "legacy unchanged; explicitly marked",
            "val_selection": "Uniform chronological coverage; no detector scores or labels used for selection",
            "train_selection": "30% uniform coverage; remainder potential persons, train-positive neighborhoods and detector disagreement, cycled across temporal groups",
            "new_sample_minimum_spacing_s": 9.5, "models": pool_config["models"],
            "new_machine_proposals_are_not_ground_truth": True, "training_started": False}
    write_json(args.out/"sampling_plan.json", plan)
    summary = {"status": "completed", "images": len(all_rows),
               "by_camera": dict(Counter(r["camera_id"] for r in all_rows)),
               "by_split": dict(Counter(r["split"] for r in all_rows)),
               "reviewed_images": len(store.reviews), "pending_images": len(additions),
               "new_by_camera_split": dict(Counter(r["camera_id"]+"_"+r["split"] for r in additions)),
               "selection_categories": dict(Counter(r["selection_category"] for r in additions)),
               "new_person_candidate_images": sum(r["proposal_count"]>0 for r in additions),
               "new_no_proposal_images": sum(r["proposal_count"]==0 for r in additions),
               "note": "Inherited manual labels retained; all added images require manual review, including empty proposals.",
               "training_started": False}
    write_json(args.out/"summary.json", summary)
    image_hash_splits = defaultdict(set)
    for row in all_rows:
        image_hash_splits[sha256(args.out/row["image"])].add(row["split"])
    copy_mismatches = [rel for rel, digest in snapshot["files"].items() if sha256(args.out/rel) != digest]
    leakage = [digest for digest, splits in image_hash_splits.items() if len(splits)>1]
    audit = {"source_files_preserved": not copy_mismatches, "source_copy_mismatches": copy_mismatches,
             "reviews_sha256_preserved": sha256(args.out/"reviews.json") == snapshot["reviews_sha256"],
             "sample_ids_unique": len({r["id"] for r in all_rows}) == len(all_rows),
             "exact_image_hashes_unique": len(image_hash_splits) == len(all_rows),
             "cross_split_exact_duplicate_images": leakage,
             "test_images_added": 0, "test_images_locked": sum(r["annotation_locked"] for r in all_rows),
             "new_samples_reviewed": 0, "training_started": False, "test_split_evaluated": False}
    write_json(args.out/"expansion_audit.json", audit)
    if copy_mismatches or leakage or not audit["sample_ids_unique"] or not audit["exact_image_hashes_unique"]:
        raise ValueError("Expansion integrity audit failed; see expansion_audit.json")
    make_contact_sheets(args.out, additions)
    print(json.dumps({"dataset": str(args.out.resolve()), **summary, "audit": audit}, ensure_ascii=False, indent=2))


def main():
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PACKAGE_DIR/".cache/ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(PACKAGE_DIR/".cache/matplotlib"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    scan_parser = sub.add_parser("scan")
    scan_parser.add_argument("--source", type=Path, required=True)
    scan_parser.add_argument("--out", type=Path, required=True)
    scan_parser.add_argument("--weights", type=Path, required=True)
    scan_parser.add_argument("--device", default="0")
    scan_parser.add_argument("--interval", type=float, default=5)
    scan_parser.add_argument("--guard", type=int, default=90)
    scan_parser.add_argument("--old-spacing", type=float, default=5)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--source", type=Path, required=True)
    build_parser.add_argument("--pool", type=Path, required=True)
    build_parser.add_argument("--out", type=Path, required=True)
    build_parser.add_argument("--target-total", type=int, default=800)
    build_parser.add_argument("--camera-totals", nargs="+", help="Optional final counts, e.g. D01=180 D02=205 D03=200 D04=215")
    build_parser.add_argument("--val-per-camera", type=int, default=18)
    args = parser.parse_args()
    if args.mode == "scan":
        scan(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
