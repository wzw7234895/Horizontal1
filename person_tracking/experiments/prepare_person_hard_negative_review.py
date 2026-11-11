"""Prepare locked-base, multi-camera hard-negative review candidates from a run."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import shutil

import cv2

from person_tracking.common import read_jsonl, write_json
from person_tracking.dataset import group_for_time, save_jpeg, write_labels
from person_tracking.review import ReviewStore


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def proposal_score(row, track_lengths, min_confidence):
    detections = [d for d in row.get("detections", []) if float(d["confidence"]) >= min_confidence]
    if not detections:
        return None
    untracked = any(d.get("track_id") is None for d in detections)
    short_track = any(track_lengths.get(int(d["track_id"]), 0) <= 10
                      for d in detections if d.get("track_id") is not None)
    score = max(float(d["confidence"]) for d in detections)
    score += 0.08 * untracked + 0.05 * short_track + 0.02 * (len(detections) > 1)
    return score, untracked, short_track


def select_candidates(frame_rows, train_groups, existing_times, track_lengths,
                      min_confidence=0.2, positive_bucket_seconds=10, empty_bucket_seconds=60,
                      duplicate_guard_seconds=2.0):
    positive, empty = {}, {}
    for row in frame_rows:
        when = datetime.fromisoformat(row["recording_time_estimate"])
        group = group_for_time(when)
        if group not in train_groups:
            continue
        if any(abs((when - old).total_seconds()) < duplicate_guard_seconds for old in existing_times):
            continue
        group_start = datetime.fromisoformat(group)
        offset = max(0.0, (when - group_start).total_seconds())
        frame_index = int(row["source_frame"])
        score = proposal_score(row, track_lengths, min_confidence)
        if score is not None:
            bucket = (group, int(offset // positive_bucket_seconds))
            prior = positive.get(bucket)
            detection_count = sum(float(d["confidence"]) >= min_confidence
                                  for d in row.get("detections", []))
            rank = (int(score[1]), int(score[2]), int(detection_count > 1),
                    score[0], -frame_index)
            if prior is None or rank > prior[0]:
                positive[bucket] = (rank, row, score)
        else:
            bucket = (group, int(offset // empty_bucket_seconds))
            distance = abs(offset % empty_bucket_seconds - empty_bucket_seconds / 2)
            prior = empty.get(bucket)
            if prior is None or distance < prior[0]:
                empty[bucket] = (distance, row)

    selected = {}
    for _, row, score in positive.values():
        selected[int(row["source_frame"])] = (row, {
            "candidate_type": "detector_candidate_review",
            "suspected_unassociated_detection": score[1],
            "suspected_short_track": score[2],
        })
    for _, row in empty.values():
        selected.setdefault(int(row["source_frame"]), (row, {
            "candidate_type": "no_detection_review",
            "suspected_unassociated_detection": False,
            "suspected_short_track": False,
        }))
    return [selected[index] for index in sorted(selected)]


def copy_locked_base(source, output):
    base_rows, base_reviews, fingerprints = [], {}, []
    for row in source.rows:
        copied = {**row, "image": f"images/{row['id']}.jpg",
                  "proposal_label": f"proposals/labels/{row['id']}.txt",
                  "review_status": "reviewed", "annotation_locked": True}
        image_source = source.root / row["image"]
        label_source = source.label_path(row)
        image_target = output / copied["image"]
        label_target = output / copied["proposal_label"]
        reviewed_target = output / "reviewed_labels" / f"{row['id']}.txt"
        shutil.copy2(image_source, image_target)
        shutil.copy2(label_source, label_target)
        shutil.copy2(label_source, reviewed_target)
        review = source.reviews[row["id"]]
        base_reviews[row["id"]] = review
        fingerprints.append({"id": row["id"], "split": row["split"],
            "image_sha256": sha256(image_target), "label_sha256": sha256(reviewed_target)})
        base_rows.append(copied)
    return base_rows, base_reviews, fingerprints


def prepare_review(base_dataset, inference_run, output, interval_seconds=10,
                   empty_interval_seconds=60, min_confidence=0.2, base_export_dataset=None):
    base, run, output = Path(base_dataset).resolve(), Path(inference_run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new --out directory: {output}")
    source = ReviewStore(base)
    allowed_split_status = {"fixed_v3_membership_for_v4_review",
                            "fixed_v4_membership_supplemental_train_only",
                            "fixed_baseline_membership_supplemental_train_only",
                            "frozen_corrected_preserved_membership"}
    if source.status()["pending"] or source.summary.get("split_selection_status") not in allowed_split_status:
        raise ValueError("Need a completed, fixed-split annotation source")
    if not source.summary.get("test_set_sealed"):
        raise ValueError("The existing test split must remain sealed")
    reference_dataset = Path(base_export_dataset).resolve() if base_export_dataset else Path(
        source.summary.get("split_reference_dataset", "")).resolve()
    reference_manifest = reference_dataset / "exported_samples.jsonl"
    if not reference_dataset.is_dir() or not reference_manifest.is_file():
        raise ValueError("Pass the matching exported baseline with --base-export")
    reference_rows = read_jsonl(reference_manifest)
    reference_splits = {row["id"]: row["split"] for row in reference_rows}
    if set(reference_splits) != {row["id"] for row in source.rows}:
        raise ValueError("The exported baseline membership does not match the review source")
    if any(reference_splits[row["id"]] != row["split"] for row in source.rows):
        raise ValueError("The exported baseline split assignments do not match the review source")
    run_config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    summaries = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if summaries.get("status") != "completed":
        raise ValueError("Need a completed inference run")
    camera_summaries = {item["camera_id"]: item for item in summaries["cameras"]}
    camera_configs = run_config.get("cameras", [])
    if not camera_configs or {camera["id"] for camera in camera_configs} != set(camera_summaries):
        raise ValueError("Inference config and completed camera summaries do not match")
    group_splits = {}
    existing_times = {}
    existing_ids = set()
    for row in source.rows:
        old = group_splits.setdefault(row["split_group"], row["split"])
        if old != row["split"]:
            raise ValueError("A temporal group has multiple split assignments")
        if row["split"] == "train":
            existing_times.setdefault(row["camera_id"], []).append(
                datetime.fromisoformat(row["recording_time_estimate"]))
        existing_ids.add(row["id"])
    train_groups = {group for group, split in group_splits.items() if split == "train"}
    selected_by_camera = {}
    for camera in camera_configs:
        camera_id = camera["id"]
        camera_summary = camera_summaries[camera_id]
        if camera_summary.get("status") != "completed":
            raise ValueError(f"Camera inference did not complete: {camera_id}")
        video_path = Path(camera["source"])
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        camera_output = run / camera_id
        track_lengths = {int(item["track_id"]): int(item["observed_frames"])
                         for item in read_jsonl(camera_output / "tracks.jsonl")}
        selected = select_candidates(read_jsonl(camera_output / "frames.jsonl"), train_groups,
            existing_times.get(camera_id, []), track_lengths, min_confidence=min_confidence,
            positive_bucket_seconds=interval_seconds, empty_bucket_seconds=empty_interval_seconds)
        if selected:
            selected_by_camera[camera_id] = (camera, camera_summary, selected)
    if not selected_by_camera:
        raise ValueError("No candidates fell inside the existing training time groups")

    (output / "images").mkdir(parents=True)
    (output / "reviewed_labels").mkdir()
    (output / "proposals" / "labels").mkdir(parents=True)
    base_rows, reviews, base_fingerprints = copy_locked_base(source, output)
    new_rows = []
    inference_manifests = {}
    for camera_id, (camera, camera_summary, selected) in selected_by_camera.items():
        video_path = Path(camera["source"])
        camera_output = run / camera_id
        inference_manifests[camera_id] = {
            "frames_sha256": sha256(camera_output / "frames.jsonl"),
            "tracks_sha256": sha256(camera_output / "tracks.jsonl"),
        }
        requested = {int(row["source_frame"]): (row, reason) for row, reason in selected}
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(camera_summary.get("source_fps", 0))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total and max(requested) >= total:
            cap.release()
            raise ValueError(f"Candidate source frame exceeds {camera_id} video's declared frame count")

        frame_index = 0
        pending = set(requested)
        try:
            while pending:
                if not cap.grab():
                    break
                if frame_index in pending:
                    ok, frame = cap.retrieve()
                    if not ok:
                        raise OSError(f"Could not retrieve selected {camera_id} frame {frame_index}")
                    source_row, reason = requested[frame_index]
                    when = datetime.fromisoformat(source_row["recording_time_estimate"])
                    group = group_for_time(when)
                    stem = f"{camera_id}_{when.strftime('%Y%m%dT%H%M%S')}_f{frame_index:09d}_hard"
                    if stem in existing_ids:
                        raise ValueError(f"Candidate ID collides with the base dataset: {stem}")
                    image_relative = f"images/{stem}.jpg"
                    proposal_relative = f"proposals/labels/{stem}.txt"
                    save_jpeg(output / image_relative, frame)
                    height, width = frame.shape[:2]
                    proposals, confidences = [], []
                    for detection in source_row.get("detections", []):
                        x1, y1, x2, y2 = detection["xyxy"]
                        x1 = min(width, max(0.0, float(x1)))
                        x2 = min(width, max(0.0, float(x2)))
                        y1 = min(height, max(0.0, float(y1)))
                        y2 = min(height, max(0.0, float(y2)))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        proposals.append([((x1 + x2) / (2 * width)), ((y1 + y2) / (2 * height)),
                                          (x2 - x1) / width, (y2 - y1) / height])
                        confidences.append(float(detection["confidence"]))
                    write_labels(output / proposal_relative, proposals)
                    priority = 0 if reason["suspected_unassociated_detection"] else (
                        1 if reason["suspected_short_track"] else 2 if proposals else 3)
                    new_rows.append({
                        "id": stem, "camera_id": camera_id, "source": str(video_path),
                        "source_frame": frame_index, "source_fps": fps,
                        "video_time_s": float(source_row["video_time_s"]),
                        "time_basis": source_row["time_basis"],
                        "source_frame_basis": "decoded_frame_count",
                        "recording_time_estimate": when.isoformat(), "split_group": group,
                        "split": "train", "image": image_relative, "width": width, "height": height,
                        "review_status": "pending", "proposal_label": proposal_relative,
                        "proposal_count": len(proposals), "proposal_confidences": confidences,
                        "proposal_generated": True, "annotation_locked": False,
                        "review_priority": priority,
                        "review_reason": "机器候选帧，仅提示检查；全画面逐一确认人员，非人员目标不得标为人员。",
                        "candidate_type": reason["candidate_type"],
                        "suspected_unassociated_detection": reason["suspected_unassociated_detection"],
                        "suspected_short_track": reason["suspected_short_track"],
                    })
                    pending.remove(frame_index)
                frame_index += 1
        finally:
            cap.release()
        if pending:
            raise OSError(f"{camera_id} source decode ended before {len(pending)} selected frames; no training started")

    rows = sorted(new_rows, key=lambda row: (row["review_priority"], row["camera_id"], row["source_frame"])) + base_rows
    for row in rows:
        previous = group_splits.setdefault(row["split_group"], row["split"])
        if previous != row["split"]:
            raise ValueError(f"New candidate crosses a fixed split boundary: {row['id']}")
    (output / "samples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    write_json(output / "reviews.json", reviews)
    summary = {
        "status": "completed", "images": len(rows), "base_images_locked": len(base_rows),
        "supplemental_images_pending": len(new_rows),
        "supplemental_candidate_types": dict(Counter(row["candidate_type"] for row in new_rows)),
        "supplemental_by_camera": dict(Counter(row["camera_id"] for row in new_rows)),
        "by_split": {"train": sum(row["split"] == "train" for row in rows),
                     "val": sum(row["split"] == "val" for row in rows),
                     "test": sum(row["split"] == "test" for row in rows)},
        "split_selection_status": "fixed_baseline_membership_supplemental_train_only",
        "test_set_sealed": True, "require_complete_review": True,
        "created_at": datetime.now().isoformat(), "source_dataset": str(source.root),
        "base_exported_dataset": str(reference_dataset),
        "inference_run": str(run),
        "videos": {camera_id: str(camera["source"]) for camera_id, (camera, _, _) in selected_by_camera.items()},
        "training_started": False,
        "review_instructions": "复核新增候选图的整幅画面：只框真实人员；吊钩、钢丝绳、工件、设备、反光和背景均不得标为人员。候选框不是标签，空画面必须使用‘确认无人并保存’；存在人员时不可确认无人。原有标注、固定 val/test 划分和 test 样本均锁定。复核完成后运行 finalize-hard-negatives 命令审计并导出，再用同一 val 对比后决定是否重训。"
    }
    write_json(output / "summary.json", summary)
    write_json(output / "provenance.json", {
        "base_dataset": str(source.root), "base_exported_dataset": str(reference_dataset),
        "base_samples_sha256": sha256(source.root / "samples.jsonl"),
        "base_reviews_sha256": sha256(source.review_path),
        "base_image_label_fingerprints": base_fingerprints,
        "base_test_ids": sorted(row["id"] for row in source.rows if row["split"] == "test"),
        "inference_run_config_sha256": sha256(run / "run_config.json"),
        "inference_summary_sha256": sha256(run / "summary.json"),
        "inference_manifests": inference_manifests,
        "candidate_selection": {"confidence_floor": min_confidence,
            "max_one_detection_candidate_per_seconds": interval_seconds,
            "max_one_no_detection_candidate_per_seconds": empty_interval_seconds,
            "train_groups_only": sorted(train_groups),
            "existing_sample_duplicate_guard_seconds": 2.0},
        "candidate_cameras": sorted(selected_by_camera),
        "candidate_ids": [row["id"] for row in new_rows],
        "test_split_evaluated": False, "training_started": False,
    })
    print(json.dumps({"dataset": str(output), "summary": summary,
                      "review_status": ReviewStore(output).status()}, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-export", type=Path)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=10)
    parser.add_argument("--empty-interval-seconds", type=int, default=60)
    parser.add_argument("--min-confidence", type=float, default=0.2)
    args = parser.parse_args()
    if args.interval_seconds <= 0 or args.empty_interval_seconds <= 0:
        parser.error("sampling intervals must be positive")
    if not 0 <= args.min_confidence <= 1:
        parser.error("--min-confidence must be between 0 and 1")
    prepare_review(args.base, args.inference, args.out, args.interval_seconds,
                   args.empty_interval_seconds, args.min_confidence, args.base_export)


if __name__ == "__main__":
    main()
