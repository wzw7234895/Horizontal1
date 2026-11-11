"""Prepare user-selected D03 plain-clothes candidates for manual person labels."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

import cv2

from person_tracking.common import PACKAGE_DIR, load_config, read_jsonl, recording_time, write_json
from person_tracking.dataset import group_for_time, make_contact_sheets, read_image, save_jpeg, write_labels
from person_tracking.review import ReviewStore
from person_tracking.video import FileReader


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output, audit_output, interval=3.0):
    output, audit_output = Path(output).resolve(), Path(audit_output).resolve()
    if interval <= 0:
        raise ValueError("interval must be positive")
    if output.exists() or audit_output.exists():
        raise FileExistsError("Use new output and audit directories")
    base = PACKAGE_DIR / "data/person_v9_corrected_snapshot_20261107"
    base_rows = read_jsonl(base / "samples.jsonl")
    source_meta_hashes = {name: digest(base / name) for name in ("samples.jsonl", "summary.json", "reviews.json")}
    cfg = load_config(PACKAGE_DIR / "configs/cameras.json", ["D03"])
    camera = cfg["cameras"][0]
    origin = datetime.fromisoformat(camera["recording_start"])
    windows = [(datetime.fromisoformat("2026-10-20T" + a), datetime.fromisoformat("2026-10-20T" + b))
               for a, b in (("16:20:00", "16:23:00"), ("16:25:00", "16:26:00"))]
    original_d03_times = [datetime.fromisoformat(row["recording_time_estimate"])
                          for row in base_rows if row["camera_id"] == "D03"]
    conflict_group = "2026-10-20T16:20:00"
    conflict_rows = [row for row in base_rows if row["split_group"] == conflict_group]
    assert conflict_rows and {row["split"] for row in conflict_rows} == {"val"}
    for folder in ("images", "proposals/labels", "reviewed_labels", "contact_sheets"):
        (output / folder).mkdir(parents=True)
    audit_output.mkdir(parents=True)
    start = (windows[0][0] - origin).total_seconds() - camera.get("time_offset_s", 0)
    end = (windows[-1][1] - origin).total_seconds() - camera.get("time_offset_s", 0)
    reader = FileReader(camera["source"], analysis_fps=1 / interval, start=start,
                        duration=end-start, decode_mode="sequential")
    rows, skipped_near_existing, skipped_between_windows = [], 0, 0
    status = "failed"
    try:
        while (item := reader.next()) is not None:
            frame, source_frame, pts, _ = item
            clock = recording_time(camera, pts)
            if not any(lo <= clock < hi for lo, hi in windows):
                skipped_between_windows += 1
                continue
            # Existing confirmed images remain in their own frozen dataset.
            if any(abs((clock-old).total_seconds()) <= 1.0 for old in original_d03_times):
                skipped_near_existing += 1
                continue
            stem = f"D03_{clock.strftime('%Y%m%dT%H%M%S')}_f{source_frame:09d}_plain"
            image = f"images/{stem}.jpg"
            label = f"proposals/labels/{stem}.txt"
            save_jpeg(output / image, frame)
            write_labels(output / label, [])
            row = {
                "id": stem, "camera_id": "D03", "source": camera["source"],
                "source_frame": source_frame, "source_fps": reader.fps,
                "video_time_s": pts, "time_basis": reader.time_basis,
                "source_frame_basis": reader.source_frame_basis,
                "recording_time_estimate": clock.isoformat(),
                "split_group": group_for_time(clock, seconds=600, guard=0),
                "split": "train", "split_assignment_status": "candidate_only_pending_reselection",
                "original_temporal_group_split": "val", "image": image,
                "width": frame.shape[1], "height": frame.shape[0],
                "review_status": "pending", "annotation_locked": False,
                "proposal_label": label, "proposal_count": 0,
                "proposal_confidences": [], "proposal_generated": False,
                "review_reason": "D03 便装专项新增候选：逐人画框，包括可确认的遮挡/远处人员；吊钩与设备不标。当前 train 仅表示训练候选，正式分集尚未确定。",
                "candidate_source": "user_selected_clock_windows",
                "clothing_attribute": "user_reported_window_not_person_level_ground_truth",
                "image_sha256": digest(output / image),
            }
            rows.append(row)
            if len(rows) % 10 == 0:
                print(f"Prepared {len(rows)} new D03 images", flush=True)
        if not rows:
            raise ValueError("No new candidate frames were sampled")
        assert len({row["id"] for row in rows}) == len(rows)
        assert len({row["source_frame"] for row in rows}) == len(rows)
        for row in rows:
            image = read_image(output / row["image"])
            assert image.shape[:2] == (1080, 1920)
        for name, before in source_meta_hashes.items():
            assert digest(base / name) == before, f"Frozen source metadata changed: {name}"
        status = "completed"
    finally:
        finish_reason = reader.finish_reason
        timestamp_fallbacks = reader.timestamp_fallback_frames
        reader.close()
        with (output / "samples.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(output / "reviews.json", {})
        write_json(output / "summary.json", {
            "status": status, "images": len(rows), "by_camera": {"D03": len(rows)},
            "by_split": {"train": len(rows)}, "reviewed_images": 0,
            "proposal_boxes": 0, "no_proposal_images": len(rows),
            "require_complete_review": True, "split_selection_status": "pending_reselection",
            "training_started": False,
            "review_instructions": "仅标注新增 D03 图像。每名可确认人员一个可见范围紧致框；空白框不是无人结论。train 为候选标记，完成复核后需要新分集才能导出训练。",
        })
    plan = {
        "source_dataset": str(base), "source": camera["source"],
        "requested_windows": [{"start": lo.isoformat(), "end_exclusive": hi.isoformat()} for lo, hi in windows],
        "interval_s": interval, "decode_mode": "sequential", "original_resolution": [1920, 1080],
        "model_proposals_used": False, "skipped_near_existing_images": skipped_near_existing,
        "skip_existing_tolerance_seconds": 1.0, "skipped_between_windows": skipped_between_windows,
        "sampled_images": len(rows), "reader_finish_reason": finish_reason,
        "timestamp_fallback_frames": timestamp_fallbacks,
        "conflicting_shared_group": conflict_group, "existing_group_images": len(conflict_rows),
        "original_group_by_split": dict(Counter(row["split"] for row in conflict_rows)),
        "source_metadata_sha256": source_meta_hashes,
        "training_policy": "Specialist fine-tuning may start only after human labels and new split/export audits; retain a general-person replay set to measure forgetting.",
        "validation_policy": "Do not retain the original shared validation group when its candidates are used for training. New D03 validation needs separate footage/events; the remaining old validation has no D03 examples.",
        "clock_limit": "Clock windows use filename plus PTS. Burn-in clock must be checked visually; do not claim hardware synchronization.",
        "training_started": False,
    }
    write_json(output / "sampling_plan.json", plan)
    make_contact_sheets(output, rows)
    store = ReviewStore(output)
    assert store.status()["pending"] == len(rows)
    assert not any(store.item(i)["boxes"] for i in range(len(rows)))
    check = {"status": "passed", "dataset": str(output), "images": len(rows),
             "all_images_decoded_at_original_resolution": True,
             "pending_manual_labels": store.status()["pending"],
             "all_proposals_empty_and_unconfirmed": True,
             "frozen_source_metadata_unchanged": True,
             "export_blocked_pending_reselection": True, "training_started": False,
             "sampling_plan": str(output / "sampling_plan.json")}
    write_json(audit_output / "preparation_check.json", check)
    print(json.dumps(check, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=3.0)
    args = parser.parse_args()
    prepare(args.out, args.audit_out, args.interval)
