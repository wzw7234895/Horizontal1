"""Sample training-only occlusion candidates and assemble a blank manual review.

No model inference or training is performed.  Discovery images are not labels.
The final review requires a separate, explicit visual-selection JSON document.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from person_tracking.common import PACKAGE_DIR, load_config, read_jsonl, recording_time, write_json
from person_tracking.dataset import group_for_time, read_image, save_jpeg, write_labels
from person_tracking.review import ReviewStore
from person_tracking.video import FileReader


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_rows(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def clock(value):
    return datetime.fromisoformat(value if "T" in value else "2026-10-20T" + value)


def history(exclude):
    """Include old manual/excluded frames, retired splits, and evaluation images."""
    paths = sorted(set((PACKAGE_DIR / "data").glob("*/samples.jsonl"))
                   | set((PACKAGE_DIR / "data").glob("*/exported_samples.jsonl"))
                   | set((PACKAGE_DIR / "evaluation").glob("**/samples.jsonl")))
    identities, hashes, unique, metadata = set(), set(), {}, {}
    for manifest in paths:
        if manifest.parent == exclude:
            continue
        for name in (manifest.name, "summary.json", "reviews.json"):
            p = manifest.parent / name
            if p.exists():
                metadata[str(p)] = digest(p)
        for row in read_jsonl(manifest):
            if not row.get("recording_time_estimate") or not row.get("image"):
                continue
            key = (row["camera_id"], row.get("oversampling_original_id", row["id"]))
            p = manifest.parent / row["image"]
            if key not in unique or not unique[key]["path"].is_file():
                unique[key] = {"row": row, "path": p, "manifest": str(manifest)}
            if row.get("image_sha256"):
                hashes.add(row["image_sha256"])
            if row.get("source_frame_basis") == "decoded_frame_count":
                identities.add((row["camera_id"], str(Path(row["source"]).resolve()), row["source_frame"]))
    by_camera = defaultdict(list)
    for item in unique.values():
        row, p = item["row"], item["path"]
        if p.is_file():
            hashes.add(digest(p))
        item["timestamp"] = clock(row["recording_time_estimate"]).timestamp()
        by_camera[row["camera_id"]].append(item)
    for values in by_camera.values():
        values.sort(key=lambda item: item["timestamp"])
    return by_camera, identities, hashes, metadata


def groups():
    manifest = PACKAGE_DIR / "data/person_reviewed_v11_balanced_20261107/exported_samples.jsonl"
    mapping = {}
    for row in read_jsonl(manifest):
        previous = mapping.setdefault(row["split_group"], row["split"])
        if previous != row["split"]:
            raise ValueError("Current temporal group has conflicting splits")
    return mapping


def permitted(t, mapping):
    if mapping.get(group_for_time(t, seconds=600, guard=60)) != "train":
        return False
    # Cross-camera protection of the frozen diagnostic events, not just frames.
    for lo, hi in (("16:28:45", "16:30:30"), ("16:38:45", "16:41:15")):
        if clock(lo) <= t < clock(hi):
            return False
    return True


def sheets(rows, root, destination, prefix, columns=3, tile_width=640, per_page=12):
    """Unannotated full-frame previews with IDs; never draw detector proposals."""
    destination.mkdir(parents=True, exist_ok=True)
    pages = []
    for offset in range(0, len(rows), per_page):
        page_rows = rows[offset:offset + per_page]
        tile_height = tile_width * 1080 // 1920 + 48
        canvas = np.full((((len(page_rows) + columns - 1) // columns) * tile_height,
                          columns * tile_width, 3), 22, dtype=np.uint8)
        for local, row in enumerate(page_rows):
            tile = cv2.resize(read_image(root / row["image"]), (tile_width, tile_height - 48))
            x, y = (local % columns) * tile_width, (local // columns) * tile_height
            canvas[y + 48:y + tile_height, x:x + tile_width] = tile
            label = f"{offset+local:03d} {row['camera_id']} {row['recording_time_estimate'][11:19]} {row['event_group']}"
            cv2.putText(canvas, label, (x+6, y+18), cv2.FONT_HERSHEY_SIMPLEX, .40, (230, 230, 230), 1)
            cv2.putText(canvas, row["id"], (x+6, y+38), cv2.FONT_HERSHEY_SIMPLEX, .31, (170, 200, 230), 1)
        path = destination / f"{prefix}_{offset // per_page + 1:02d}.jpg"
        save_jpeg(path, canvas)
        pages.append(str(path))
    return pages


def scan(windows_path, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new scan output directory")
    spec = json.loads(Path(windows_path).read_text(encoding="utf-8-sig"))
    windows = spec["windows"]
    for window in windows:
        if clock(window["start"]) >= clock(window["end"]) or window["interval_s"] <= 0:
            raise ValueError("Invalid sample window")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    write_json(output / "requested_windows.json", spec)
    old, identities, hashes, metadata = history(output)
    time_index = {camera: [item["timestamp"] for item in values] for camera, values in old.items()}
    mapping = groups()
    cfg = load_config()
    results, reader_summaries = [], []
    all_rejections = []

    def per_camera(camera):
        selected = [w for w in windows if w["camera_id"] == camera["id"]]
        if not selected:
            return [], {}, []
        selected.sort(key=lambda w: clock(w["start"]))
        for left, right in zip(selected, selected[1:]):
            if clock(left["end"]) > clock(right["start"]):
                raise ValueError("Overlapping windows for a camera")
        origin = clock(camera["recording_start"])
        start = (clock(selected[0]["start"]) - origin).total_seconds() - camera.get("time_offset_s", 0)
        end = (clock(selected[-1]["end"]) - origin).total_seconds() - camera.get("time_offset_s", 0)
        reader = FileReader(camera["source"], analysis_fps=1, start=start, duration=end-start,
                            decode_mode="sequential")
        rows, rejected, next_time, local_hashes = [], [], {}, set()
        thumbnails = {}
        try:
            while (item := reader.next()) is not None:
                frame, source_frame, pts, _ = item
                t = recording_time(camera, pts)
                window = next((w for w in selected if clock(w["start"]) <= t < clock(w["end"])), None)
                if window is None:
                    continue
                key = window["window_id"]
                if t.timestamp() + 1e-6 < next_time.get(key, 0):
                    continue
                next_time[key] = t.timestamp() + window["interval_s"] - .05
                reason = None
                near = []
                if not permitted(t, mapping):
                    reason = "not_current_train_or_in_guard"
                times = time_index.get(camera["id"], [])
                left, right = bisect_left(times, t.timestamp()-2.0), bisect_right(times, t.timestamp()+2.0)
                if left != right:
                    reason = "within_2s_of_historical_frame"
                    near = [v["row"]["id"] for v in old[camera["id"]][left:right]]
                if (camera["id"], str(Path(camera["source"]).resolve()), source_frame) in identities:
                    reason = "historical_decoded_source_frame"
                stem = f"{camera['id']}_{t.strftime('%Y%m%dT%H%M%S')}_f{source_frame:09d}_occ12"
                # Encode in memory before creating a candidate file.
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise OSError("JPEG encoding failed")
                raw = encoded.tobytes()
                image_hash = hashlib.sha256(raw).hexdigest()
                if image_hash in hashes or image_hash in local_hashes:
                    reason = "historical_or_local_image_hash"
                # Legacy seeks and JPEG recoding can evade frame-number/hash checks.
                # This strict pixel heuristic supplements, not replaces, the time guard.
                if reason is None:
                    thumb = cv2.resize(frame, (576, 324), interpolation=cv2.INTER_AREA)
                    left, right = bisect_left(times, t.timestamp()-15), bisect_right(times, t.timestamp()+15)
                    for candidate in old.get(camera["id"], [])[left:right]:
                        p = candidate["path"]
                        if not p.is_file():
                            continue
                        if p not in thumbnails:
                            thumbnails[p] = cv2.resize(read_image(p), (576, 324), interpolation=cv2.INTER_AREA)
                        diff = cv2.absdiff(thumb, thumbnails[p])
                        if float(diff.mean()) < .35 and float((diff.max(axis=2) > 20).mean()) < .00002:
                            reason = "near_identical_pixels_within_15s"
                            near = [candidate["row"]["id"]]
                            break
                if reason:
                    rejected.append({"id": stem, "camera_id": camera["id"], "time": t.isoformat(),
                                     "reason": reason, "near_historical_ids": near})
                    continue
                relative = f"images/{stem}.jpg"
                (output / relative).write_bytes(raw)
                local_hashes.add(image_hash)
                rows.append({
                    "id": stem, "camera_id": camera["id"], "source": camera["source"],
                    "source_frame": source_frame, "source_frame_basis": reader.source_frame_basis,
                    "source_fps": reader.fps, "video_time_s": pts, "time_basis": reader.time_basis,
                    "recording_time_estimate": t.isoformat(), "width": frame.shape[1], "height": frame.shape[0],
                    "split_group": group_for_time(t, seconds=600, guard=0), "split": "train",
                    "split_assignment_status": "training_candidate_only",
                    "image": relative, "image_sha256": image_hash,
                    "event_group": window["event_group"], "window_id": key,
                    "target_conditions": window["targets"], "window_evidence": window["evidence"],
                    "condition_status": "selection_hint_not_person_level_ground_truth",
                    "proposal_generated": False, "proposal_count": 0,
                })
                if len(rows) % 10 == 0:
                    print(f"{camera['id']}: {len(rows)} new candidates, {t:%H:%M:%S}", flush=True)
        finally:
            info = {"camera_id": camera["id"], "selected_images": len(rows), "rejected": len(rejected),
                    "finish_reason": reader.finish_reason, "timestamp_fallback_frames": reader.timestamp_fallback_frames,
                    "last_processed_pts": reader.last_time, "decode_mode": reader.decode_mode}
            reader.close()
        return rows, info, rejected

    with ThreadPoolExecutor(max_workers=len(cfg["cameras"])) as pool:
        futures = [pool.submit(per_camera, camera) for camera in cfg["cameras"]]
        for future in as_completed(futures):
            rows, info, rejected = future.result()
            results.extend(rows)
            reader_summaries.append(info)
            all_rejections.extend(rejected)
    results.sort(key=lambda row: (row["event_group"], row["camera_id"], row["video_time_s"]))
    for p, before in metadata.items():
        if digest(p) != before:
            raise ValueError("Historical metadata changed during discovery: " + p)
    assert len(results) == len({row["id"] for row in results}) == len({row["image_sha256"] for row in results})
    assert all(permitted(clock(row["recording_time_estimate"]), mapping) for row in results)
    write_rows(output / "samples.jsonl", results)
    write_rows(output / "rejected.jsonl", all_rejections)
    pages = sheets(results, output, output / "contact_sheets", "scan")
    write_json(output / "scan_summary.json", {
        "status": "completed", "images": len(results), "by_camera": dict(Counter(r["camera_id"] for r in results)),
        "by_event_group": dict(Counter(r["event_group"] for r in results)),
        "rejection_reasons": dict(Counter(r["reason"] for r in all_rejections)),
        "historical_unique_ids": sum(map(len, old.values())), "historical_metadata_sha256": metadata,
        "reader_summaries": reader_summaries, "contact_sheets": pages,
        "model_inference_performed": False, "training_started": False,
        "near_duplicate_limit": "2s time exclusion plus exact decoded-frame/hash and strict pixel check within 15s; not proof of event independence.",
        "event_policy": "Shared conservative operation groups across cameras; not tracked-person identity or independent generalization events.",
    })
    print(f"Discovery complete: {len(results)} images in {output}", flush=True)


def assemble(scan_root, selection_path, output, audit_output):
    scan_root, output, audit_output = (Path(p).resolve() for p in (scan_root, output, audit_output))
    if output.exists() or audit_output.exists():
        raise FileExistsError("Use new review and assembly-audit directories")
    discovered = {r["id"]: r for r in read_jsonl(scan_root / "samples.jsonl")}
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8-sig"))
    ids = selection["selected_ids"]
    if not ids or len(ids) != len(set(ids)) or set(ids) - discovered.keys():
        raise ValueError("Nonempty unique discovered sample IDs are required")
    summary = json.loads((scan_root / "scan_summary.json").read_text(encoding="utf-8"))
    for p, before in summary["historical_metadata_sha256"].items():
        if digest(p) != before:
            raise ValueError("History changed; rerun discovery before assembling: " + p)
    for folder in ("images", "proposals/labels", "reviewed_labels", "contact_sheets"):
        (output / folder).mkdir(parents=True)
    audit_output.mkdir(parents=True)
    mapping, rows = groups(), []
    for stem in ids:
        row = dict(discovered[stem])
        if not permitted(clock(row["recording_time_estimate"]), mapping):
            raise ValueError("Forbidden sample in visual selection: " + stem)
        source, destination = scan_root / row["image"], output / row["image"]
        if digest(source) != row["image_sha256"]:
            raise ValueError("Discovery image changed: " + stem)
        shutil.copy2(source, destination)
        if digest(destination) != row["image_sha256"]:
            raise ValueError("Review image copy changed: " + stem)
        im = read_image(destination)
        if im.shape[:2] != (row["height"], row["width"]) or im.shape[:2] != (1080, 1920):
            raise ValueError("Wrong source image dimensions")
        row.update({"review_status": "pending", "annotation_locked": False,
                    "proposal_label": f"proposals/labels/{stem}.txt", "proposal_confidences": [],
                    "visual_selection_tags": selection.get("tags_by_id", {}).get(stem, []),
                    "review_reason": f"新增多事件人员样本；{row['event_group']}，{row['recording_time_estimate'][11:19]}。重点检查：{'、'.join(row['target_conditions'])}。逐人框住全部可见范围，不补画被挡身体；整帧检查，无人请明确保存无人；不确定孤立肢体请排除。",
                    "candidate_source": "sequential_video_sampling_then_visual_selection"})
        write_labels(output / row["proposal_label"], [])
        rows.append(row)
    for p, before in summary["historical_metadata_sha256"].items():
        if digest(p) != before:
            raise ValueError("Historical metadata changed during assembly: " + p)
    write_rows(output / "samples.jsonl", rows)
    write_json(output / "reviews.json", {})
    write_json(output / "summary.json", {
        "status": "completed", "images": len(rows), "by_camera": dict(Counter(r["camera_id"] for r in rows)),
        "by_split": {"train": len(rows)}, "reviewed_images": 0, "proposal_boxes": 0,
        "no_proposal_images": len(rows), "require_complete_review": True,
        "split_selection_status": "pending_reselection", "training_started": False,
        "review_instructions": "仅标新增图。可确认人员每人一个覆盖全部可见部分的紧框；不猜不可见全身，不将胳膊/腿拆成独立人员。边缘人员框到图像边界，俯视人员框住头肩和可见躯干。空白不是无人结论；无法确认、画面损坏请排除。候选属性并非真值。",
    })
    write_json(output / "sampling_plan.json", {"scan_root": str(scan_root),
        "selection_sha256": digest(selection_path), "selection": selection,
        "windows": json.loads((scan_root / "requested_windows.json").read_text(encoding="utf-8")),
        "scan_summary": summary, "use": "training_candidates_only_pending_manual_labels",
        "validation_policy": "Keep current val/test and frozen diagnostic events isolated; same-day selected training data are not independent evaluation.",
        "training_started": False})
    pages = sheets(rows, output, output / "contact_sheets", "review")
    store = ReviewStore(output)
    if store.status()["pending"] != len(rows) or any(store.item(i)["boxes"] for i in range(len(rows))):
        raise ValueError("New review must start pending with no machine/person labels")
    check = {"status": "passed_mechanical_checks", "external_review_status": "REVIEW_UNAVAILABLE",
        "dataset": str(output), "images": len(rows), "by_camera": dict(Counter(r["camera_id"] for r in rows)),
        "by_event_group": dict(Counter(r["event_group"] for r in rows)),
        "original_dimensions": [1920, 1080], "all_images_decoded": True,
        "all_labels_empty_pending_human": True, "all_current_train_with_temporal_guards": True,
        "no_historical_metadata_changed": True, "source_copy_hashes_verified": True,
        "no_unchanged_old_negative_images_returned_for_review": True,
        "export_blocked_pending_merge_and_split_audit": True, "contact_sheets": pages,
        "training_started": False}
    write_json(audit_output / "preparation_check.json", check)
    print(json.dumps(check, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    first = sub.add_parser("scan")
    first.add_argument("--windows", type=Path, required=True)
    first.add_argument("--out", type=Path, required=True)
    second = sub.add_parser("assemble")
    second.add_argument("--scan-root", type=Path, required=True)
    second.add_argument("--selection", type=Path, required=True)
    second.add_argument("--out", type=Path, required=True)
    second.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "scan":
        scan(args.windows, args.out)
    else:
        assemble(args.scan_root, args.selection, args.out, args.audit_out)
