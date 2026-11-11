from __future__ import annotations

import json
import math
import os
import random
import shutil
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from .common import load_config, read_jsonl, recording_time, start_seconds, write_json
from .detector import load_detector
from .video import FileReader, inspect_source


def group_for_time(time: datetime, seconds=600, guard=60):
    epoch = datetime(1970, 1, 1)
    absolute = (time - epoch).total_seconds()
    number = math.floor(absolute / seconds)
    phase = absolute - number * seconds
    if phase < guard or phase >= seconds - guard:
        return None
    return (epoch + timedelta(seconds=number * seconds)).isoformat()


def assign_splits(groups, seed=42):
    groups = sorted(set(groups))
    random.Random(seed).shuffle(groups)
    n = len(groups)
    n_val = min(max(1, round(n * 0.15)), n - 1) if n >= 2 else 0
    n_test = min(max(1, round(n * 0.15)), n - n_val - 1) if n >= 3 else 0
    return {group: "val" if i < n_val else "test" if i < n_val + n_test else "train"
            for i, group in enumerate(groups)}


def read_image(path):
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Cannot decode image: {path}")
    return image


def save_jpeg(path, image):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise OSError(f"Cannot encode image: {path}")
    Path(path).write_bytes(encoded.tobytes())


def validate_boxes(boxes):
    clean = []
    for box in boxes:
        if len(box) != 4:
            raise ValueError("Expected normalized [cx,cy,width,height]")
        cx, cy, w, h = [float(v) for v in box]
        if not all(math.isfinite(v) for v in (cx, cy, w, h)) or not (0 < w <= 1 and 0 < h <= 1):
            raise ValueError("Box must have finite positive width and height")
        if cx-w/2 < -1e-6 or cy-h/2 < -1e-6 or cx+w/2 > 1+1e-6 or cy+h/2 > 1+1e-6:
            raise ValueError("Box exceeds image bounds")
        clean.append([cx, cy, w, h])
    return clean


def read_labels(path):
    boxes = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5 or parts[0] != "0":
            raise ValueError(f"Expected YOLO person labels (class 0) in {path}")
        boxes.append([float(v) for v in parts[1:]])
    return validate_boxes(boxes)


def write_labels(path, boxes):
    clean = validate_boxes(boxes)
    Path(path).write_text("".join("0 " + " ".join(f"{v:.8f}" for v in b) + "\n" for b in clean), encoding="utf-8")


def run_sampling(args):
    if args.interval <= 0 or not math.isfinite(args.interval):
        raise ValueError("interval must be finite and positive")
    if args.group_seconds <= 0 or not 0 <= args.guard_seconds < args.group_seconds / 2:
        raise ValueError("Need group_seconds > 0 and 0 <= guard_seconds < group_seconds/2")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        raise ValueError("duration must be positive")
    if args.batch_size < 1 or not math.isfinite(args.start) or args.start < 0:
        raise ValueError("Need batch_size >= 1 and start >= 0")
    if args.out.exists():
        raise FileExistsError(f"Choose a new --out directory: {args.out}")
    cfg = load_config(args.config, args.cameras)
    selections, metadata = [], []
    skipped = 0
    for camera in cfg["cameras"]:
        if "://" in camera["source"]:
            raise ValueError("Training sampling requires a saved video file")
        meta = inspect_source(camera["source"])
        metadata.append({"camera_id": camera["id"], **meta})
        start = start_seconds(camera, args.start, args.start_clock)
        if args.interval < 1 / meta["fps"]:
            raise ValueError("interval must be at least one nominal source frame period")
        selections.append((camera, start))
    detector = None
    if not args.no_proposals:
        detector, predict_args = load_detector(cfg, args.device, args.imgsz)
    for folder in ("images", "proposals/labels", "reviewed_labels", "contact_sheets"):
        (args.out / folder).mkdir(parents=True, exist_ok=False)
    sampling_plan = {"interval_s": args.interval, "group_seconds": args.group_seconds,
               "decode_mode": args.decode_mode,
               "guard_seconds": args.guard_seconds, "seed": args.seed,
               "model": cfg["model"] if detector else None, "video_metadata": metadata,
               "selections": [{"camera_id": c["id"], "start_s": start, "duration_s": args.duration}
                              for c, start in selections],
               "time_basis": "filename_plus_video_pts_with_explicit_fps_fallback"}
    write_json(args.out / "sampling_plan.json", sampling_plan)
    reader = None
    reader_summaries = []
    completed, pending_images, pending_rows = [], [], []
    status = "failed"
    manifest_path = args.out / "samples.jsonl"

    def flush(manifest):
        if not pending_images:
            return
        results = detector.predict(pending_images, **predict_args) if detector else [None] * len(pending_images)
        for row, frame, result in zip(pending_rows, pending_images, results):
            proposed = []
            confidences = []
            if result is not None:
                for box, conf in zip(result.boxes.xyxyn.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                    x1, y1, x2, y2 = [min(1.0, max(0.0, float(v))) for v in box]
                    if x2 > x1 and y2 > y1:
                        proposed.append([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])
                        confidences.append(float(conf))
            save_jpeg(args.out / row["image"], frame)
            write_labels(args.out / row["proposal_label"], proposed)
            row["proposal_count"] = len(proposed)
            row["proposal_confidences"] = confidences
            row["proposal_generated"] = detector is not None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            row["blur_laplacian_variance"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            row["mean_brightness"] = float(gray.mean())
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
            completed.append(row)
        manifest.flush()
        pending_images.clear()
        pending_rows.clear()
        print(f"Sampled {len(completed)} images", flush=True)

    try:
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for camera, start in selections:
                reader = FileReader(camera["source"], analysis_fps=1 / args.interval, start=start,
                                    duration=args.duration, decode_mode=args.decode_mode)
                try:
                    while (item := reader.next()) is not None:
                        frame, source_frame, video_time, _ = item
                        wall_time = recording_time(camera, video_time)
                        group = group_for_time(wall_time, args.group_seconds, args.guard_seconds)
                        if group is None:
                            skipped += 1
                            continue
                        stem = f"{camera['id']}_{wall_time.strftime('%Y%m%dT%H%M%S')}_f{source_frame:09d}"
                        row = {"id": stem, "camera_id": camera["id"], "source": camera["source"],
                               "source_frame": source_frame, "source_fps": reader.fps, "video_time_s": video_time,
                               "time_basis": reader.time_basis, "source_frame_basis": reader.source_frame_basis,
                               "recording_time_estimate": wall_time.isoformat(), "split_group": group,
                               "image": f"images/{stem}.jpg", "width": frame.shape[1], "height": frame.shape[0],
                               "review_status": "pending", "proposal_label": f"proposals/labels/{stem}.txt"}
                        pending_images.append(frame)
                        pending_rows.append(row)
                        if len(pending_images) >= args.batch_size:
                            flush(manifest)
                finally:
                    reader_summaries.append({"camera_id": camera["id"], "finish_reason": reader.finish_reason,
                                             "timestamp_fallback_frames": reader.timestamp_fallback_frames})
                    reader.close()
                    reader = None
            flush(manifest)
        if not completed:
            raise ValueError("No frames selected: choose a longer interval or reduce --guard-seconds")
        splits = assign_splits([row["split_group"] for row in completed], args.seed)
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for row in completed:
                row["split"] = splits[row["split_group"]]
                manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
        sampling_plan.update(group_splits=splits, skipped_by_guard=skipped, sampled_images=len(completed),
                             reader_summaries=reader_summaries)
        write_json(args.out / "sampling_plan.json", sampling_plan)
        status = "completed"
    finally:
        if reader is not None:
            reader.close()
        summary = {"status": status, "images": len(completed), "reader_summaries": reader_summaries,
                   "by_camera": dict(Counter(x["camera_id"] for x in completed)),
                   "by_split": dict(Counter(x.get("split", "unassigned") for x in completed)),
                   "proposal_boxes": sum(x["proposal_count"] for x in completed),
                   "no_proposal_images": sum(x["proposal_count"] == 0 for x in completed),
                   "reviewed_images": 0, "training_started": False,
                   "note": "Machine proposals only; empty proposals do NOT confirm a negative image."}
        write_json(args.out / "summary.json", summary)
    make_contact_sheets(args.out, completed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def make_contact_sheets(dataset, rows):
    for camera in sorted({r["camera_id"] for r in rows}):
        selected = [r for r in rows if r["camera_id"] == camera]
        selected = [selected[i] for i in np.linspace(0, len(selected)-1, min(12, len(selected)), dtype=int)]
        tiles = []
        for row in selected:
            frame = read_image(dataset / row["image"])
            for box in read_labels(dataset / row["proposal_label"]):
                cx, cy, w, h = box
                p1 = (int((cx-w/2)*frame.shape[1]), int((cy-h/2)*frame.shape[0]))
                p2 = (int((cx+w/2)*frame.shape[1]), int((cy+h/2)*frame.shape[0]))
                cv2.rectangle(frame, p1, p2, (0, 255, 255), 2)
            tile = np.zeros((302, 480, 3), np.uint8)
            tile[:270] = cv2.resize(frame, (480, 270))
            text = f"{camera} {row['recording_time_estimate'][11:]} {row['split']} n={row['proposal_count']}"
            cv2.putText(tile, text, (5, 291), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles.append(tile)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)])
        save_jpeg(dataset / "contact_sheets" / f"{camera}_proposals.jpg", sheet)
