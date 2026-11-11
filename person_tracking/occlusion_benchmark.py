"""Prepare, human-review, freeze and evaluate a stratified person holdout."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import csv
import json
import math
import os
from pathlib import Path
import shutil
import time
from urllib.parse import parse_qs, urlparse

from .common import DEFAULT_CONFIG, PACKAGE_DIR, load_config, recording_time, write_json
from .dataset import read_image, save_jpeg
from .occlusion import OcclusionStore, atomic_json, match_frame, recall_report, sha256, validate_objects
from .occlusion_history import (allowed_at, assert_isolation, blocked_intervals, evidence_unchanged,
                               history_inventory, iso, subtract_intervals, timestamp)
from .video import FileReader, inspect_source


def protocol_for(mode, guard):
    return {"schema_version": 1, "mode": mode, "guard_seconds": guard,
            "independence_scope": "same_day_temporal_gap_development" if mode == "temporal_gap" else "unseen_temporal_group",
            "frame_size": [1280, 720], "imgsz": 1280, "confidence": 0.10, "nms_iou": 0.60,
            "match_iou": 0.50, "ignore_prediction_ioa": 0.50, "small_visible_height_720p": 64,
            "target_recall": 0.85, "min_people_per_stratum": 30, "min_clips_per_stratum": 3,
            "occlusion_bands": {"none": "0%", "light": "(0,25]%", "moderate": "(25,50]%",
                                "heavy": "(50,75]%", "extreme": "(75,100)%", "unknown": "not estimable"},
            "box_policy": "One tight envelope of all visible parts of one identifiable person; no amodal body guesses.",
            "ratio_policy": "Human-estimated occluded fraction of full projected body; uncertain values use unknown.",
            "sampling_policy": "Clock-based uniform sampling, independent of detector output; no proposals shown.",
            "inference_policy": "Resize same-aspect original frame to 1280x720, detector only; no tracking, ROI or display suppression.",
            "evaluation_only": True, "model_proposals_used_as_truth": False,
            "fully_invisible_people": "Not included in image Recall; require continuous event/multicamera evidence."}


def plan(args, history=None):
    history = history or history_inventory(args.history_root, args.runs_root,
                                           evaluation_root=getattr(args, "evaluation_root", None))
    cfg = load_config(args.config)
    blocks = blocked_intervals(history, args.mode, args.guard_seconds)
    cameras = {c["id"]: c for c in cfg["cameras"]}
    clips = json.loads(args.clips.read_text(encoding="utf8"))["clips"] if args.clips else [
        {"id": c["id"] + "_available", "camera_id": c["id"]} for c in cfg["cameras"]]
    if len({c["id"] for c in clips}) != len(clips):
        raise ValueError("Duplicate clip IDs")
    results = []
    metadata = {cid: inspect_source(c["source"]) for cid, c in cameras.items()}
    for clip in clips:
        camera = cameras[clip["camera_id"]]
        origin = timestamp(recording_time(camera, 0).isoformat())
        meta = metadata[camera["id"]]
        start = timestamp(clip["start"]) if "start" in clip else origin
        end = timestamp(clip["end"]) if "end" in clip else origin + meta["duration_s"]
        if start < origin or end <= start or end > origin + meta["duration_s"] + 0.1:
            raise ValueError(f"Clip exceeds nominal recording bounds: {clip['id']}")
        clip_blocks = list(blocks)
        if args.mode == "strict_group":
            group = math.floor(start / 600) * 600
            while group < end:
                clip_blocks.extend([[group, group + 60], [group + 540, group + 600]])
                group += 600
        windows = subtract_intervals(start, end, clip_blocks)
        results.append({"clip_id": clip["id"], "evaluation_group": clip.get("evaluation_group", clip["id"]),
                        "camera": camera, "metadata": meta,
                        "requested_start": iso(start), "requested_end": iso(end),
                        "allowed_windows": [{"start": iso(a), "end": iso(b), "duration_s": b-a} for a, b in windows],
                        "available_seconds_nominal": sum(b-a for a, b in windows)})
    return {"created_at": datetime.now().isoformat(), "protocol": protocol_for(args.mode, args.guard_seconds),
            "clips": results, "history": history,
            "warning": "Container duration is nominal. Actual frame PTS and EOF are rechecked during sequential decoding."}


def sample(args):
    import cv2
    if args.out.exists():
        raise FileExistsError(f"Choose a new --out: {args.out}")
    if not math.isfinite(args.interval) or args.interval <= 0:
        raise ValueError("--interval must be finite and positive")
    prepared = plan(args)
    if not any(c["allowed_windows"] for c in prepared["clips"]):
        raise ValueError("No eligible windows under this historical exclusion policy")
    root = args.out.resolve()
    (root / "images").mkdir(parents=True)
    (root / "contact_sheets").mkdir()
    write_json(root / "sampling_plan.json", prepared)
    write_json(root / "protocol.json", prepared["protocol"])
    write_json(root / "history_audit.json", prepared["history"])
    write_json(root / "reviews.json", {})
    summary = {"kind": "occlusion_holdout", "status": "sampling", "evaluation_only": True,
               "training_started": False, "human_review_required": True}
    write_json(root / "summary.json", summary)
    rows, reader_records = [], []
    try:
        # Decode each original file once, sequentially, including decoder
        # context before the earliest requested window. Never seek H264 cuts.
        for cid in sorted({c["camera"]["id"] for c in prepared["clips"]}):
            clips = [c for c in prepared["clips"] if c["camera"]["id"] == cid and c["allowed_windows"]]
            if not clips:
                continue
            camera = clips[0]["camera"]
            evaluation_groups = {c["clip_id"]: c["evaluation_group"] for c in clips}
            origin = timestamp(recording_time(camera, 0).isoformat())
            windows = [(timestamp(w["start"]), timestamp(w["end"]), c["clip_id"])
                       for c in clips for w in c["allowed_windows"]]
            for previous, following in zip(sorted(windows), sorted(windows)[1:]):
                if previous[1] > following[0]:
                    raise ValueError("Overlapping clips would duplicate evaluation frames")
            start, end = min(w[0] for w in windows), max(w[1] for w in windows)
            reader = FileReader(camera["source"], analysis_fps=1/args.interval,
                                start=start-origin, duration=end-start, decode_mode="sequential")
            last_report = time.monotonic()
            try:
                while (item := reader.next()) is not None:
                    frame, source_frame, video_time, _ = item
                    when = recording_time(camera, video_time)
                    wall = timestamp(when.isoformat())
                    clip_id = next((name for a, b, name in windows if a <= wall < b), None)
                    if clip_id is None or not allowed_at(wall, prepared["history"], args.mode, args.guard_seconds):
                        continue
                    if reader.time_basis != "file_pts":
                        raise ValueError("Independent timestamp isolation requires media PTS, not nominal FPS fallback")
                    if abs(frame.shape[1]/frame.shape[0] - 16/9) > 0.02:
                        raise ValueError("720p simulation expects a 16:9 frame with the same field of view")
                    sid = f"{cid}_{when.strftime('%Y%m%dT%H%M%S')}_f{source_frame:09d}_holdout"
                    image = root / "images" / (sid + ".jpg")
                    save_jpeg(image, frame)
                    rows.append({"id": sid, "camera_id": cid, "clip_id": clip_id,
                                 "evaluation_group": evaluation_groups[clip_id],
                                 "source": camera["source"], "source_frame": source_frame, "video_time_s": video_time,
                                 "time_basis": reader.time_basis, "source_frame_basis": reader.source_frame_basis,
                                 "recording_time_estimate": when.isoformat(), "width": frame.shape[1], "height": frame.shape[0],
                                 "image": image.relative_to(root).as_posix(), "image_sha256": sha256(image),
                                 "split": "occlusion_holdout", "evaluation_only": True})
                    if time.monotonic() - last_report > 15:
                        print(f"{cid}: sampled {len(rows)} total, clock {when.isoformat()}", flush=True)
                        last_report = time.monotonic()
            finally:
                reader_records.append({"camera_id": cid, "finish_reason": reader.finish_reason,
                                       "timestamp_fallback_frames": reader.timestamp_fallback_frames,
                                       "last_pts_s": reader.last_time})
                reader.close()
            print(f"{cid}: {sum(r['camera_id']==cid for r in rows)} frames saved", flush=True)
        if not rows:
            raise ValueError("No eligible frames decoded (nominal duration may exceed actual PTS EOF)")
        assert_isolation(rows, prepared["history"], prepared["protocol"])
        rows.sort(key=lambda r: (r["clip_id"], r["video_time_s"]))
        (root / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf8")
        # Contact sheets are unannotated, never detector-filtered candidates.
        import numpy as np
        for clip_id in sorted({r["clip_id"] for r in rows}):
            chosen = [r for r in rows if r["clip_id"] == clip_id]
            chosen = [chosen[i] for i in np.linspace(0, len(chosen)-1, min(12, len(chosen)), dtype=int)]
            tiles = []
            for row in chosen:
                tile = np.zeros((204, 320, 3), dtype=np.uint8)
                tile[:180] = cv2.resize(read_image(root / row["image"]), (320, 180))
                cv2.putText(tile, f"{row['camera_id']} {row['recording_time_estimate'][11:19]}",
                            (5, 197), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
                tiles.append(tile)
            while len(tiles) % 4:
                tiles.append(np.zeros_like(tiles[0]))
            save_jpeg(root / "contact_sheets" / f"{clip_id}.jpg", np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]))
        summary.update(status="completed", images=len(rows), interval_seconds=args.interval,
                       by_camera=dict(Counter(r["camera_id"] for r in rows)), by_clip=dict(Counter(r["clip_id"] for r in rows)),
                       readers=reader_records, independence_scope=prepared["protocol"]["independence_scope"])
    except BaseException as exc:
        summary.update(status="failed", error=repr(exc))
        raise
    finally:
        write_json(root / "summary.json", summary)
    print(json.dumps(OcclusionStore(root).status(), ensure_ascii=False, indent=2), flush=True)


def make_server(dataset, port=8773):
    store = OcclusionStore(dataset)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, status, value, content_type="application/json; charset=utf-8"):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode("utf8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path == "/":
                    self.respond(200, (PACKAGE_DIR / "occlusion_review.html").read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/items":
                    self.respond(200, {"items": store.items(), "status": store.status(), "protocol": store.protocol})
                elif url.path in ("/api/item", "/api/image"):
                    index = int(parse_qs(url.query)["index"][0])
                    self.respond(200, store.item(index)) if url.path == "/api/item" else self.respond(
                        200, (store.root / store.row(index)["image"]).read_bytes(), "image/jpeg")
                else:
                    self.respond(404, {"error": "Not found"})
            except (ValueError, KeyError, IndexError, OSError) as exc:
                self.respond(400, {"error": str(exc)})

        def do_POST(self):
            if self.path != "/api/review":
                self.respond(404, {"error": "Not found"})
                return
            origin = self.headers.get("Origin")
            if origin and origin != f"http://127.0.0.1:{self.server.server_port}":
                self.respond(403, {"error": "Use the local review page"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 2_000_000:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(length))
                saved = store.save(body["index"], body["status"], body.get("objects"), body.get("ignore_regions"),
                                   body.get("confirm_empty", False), body.get("reason", ""))
                self.respond(200, {"saved": saved, "status": store.status()})
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.respond(400, {"error": str(exc)})
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def verify_images(store):
    for row in store.rows:
        if sha256(store.root / row["image"]) != row["image_sha256"]:
            raise ValueError(f"Holdout image changed: {row['id']}")


def check_reviews(store):
    if store.status()["pending"]:
        raise ValueError("Complete all pending human annotations before freezing/evaluating")
    for row in store.rows:
        rev = store.reviews[row["id"]]
        if rev.get("review_source") != "human_occlusion_review":
            raise ValueError("Human reviewed ground truth required")
        if rev["status"] == "reviewed":
            validate_objects(rev["objects"])
            from .dataset import validate_boxes
            validate_boxes(rev["ignore_regions"])
            if not rev["objects"] and not rev.get("confirmed_no_evaluable_person"):
                raise ValueError("Unconfirmed empty label")
        elif rev["status"] == "excluded":
            if not rev.get("reason"):
                raise ValueError("Excluded frame without reason")
        else:
            raise ValueError("Unknown review status")
    if not any(r["status"] == "reviewed" for r in store.reviews.values()):
        raise ValueError("No evaluable reviewed frames")


def freeze(dataset, output):
    store, output = OcclusionStore(dataset), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new --out: {output}")
    check_reviews(store)
    verify_images(store)
    history = json.loads((store.root / "history_audit.json").read_text(encoding="utf8"))
    current = history_inventory(history["data_root"], history["runs_root"],
                                evaluation_root=history.get("evaluation_root"),
                                ignore_evaluation_datasets=[store.root])
    evidence_unchanged(history, current)
    assert_isolation(store.rows, current, store.protocol)
    output.mkdir(parents=True)
    (output / "images").mkdir()
    for row in store.rows:
        shutil.copy2(store.root / row["image"], output / row["image"])
    for name in ("summary.json", "samples.jsonl", "reviews.json", "protocol.json", "sampling_plan.json"):
        shutil.copy2(store.root / name, output / name)
    write_json(output / "history_audit.json", current)
    files = ["summary.json", "samples.jsonl", "reviews.json", "protocol.json", "sampling_plan.json", "history_audit.json"]
    audit = {"status": "frozen", "source": str(store.root), "created_at": datetime.now().isoformat(),
             "files_sha256": {name: sha256(output / name) for name in files},
             "evaluation_only": True, "status_at_freeze": store.status(), "test_split_used": False}
    write_json(output / "freeze.json", audit)
    return audit


def verify_frozen(dataset, *, development_recheck=False):
    store = OcclusionStore(dataset)
    if not store.frozen:
        raise ValueError("Freeze human annotations before evaluating model weights")
    audit = json.loads((store.root / "freeze.json").read_text(encoding="utf8"))
    for name, expected in audit["files_sha256"].items():
        if sha256(store.root / name) != expected:
            raise ValueError(f"Frozen evidence changed: {name}")
    check_reviews(store)
    verify_images(store)
    history = json.loads((store.root / "history_audit.json").read_text(encoding="utf8"))
    if development_recheck:
        # Reuse immutable truth for a declared development diagnostic. Later
        # inference/model comparisons do not make this an unseen holdout again.
        assert_isolation(store.rows, history, store.protocol)
        return store
    current = history_inventory(history["data_root"], history["runs_root"],
                                evaluation_root=history.get("evaluation_root"),
                                ignore_evaluation_datasets=[store.root])
    evidence_unchanged(history, current)
    assert_isolation(store.rows, current, store.protocol)
    return store


def write_report(output, report):
    rows = report["strata"]
    with (output / "recall_by_group.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[k for k in rows[0] if k != "wilson95_descriptive"])
        writer.writeheader()
        writer.writerows({k: v for k, v in r.items() if k != "wilson95_descriptive"} for r in rows)
    labels = {"all": "全部", "camera": "相机", "occlusion": "遮挡程度", "occluder": "遮挡物",
              "distance": "远近", "size": "720p可见框高", "small_target": "小目标", "truncated": "画面截断",
              "none": "无遮挡", "light": "轻度(0,25%]", "moderate": "中度(25%,50%]",
              "heavy": "重度(50%,75%]", "extreme": "极重度(75%,100%)", "unknown": "无法判断",
              "load": "吊物/工件", "structure": "结构/设备", "person": "其他人员", "other": "其他",
              "far": "远处", "near": "非远处", "visible_small": "可见框高<64px",
              "far_and_visible_small": "远处且可见框高<64px"}
    coverage_labels = {"not_verified": "未验证", "limited": "样本不足", "supported_counts": "计数达到预设下限"}
    lines = ["# 人员遮挡分组验证", "", f"验证范围：`{report['independence_scope']}`。", "",
             f"模型：`{report['model']}`；置信度={report['protocol']['confidence']}，匹配IoU={report['protocol']['match_iou']}。", "",
             "同日时段验证不能证明新日期/新作业泛化。真值为人工可见人员框；完全不可见人员不计入本次逐帧 Recall。", "",
             f"有效帧 {report['review_status']['reviewed']}；排除 {report['review_status']['excluded']}。", "",
             "| 分组 | 类别 | TP | FN | 真值框 | 正样本帧 | 时间片 | Recall | 覆盖 |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        value = "未验证" if row["recall"] is None else f"{row['recall']:.2%}"
        lines.append(f"| {labels.get(row['category'], row['category'])} | {labels.get(row['stratum'], row['stratum'])} | {row['tp']} | {row['fn']} | {row['ground_truth_people']} | {row['positive_images']} | {row['positive_clips']} | {value} | {coverage_labels[row['coverage']]} |")
    lines += ["", "分遮挡物分组允许重叠，不能求和。far_and_visible_small 同时要求人工确认远处且720p可见框高<64像素。",
              "", "limited 表示人数框不足30或时间片不足3；相邻帧并非独立事件，即使supported_counts也不等于现场验收通过。",
              "", "冻结后首次比较可作留出验证；据此调模型或阈值后，该集合转为开发验证集，正式验收须另采新录像。"]
    if report.get("development_recheck"):
        lines += ["", "本轮显式使用 development_recheck：复用原冻结真值和推理协议，仅作开发诊断；未重新认证当前历史隔离，不属于独立验证。"]
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf8")


def evaluate(args):
    import cv2
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PACKAGE_DIR / ".cache/ultralytics"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    from ultralytics import YOLO, __version__ as ultralytics_version
    development_recheck = bool(getattr(args, "development_recheck", False))
    store = verify_frozen(args.dataset, development_recheck=development_recheck)
    if args.out.exists():
        raise FileExistsError(f"Choose a new --out: {args.out}")
    if not args.weights or any(not p.is_file() for p in args.weights):
        raise ValueError("All requested model checkpoints must exist")
    root, protocol = args.out.resolve(), store.protocol
    root.mkdir(parents=True)
    models = []
    for index, weights in enumerate(args.weights):
        output = root / f"model_{index+1}_{weights.parent.parent.name}"
        output.mkdir()
        model = YOLO(str(weights.resolve()))
        if model.names[0].lower() != "person":
            raise ValueError("Model class 0 must be person")
        frames = []
        model_hash = sha256(weights)
        with (output / "predictions.jsonl").open("w", encoding="utf8") as predictions_stream:
            for row in store.rows:
                rev = store.reviews[row["id"]]
                if rev["status"] != "reviewed":
                    continue
                frame = cv2.resize(read_image(store.root / row["image"]), tuple(protocol["frame_size"]))
                result = model.predict(frame, imgsz=protocol["imgsz"], conf=protocol["confidence"],
                                       iou=protocol["nms_iou"], classes=[0], device=args.device,
                                       half=args.device != "cpu", max_det=300, verbose=False)[0]
                predictions = []
                for xyxy, score in zip(result.boxes.xyxyn.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                    x1, y1, x2, y2 = [min(1., max(0., float(v))) for v in xyxy]
                    if x2 > x1 and y2 > y1:
                        predictions.append({"box": [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], "confidence": score})
                predictions_stream.write(json.dumps({"id": row["id"], "predictions": predictions})+"\n")
                result = match_frame(rev["objects"], predictions, rev["ignore_regions"], protocol["confidence"],
                                     protocol["match_iou"], protocol["ignore_prediction_ioa"])
                frames.append({"id": row["id"], "camera_id": row["camera_id"], "clip_id": row["clip_id"],
                               "evaluation_group": row.get("evaluation_group", row["clip_id"]), **result})
        if sha256(weights) != model_hash:
            raise ValueError("Model checkpoint changed during evaluation")
        verify_frozen(args.dataset, development_recheck=development_recheck)
        report = {**recall_report(frames, protocol), "model": str(weights.resolve()), "model_sha256": model_hash,
                  "dataset": str(store.root), "freeze_sha256": sha256(store.root / "freeze.json"),
                  "protocol": protocol,
                  "independence_scope": "development_recheck_after_prior_model_comparisons" if development_recheck else protocol["independence_scope"],
                  "development_recheck": development_recheck,
                  "current_history_rechecked": not development_recheck,
                  "review_status": store.status(), "ultralytics": ultralytics_version,
                  "exclusions": [{"id": sid, "reason": rev["reason"]} for sid, rev in store.reviews.items() if rev["status"] == "excluded"],
                  "evaluation_only": True, "test_split_used": False, "training_started": False}
        write_json(output / "metrics.json", report)
        (output / "per_person.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in frames), encoding="utf8")
        write_report(output, report)
        models.append({"model": report["model"], "metrics": str(output / "metrics.json"), "totals": report["totals"]})
        print(json.dumps(models[-1], ensure_ascii=False, indent=2), flush=True)
    write_json(root / "comparison.json", models)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "sample"):
        p = sub.add_parser(command)
        p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        p.add_argument("--clips", type=Path, help="JSON clips: id,camera_id,start,end in ISO local clock")
        p.add_argument("--history-root", type=Path, default=PACKAGE_DIR / "data")
        p.add_argument("--runs-root", type=Path, default=PACKAGE_DIR / "runs")
        p.add_argument("--evaluation-root", type=Path, default=PACKAGE_DIR / "evaluation",
                       help="Completed holdout comparisons to exclude from new independent sampling")
        p.add_argument("--mode", choices=("strict_group", "temporal_gap"), default="strict_group")
        p.add_argument("--guard-seconds", type=float, default=30)
        p.add_argument("--out", type=Path, required=True)
        if command == "sample":
            p.add_argument("--interval", type=float, default=1)
    for command in ("review", "status", "freeze", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--dataset", type=Path, required=True)
        if command == "review":
            p.add_argument("--port", type=int, default=8773)
        if command in ("freeze", "evaluate"):
            p.add_argument("--out", type=Path, required=True)
        if command == "evaluate":
            p.add_argument("--weights", type=Path, nargs="+", required=True)
            p.add_argument("--device", default="cpu")
            p.add_argument("--development-recheck", action="store_true",
                           help="Reuse immutable frozen labels as development diagnostics; do not certify current temporal independence")
    args = parser.parse_args(argv)
    if args.command == "plan":
        if args.out.exists():
            raise FileExistsError(f"Choose a new --out: {args.out}")
        result = plan(args)
        write_json(args.out, result)
        print(json.dumps({"plan": str(args.out), "clips": [
            {k: c[k] for k in ("clip_id", "allowed_windows", "available_seconds_nominal")} for c in result["clips"]]}, ensure_ascii=False, indent=2))
    elif args.command == "sample":
        sample(args)
    elif args.command == "status":
        print(json.dumps(OcclusionStore(args.dataset).status(), ensure_ascii=False, indent=2))
    elif args.command == "review":
        server = make_server(args.dataset, args.port)
        print(f"Occlusion review: http://127.0.0.1:{server.server_port}/?status=pending", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    elif args.command == "freeze":
        print(json.dumps(freeze(args.dataset, args.out), ensure_ascii=False, indent=2))
    elif args.command == "evaluate":
        evaluate(args)


if __name__ == "__main__":
    main()
