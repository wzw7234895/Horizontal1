"""Prepare, review, and export B3/B4 detector data from the crane videos."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import threading
from collections import Counter
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from .b3b4 import DEFAULT_COLORS, class_names_for_colors
from .common import PACKAGE_DIR, read_jsonl, write_json
from .dataset import read_labels, run_sampling


DEFAULT_VIDEO_ROOT = PACKAGE_DIR.parent.parent
DEFAULT_PERSON_WEIGHTS = (PACKAGE_DIR / "runs" / "train" /
                         "person_yolo11n_static_v13_hard_merged_20261108_r2" /
                         "weights" / "best.pt")
DEFAULT_OUTPUT = PACKAGE_DIR / "data" / "b3b4_review_20261110"
DEFAULT_POLICY = PACKAGE_DIR / "configs" / "b3b4_attributes.json"


def _taxonomy(policy_path):
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8-sig"))
    colors = policy.get("candidate_label_colors", list(DEFAULT_COLORS))
    if (not isinstance(colors, list) or not colors
            or any(not isinstance(color, str) or not color.strip() for color in colors)):
        raise ValueError("candidate_label_colors must contain nonempty color names")
    colors = [color.strip().lower() for color in colors]
    if len(set(colors)) != len(colors):
        raise ValueError("candidate_label_colors must be unique")
    classes = class_names_for_colors(colors, colors)
    labels = [{"name": "head_no_helmet", "capability": "B3", "part": "head",
               "meaning": "visible head without a safety helmet"}]
    labels.extend({"name": f"helmet_{color}", "capability": "B3", "part": "helmet",
                   "color": color, "meaning": "worn safety helmet color"} for color in colors)
    labels.extend({"name": f"torso_{color}", "capability": "B4", "part": "torso",
                   "color": color, "meaning": "visible workwear torso color"} for color in colors)
    return {"schema_version": "1.0", "classes": classes, "labels": labels,
            "candidate_colors": colors, "customer_policy_confirmed": False,
            "policy_file": str(Path(policy_path).resolve())}


def prepare_review_dataset(*, output, video_root=DEFAULT_VIDEO_ROOT, config=None,
                           weights=DEFAULT_PERSON_WEIGHTS, policy=DEFAULT_POLICY,
                           camera_ids=None, interval=30.0, start=0.0, duration=None,
                           group_seconds=600, guard_seconds=60, seed=42,
                           device="0", image_size=1280, batch_size=4,
                           decode_mode="sequential"):
    output = Path(output).resolve()
    video_root = Path(video_root).resolve()
    config = Path(config or PACKAGE_DIR / "configs" / "cameras.json").resolve()
    weights = Path(weights).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if not weights.is_file():
        raise FileNotFoundError(f"Person proposal weights not found: {weights}")
    source_config = json.loads(config.read_text(encoding="utf-8-sig"))
    selected = source_config["cameras"]
    if camera_ids:
        unknown = set(camera_ids) - {camera["id"] for camera in selected}
        if unknown:
            raise ValueError(f"Unknown camera IDs: {sorted(unknown)}")
        selected = [camera for camera in selected if camera["id"] in camera_ids]
    for camera in selected:
        source = video_root / Path(camera["source"]).name
        if not source.is_file():
            raise FileNotFoundError(f"{camera['id']} source video not found: {source}")
        camera["source"] = str(source)
        camera["roi"] = []
    source_config["cameras"] = selected
    source_config["project_root"] = str(video_root)
    source_config["model"] = str(weights)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="b3b4-camera-config-") as temporary:
        temporary_config = Path(temporary) / "cameras.json"
        write_json(temporary_config, source_config)
        sample_args = SimpleNamespace(
            config=temporary_config,
            cameras=None,
            out=output,
            interval=interval,
            start=start,
            start_clock=None,
            duration=duration,
            group_seconds=group_seconds,
            guard_seconds=guard_seconds,
            seed=seed,
            device=device,
            imgsz=image_size,
            batch_size=batch_size,
            no_proposals=False,
            decode_mode=decode_mode,
        )
        run_sampling(sample_args)

    taxonomy = _taxonomy(policy)
    rows = read_jsonl(output / "samples.jsonl")
    annotations = {row["id"]: {"status": "pending", "frame_complete": False, "boxes": []}
                   for row in rows}
    write_json(output / "b3b4_taxonomy.json", taxonomy)
    write_json(output / "b3b4_annotations.json", annotations)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({"b3b4_samples": len(rows), "b3b4_reviewed": 0,
                    "customer_policy_confirmed": False,
                    "training_started": False,
                    "note": "Person boxes are machine proposals only. B3/B4 boxes and attributes require human review."})
    write_json(summary_path, summary)
    return summary


def _validate_boxes(boxes, width, height, class_names):
    if not isinstance(boxes, list):
        raise ValueError("boxes must be a list")
    clean = []
    for item in boxes:
        if not isinstance(item, dict) or item.get("class_name") not in class_names:
            raise ValueError("Each box needs a class_name from b3b4_taxonomy.json")
        xyxy = item.get("xyxy")
        if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
            raise ValueError("Each box needs pixel xyxy coordinates")
        try:
            x1, y1, x2, y2 = [float(value) for value in xyxy]
        except (TypeError, ValueError):
            raise ValueError("Box coordinates must be numeric") from None
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError("Box coordinates must be finite")
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Boxes must have positive area and stay inside the image")
        clean.append({"class_name": item["class_name"],
                      "xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]})
    return clean


class B3B4ReviewStore:
    def __init__(self, dataset):
        self.root = Path(dataset).resolve()
        summary_path = self.root / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        self.summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if self.summary.get("status") != "completed":
            raise ValueError("Sampling is not complete")
        self.rows = read_jsonl(self.root / "samples.jsonl")
        self.taxonomy = json.loads((self.root / "b3b4_taxonomy.json").read_text(encoding="utf-8"))
        self.class_names = set(self.taxonomy["classes"])
        self.annotations_path = self.root / "b3b4_annotations.json"
        self.annotations = json.loads(self.annotations_path.read_text(encoding="utf-8"))
        self.lock = threading.RLock()

    def row(self, index):
        if type(index) is not int or not 0 <= index < len(self.rows):
            raise ValueError("Invalid sample index")
        return self.rows[index]

    def items(self):
        with self.lock:
            return [{"index": index, "id": row["id"], "camera_id": row["camera_id"],
                     "time": row["recording_time_estimate"], "split": row["split"],
                     "proposal_count": row["proposal_count"],
                     "status": self.annotations[row["id"]]["status"]}
                    for index, row in enumerate(self.rows)]

    def item(self, index):
        with self.lock:
            row = self.row(index)
            proposal_path = self.root / row["proposal_label"]
            proposals = []
            for cx, cy, width, height in read_labels(proposal_path):
                x1 = (cx - width / 2) * row["width"]
                y1 = (cy - height / 2) * row["height"]
                x2 = (cx + width / 2) * row["width"]
                y2 = (cy + height / 2) * row["height"]
                proposals.append([x1, y1, x2, y2])
            return {**row, "index": index, "proposals_xyxy": proposals,
                    "annotation": self.annotations[row["id"]],
                    "taxonomy": self.taxonomy}

    def save(self, index, status, frame_complete, boxes):
        if status not in {"reviewed", "excluded"}:
            raise ValueError("status must be reviewed or excluded")
        with self.lock:
            row = self.row(index)
            clean = _validate_boxes(boxes, row["width"], row["height"], self.class_names)
            if status == "reviewed" and frame_complete is not True:
                raise ValueError("Confirm the whole frame is reviewed, or exclude it")
            self.annotations[row["id"]] = {
                "status": status,
                "frame_complete": status == "reviewed",
                "boxes": clean if status == "reviewed" else [],
                "reviewed_at": datetime.now().isoformat(timespec="seconds"),
                "review_source": "b3b4_local_review_interface",
            }
            temporary = self.annotations_path.with_suffix(".tmp")
            write_json(temporary, self.annotations)
            temporary.replace(self.annotations_path)
            return self.annotations[row["id"]]

    def status(self):
        with self.lock:
            counts = Counter(value["status"] for value in self.annotations.values())
            return {"dataset": str(self.root), "total": len(self.rows),
                    "pending": counts["pending"], "reviewed": counts["reviewed"],
                    "excluded": counts["excluded"],
                    "reviewed_by_camera": dict(Counter(
                        row["camera_id"] for row in self.rows
                        if self.annotations[row["id"]]["status"] == "reviewed")),
                    "class_counts": dict(Counter(
                        box["class_name"] for annotation in self.annotations.values()
                        for box in annotation.get("boxes", [])))}


def export_yolo_dataset(dataset, output):
    store = B3B4ReviewStore(dataset)
    status = store.status()
    if status["pending"]:
        raise ValueError(f"Complete or exclude all pending frames first; pending={status['pending']}")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new export directory: {output}")
    classes = store.taxonomy["classes"]
    class_ids = {name: index for index, name in enumerate(classes)}
    counts = {split: Counter() for split in ("train", "val", "test")}
    exported = Counter()
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    manifest_path = output / "exported_samples.jsonl"
    for row in store.rows:
        annotation = store.annotations[row["id"]]
        if annotation["status"] != "reviewed":
            continue
        split = row["split"]
        if split not in counts:
            raise ValueError(f"Invalid sample split: {split}")
        image_name = Path(row["image"]).name
        shutil.copy2(store.root / row["image"], output / "images" / split / image_name)
        lines = []
        for box in annotation["boxes"]:
            x1, y1, x2, y2 = box["xyxy"]
            cx, cy = (x1 + x2) / (2 * row["width"]), (y1 + y2) / (2 * row["height"])
            width, height = (x2 - x1) / row["width"], (y2 - y1) / row["height"]
            class_id = class_ids[box["class_name"]]
            lines.append(f"{class_id} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}")
            counts[split][box["class_name"]] += 1
        (output / "labels" / split / f"{Path(image_name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        exported[split] += 1
    with manifest_path.open("w", encoding="utf-8") as stream:
        for row in store.rows:
            if store.annotations[row["id"]]["status"] == "reviewed":
                stream.write(json.dumps({"id": row["id"], "camera_id": row["camera_id"],
                    "recording_time_estimate": row["recording_time_estimate"],
                    "split_group": row["split_group"], "split": row["split"]},
                    ensure_ascii=False) + "\n")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML in the crane-person environment to export data.yaml") from exc
    data_yaml = {"path": output.as_posix(), "train": "images/train", "val": "images/val",
                 "test": "images/test", "names": classes}
    (output / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False),
                                      encoding="utf-8")
    missing_train = [name for name in classes if counts["train"][name] == 0]
    missing_val = [name for name in classes if counts["val"][name] == 0]
    summary = {"source_dataset": str(store.root), "images_exported_by_split": dict(exported),
               "boxes_by_split": {split: dict(counts[split]) for split in counts},
               "missing_classes_in_train": missing_train, "missing_classes_in_val": missing_val,
               "training_ready": bool(exported["train"] and exported["val"]
                                       and not missing_train and not missing_val),
               "training_started": False,
               "evaluation_note": "These four videos are a same-day development corpus, not independent site acceptance."}
    write_json(output / "audit.json", summary)
    return summary


def make_server(dataset, port=8773):
    store = B3B4ReviewStore(dataset)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, status_code, body, content_type="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path == "/":
                    page = Path(__file__).parent / "b3b4_review.html"
                    self.respond(200, page.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/items":
                    self.respond(200, {"items": store.items(), "status": store.status()})
                elif url.path == "/api/item":
                    index = int(parse_qs(url.query)["index"][0])
                    self.respond(200, store.item(index))
                elif url.path == "/api/image":
                    index = int(parse_qs(url.query)["index"][0])
                    row = store.row(index)
                    self.respond(200, (store.root / row["image"]).read_bytes(), "image/jpeg")
                else:
                    self.respond(404, {"error": "Not found"})
            except (ValueError, KeyError, IndexError, OSError) as exc:
                self.respond(400, {"error": str(exc)})

        def do_POST(self):
            if self.path != "/api/save":
                self.respond(404, {"error": "Not found"})
                return
            origin = self.headers.get("Origin")
            if origin and origin != f"http://127.0.0.1:{self.server.server_port}":
                self.respond(403, {"error": "Use the local review page"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 2_000_000:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(size))
                saved = store.save(body["index"], body["status"], body.get("frame_complete"),
                                   body.get("boxes", []))
                self.respond(200, {"saved": saved, "status": store.status()})
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.respond(400, {"error": str(exc)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def _serve(args):
    server = make_server(args.dataset, args.port)
    print(f"B3/B4 review page: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="B3/B4 attribute dataset tools")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Sample videos into a B3/B4 human-review dataset")
    prepare.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    prepare.add_argument("--config", type=Path, default=PACKAGE_DIR / "configs" / "cameras.json")
    prepare.add_argument("--weights", type=Path, default=DEFAULT_PERSON_WEIGHTS)
    prepare.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    prepare.add_argument("--cameras", nargs="+")
    prepare.add_argument("--interval", type=float, default=30.0)
    prepare.add_argument("--start", type=float, default=0.0)
    prepare.add_argument("--duration", type=float)
    prepare.add_argument("--group-seconds", type=int, default=600)
    prepare.add_argument("--guard-seconds", type=int, default=60)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--device", default="0")
    prepare.add_argument("--imgsz", type=int, default=1280)
    prepare.add_argument("--batch-size", type=int, default=4)
    prepare.add_argument("--decode-mode", choices=("sequential", "seek"), default="sequential")
    review = sub.add_parser("review", help="Open the local B3/B4 annotation page")
    review.add_argument("--dataset", type=Path, required=True)
    review.add_argument("--port", type=int, default=8773)
    status = sub.add_parser("status", help="Show B3/B4 annotation progress")
    status.add_argument("--dataset", type=Path, required=True)
    export = sub.add_parser("export", help="Export reviewed boxes to a YOLO dataset")
    export.add_argument("--dataset", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        summary = prepare_review_dataset(output=args.out, video_root=args.video_root,
            config=args.config, weights=args.weights, policy=args.policy, camera_ids=args.cameras,
            interval=args.interval, start=args.start, duration=args.duration,
            group_seconds=args.group_seconds, guard_seconds=args.guard_seconds, seed=args.seed,
            device=args.device, image_size=args.imgsz, batch_size=args.batch_size,
            decode_mode=args.decode_mode)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "review":
        _serve(args)
    elif args.command == "status":
        print(json.dumps(B3B4ReviewStore(args.dataset).status(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(export_yolo_dataset(args.dataset, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
