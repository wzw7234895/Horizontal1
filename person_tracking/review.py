from __future__ import annotations

import json
import shutil
import threading
from collections import Counter
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .common import read_jsonl, write_json
from .dataset import read_labels, validate_boxes, write_labels


class ReviewStore:
    def __init__(self, dataset):
        self.root = Path(dataset).resolve()
        summary = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "completed":
            raise ValueError("Sampling did not finish; use a completed dataset")
        self.summary = summary
        self.rows = read_jsonl(self.root / "samples.jsonl")
        if len({r["id"] for r in self.rows}) != len(self.rows):
            raise ValueError("Duplicate sample IDs")
        self.review_path = self.root / "reviews.json"
        self.reviews = json.loads(self.review_path.read_text(encoding="utf-8")) if self.review_path.exists() else {}
        self.lock = threading.RLock()

    def row(self, index):
        if type(index) is not int or not 0 <= index < len(self.rows):
            raise ValueError("Invalid sample index")
        return self.rows[index]

    def label_path(self, row):
        return self.root / "reviewed_labels" / f"{row['id']}.txt"

    def items(self):
        with self.lock:
            return [{"index": i, "id": r["id"], "camera_id": r["camera_id"], "split": r["split"],
                     "time": r["recording_time_estimate"], "proposal_count": r["proposal_count"],
                     "review_reason": r.get("review_reason", ""),
                     "annotation_locked": bool(r.get("annotation_locked", False)),
                     "status": self.reviews.get(r["id"], {}).get("status", "pending")}
                    for i, r in enumerate(self.rows)]

    def item(self, index):
        with self.lock:
            row = self.row(index)
            path = self.label_path(row)
            if not path.exists():
                path = self.root / row["proposal_label"]
            return {**self.items()[index], "width": row["width"], "height": row["height"],
                    "boxes": read_labels(path)}

    def save(self, index, status, boxes=None, confirm_empty=False):
        if status not in ("reviewed", "excluded"):
            raise ValueError("Status must be reviewed or excluded")
        with self.lock:
            row = self.row(index)
            if row.get("annotation_locked", False):
                raise ValueError("This inherited test sample is sealed; its annotation cannot be changed here")
            clean = validate_boxes(boxes or [])
            if status == "reviewed":
                if not clean and confirm_empty is not True:
                    raise ValueError("Empty labels require explicit confirm_empty=true")
                path = self.label_path(row)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                write_labels(temporary, clean)
                temporary.replace(path)
            self.reviews[row["id"]] = {"status": status, "box_count": len(clean),
                "confirmed_empty": status == "reviewed" and not clean,
                "reviewed_at": datetime.now().isoformat(), "review_source": "local_review_interface"}
            temporary = self.review_path.with_suffix(".tmp")
            write_json(temporary, self.reviews)
            temporary.replace(self.review_path)
            return self.reviews[row["id"]]

    def status(self):
        with self.lock:
            counts = Counter(item["status"] for item in self.items())
            return {"dataset": str(self.root), "total": len(self.rows),
                    "pending": counts["pending"], "reviewed": counts["reviewed"], "excluded": counts["excluded"],
                    "reviewed_by_split": dict(Counter(r["split"] for r in self.rows
                        if self.reviews.get(r["id"], {}).get("status") == "reviewed")),
                    "confirmed_negative_images": sum(r.get("status") == "reviewed" and r.get("confirmed_empty", False)
                                                     for r in self.reviews.values()),
                    **{key: self.summary[key] for key in ("split_selection_status", "test_set_sealed")
                       if key in self.summary},
                    "review_instructions": self.summary.get("review_instructions", "")}


def export_dataset(dataset, output):
    store = ReviewStore(dataset)
    if store.summary.get("split_selection_status") == "pending_reselection":
        raise ValueError("Dataset split selection is pending; reselect train/val/test before export")
    if store.summary.get("require_complete_review") and store.status()["pending"]:
        raise ValueError("Complete all pending annotations before exporting this review version")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new export directory: {output}")
    selected, positives = [], Counter()
    group_splits = {}
    for row in store.rows:
        previous = group_splits.setdefault(row["split_group"], row["split"])
        if previous != row["split"]:
            raise ValueError("One temporal group appears in multiple splits")
        review = store.reviews.get(row["id"], {})
        if review.get("status") != "reviewed":
            continue
        labels = read_labels(store.label_path(row))
        if not labels and not review.get("confirmed_empty"):
            raise ValueError(f"Unconfirmed negative labels: {row['id']}")
        if len(labels) != review.get("box_count"):
            raise ValueError(f"Reviewed label file was modified; review again: {row['id']}")
        if not (store.root / row["image"]).is_file():
            raise FileNotFoundError(row["image"])
        if row["split"] not in ("train", "val", "test"):
            raise ValueError("Invalid dataset split")
        positives[row["split"]] += bool(labels)
        selected.append(row)
    if not selected:
        raise ValueError("No reviewed images. Open the review UI and correct proposals first; no dataset was exported.")
    missing = [split for split in ("train", "val", "test") if positives[split] == 0]
    if missing:
        raise ValueError(f"Need >=1 reviewed positive image in every split; missing: {missing}")
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    for row in selected:
        target = output / "images" / row["split"] / Path(row["image"]).name
        shutil.copy2(store.root / row["image"], target)
        shutil.copy2(store.label_path(row), output / "labels" / row["split"] / f"{row['id']}.txt")
    import yaml
    (output / "data.yaml").write_text(yaml.safe_dump({"path": output.as_posix(), "train": "images/train",
        "val": "images/val", "test": "images/test", "names": {0: "person"}}, allow_unicode=True,
        sort_keys=False), encoding="utf-8")
    with (output / "exported_samples.jsonl").open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps({**row, "source_sample_image": row["image"],
                "image": f"images/{row['split']}/{Path(row['image']).name}",
                "label": f"labels/{row['split']}/{row['id']}.txt", "review_status": "reviewed",
                "review": store.reviews[row["id"]]}, ensure_ascii=False) + "\n")
    summary = {"reviewed_images_exported": len(selected), "by_split": dict(Counter(r["split"] for r in selected)),
        "positive_images_by_split": dict(positives), "omitted_unreviewed_or_excluded": len(store.rows)-len(selected),
        "source_dataset": str(store.root), "training_started": False}
    write_json(output / "summary.json", summary)
    return summary


def make_server(dataset, port=8765):
    store = ReviewStore(dataset)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, status, body, content_type="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
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
                    self.respond(200, (Path(__file__).parent / "review.html").read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/items":
                    self.respond(200, {"items": store.items(), "status": store.status()})
                elif url.path in ("/api/item", "/api/image"):
                    index = int(parse_qs(url.query)["index"][0])
                    if url.path == "/api/item":
                        self.respond(200, store.item(index))
                    else:
                        self.respond(200, (store.root / store.row(index)["image"]).read_bytes(), "image/jpeg")
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
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                saved = store.save(body["index"], body["status"], body.get("boxes"), body.get("confirm_empty", False))
                self.respond(200, {"saved": saved, "status": store.status()})
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.respond(400, {"error": str(exc)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def run_review(args):
    server = make_server(args.dataset, args.port)
    print(f"Review page: http://127.0.0.1:{server.server_port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
