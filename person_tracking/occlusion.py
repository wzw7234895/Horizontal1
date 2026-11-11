"""Human ground truth and fixed-policy metrics for the occlusion holdout.

This format deliberately has no YOLO training export and no model proposals.
Boxes cover all visible parts, consistent with the corrected person labels.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import threading

from .common import read_jsonl, write_json
from .dataset import validate_boxes


OCCLUSIONS = ("none", "light", "moderate", "heavy", "extreme", "unknown")
OCCLUDERS = ("load", "structure", "person", "other", "unknown")
DISTANCES = ("near", "far", "unknown")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def validate_objects(objects):
    if not isinstance(objects, list):
        raise ValueError("objects must be a list")
    clean = []
    for obj in objects:
        box = validate_boxes([obj["box"]])[0]
        level, causes, distance = obj.get("occlusion"), obj.get("occluders"), obj.get("distance")
        if level not in OCCLUSIONS or distance not in DISTANCES:
            raise ValueError("每个人员框必须选择遮挡比例和远近属性（可选无法判断）")
        if not isinstance(causes, list) or any(c not in OCCLUDERS for c in causes) or len(causes) != len(set(causes)):
            raise ValueError("Invalid occluders")
        if (level == "none" and causes) or (level != "none" and not causes):
            raise ValueError("无遮挡应清空遮挡物；有遮挡或未知时至少选择一种遮挡物")
        if type(obj.get("truncated")) is not bool:
            raise ValueError("truncated must be a boolean")
        if box in [p["box"] for p in clean]:
            raise ValueError("Duplicate person box")
        clean.append({"box": box, "occlusion": level, "occluders": causes,
                      "distance": distance, "truncated": obj["truncated"]})
    return clean


class OcclusionStore:
    def __init__(self, dataset):
        self.root = Path(dataset).resolve()
        self.summary = json.loads((self.root / "summary.json").read_text(encoding="utf8"))
        if self.summary.get("status") != "completed" or self.summary.get("kind") != "occlusion_holdout":
            raise ValueError("Need a completed occlusion holdout")
        self.rows = read_jsonl(self.root / "samples.jsonl")
        if not self.rows or len({r["id"] for r in self.rows}) != len(self.rows):
            raise ValueError("Empty holdout or duplicate IDs")
        for row in self.rows:
            if not (self.root / row["image"]).resolve().is_relative_to(self.root):
                raise ValueError("Image must remain inside holdout")
        self.protocol = json.loads((self.root / "protocol.json").read_text(encoding="utf8"))
        self.review_path = self.root / "reviews.json"
        self.reviews = json.loads(self.review_path.read_text(encoding="utf8")) if self.review_path.exists() else {}
        self.lock = threading.RLock()
        self.frozen = (self.root / "freeze.json").exists()

    def row(self, index):
        if type(index) is not int or not 0 <= index < len(self.rows):
            raise ValueError("Invalid index")
        return self.rows[index]

    def items(self):
        with self.lock:
            return [{"index": i, "id": row["id"], "camera_id": row["camera_id"],
                     "time": row["recording_time_estimate"], "clip_id": row["clip_id"],
                     "status": self.reviews.get(row["id"], {}).get("status", "pending")}
                    for i, row in enumerate(self.rows)]

    def item(self, index):
        with self.lock:
            row = self.row(index)
            review = self.reviews.get(row["id"], {})
            return {**self.items()[index], "width": row["width"], "height": row["height"],
                    "objects": review.get("objects", []), "ignore_regions": review.get("ignore_regions", []),
                    "review": review, "annotation_locked": self.frozen}

    def save(self, index, status, objects=None, ignore_regions=None, confirm_empty=False, reason=""):
        with self.lock:
            if self.frozen:
                raise ValueError("冻结评估集不可修改，请回到复核源目录修改并生成新版本")
            row = self.row(index)
            if status not in ("reviewed", "excluded"):
                raise ValueError("Invalid review status")
            if not isinstance(reason, str):
                raise ValueError("reason must be text")
            if status == "excluded":
                if not reason.strip():
                    raise ValueError("排除帧必须填写原因，排除数量和原因会写入报告")
                clean, ignore = [], []
            else:
                clean = validate_objects(objects if objects is not None else [])
                ignore = validate_boxes(ignore_regions if ignore_regions is not None else [])
                if not clean and confirm_empty is not True:
                    raise ValueError("无可确认人员的帧必须显式确认，无法辨认的区域请画忽略框")
            review = {"status": status, "objects": clean, "ignore_regions": ignore,
                      "confirmed_empty": status == "reviewed" and not clean and not ignore,
                      "confirmed_no_evaluable_person": status == "reviewed" and not clean,
                      "reason": reason.strip(), "reviewed_at": datetime.now().isoformat(),
                      "review_source": "human_occlusion_review"}
            self.reviews[row["id"]] = review
            atomic_json(self.review_path, self.reviews)
            return review

    def status(self):
        with self.lock:
            counts = Counter(x["status"] for x in self.items())
            objects = [p for row in self.rows for p in self.reviews.get(row["id"], {}).get("objects", [])
                       if self.reviews.get(row["id"], {}).get("status") == "reviewed"]
            return {"dataset": str(self.root), "total": len(self.rows), "pending": counts["pending"],
                    "reviewed": counts["reviewed"], "excluded": counts["excluded"], "people": len(objects),
                    "by_occlusion": dict(Counter(p["occlusion"] for p in objects)),
                    "by_occluder": dict(Counter(c for p in objects for c in p["occluders"])),
                    "frozen": self.frozen, "evaluation_only": True}


def corners(box):
    x, y, w, h = box
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def overlap(a, b, denominator="union"):
    ax1, ay1, ax2, ay2 = corners(a)
    bx1, by1, bx2, by2 = corners(b)
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    area_a, area_b = a[2] * a[3], b[2] * b[3]
    divisor = area_a if denominator == "prediction" else area_a + area_b - intersection
    return intersection / divisor if divisor > 0 else 0.0


def match_frame(objects, predictions, ignore_regions, confidence=0.10, match_iou=0.50, ignore_ioa=0.50):
    """One prediction can match one person; valid GT takes priority over ignore."""
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (confidence, match_iou, ignore_ioa)):
        raise ValueError("Invalid matching thresholds")
    predictions = sorted((p for p in predictions if float(p["confidence"]) >= confidence),
                         key=lambda p: float(p["confidence"]), reverse=True)
    unmatched, matched = set(range(len(objects))), {}
    fp = ignored = 0
    for prediction in predictions:
        box = validate_boxes([prediction["box"]])[0]
        score = float(prediction["confidence"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid prediction confidence")
        best = max(unmatched, key=lambda j: overlap(box, objects[j]["box"]), default=None)
        if best is not None and overlap(box, objects[best]["box"]) >= match_iou:
            matched[best] = {"confidence": score, "iou": overlap(box, objects[best]["box"])}
            unmatched.remove(best)
        elif any(overlap(box, region, "prediction") >= ignore_ioa for region in ignore_regions):
            ignored += 1
        else:
            fp += 1
    return {"tp": len(matched), "fn": len(unmatched), "fp": fp, "ignored_predictions": ignored,
            "predictions_above_threshold": len(predictions),
            "objects": [{**obj, "matched": j in matched, **matched.get(j, {})} for j, obj in enumerate(objects)]}


def group_keys(obj, camera, small_height=64):
    h720 = obj["box"][3] * 720
    size = "visible_h_lt32" if h720 < 32 else "visible_h_32_to64" if h720 < 64 else "visible_h_ge64"
    keys = [("all", "all"), ("camera", camera), ("occlusion", obj["occlusion"]), ("size", size),
            ("distance", obj["distance"]), ("truncated", str(obj["truncated"]).lower())]
    for cause in obj["occluders"]:
        keys.extend([("occluder", cause), ("occlusion_by_occluder", f"{obj['occlusion']}__{cause}")])
    if h720 < small_height:
        keys.append(("small_target", "visible_small"))
        if obj["distance"] == "far":
            keys.append(("small_target", "far_and_visible_small"))
    return keys


def wilson(tp, n):
    if not n:
        return None
    z, p = 1.959963984540054, tp / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    delta = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0.0, center-delta), min(1.0, center+delta)]


def recall_report(frame_results, protocol):
    """Recall denominators come exclusively from human annotated people."""
    groups = {}
    predefined = {"all": ["all"], "occlusion": OCCLUSIONS, "occluder": OCCLUDERS,
                  "distance": DISTANCES, "size": ["visible_h_lt32", "visible_h_32_to64", "visible_h_ge64"],
                  "small_target": ["visible_small", "far_and_visible_small"]}
    for category, names in predefined.items():
        for name in names:
            groups[category, name] = {"tp": 0, "fn": 0, "images": set(), "clips": set()}
    totals = Counter()
    for frame in frame_results:
        totals.update({k: frame[k] for k in ("tp", "fp", "fn", "ignored_predictions")})
        for obj in frame["objects"]:
            for key in group_keys(obj, frame["camera_id"], protocol["small_visible_height_720p"]):
                group = groups.setdefault(key, {"tp": 0, "fn": 0, "images": set(), "clips": set()})
                group["tp" if obj["matched"] else "fn"] += 1
                group["images"].add(frame["id"])
                group["clips"].add(frame.get("evaluation_group", frame["clip_id"]))
    rows = []
    for (category, name), group in sorted(groups.items()):
        n = group["tp"] + group["fn"]
        support = n >= protocol["min_people_per_stratum"] and len(group["clips"]) >= protocol["min_clips_per_stratum"]
        recall = group["tp"] / n if n else None
        rows.append({"category": category, "stratum": name, "tp": group["tp"], "fn": group["fn"],
                     "ground_truth_people": n, "positive_images": len(group["images"]),
                     "positive_clips": len(group["clips"]), "recall": recall,
                     "wilson95_descriptive": wilson(group["tp"], n),
                     "coverage": "not_verified" if not n else "limited" if not support else "supported_counts",
                     "recall_target_met": None if not support else recall >= protocol["target_recall"]})
    totals["images"] = len(frame_results)
    precision_denominator = totals["tp"] + totals["fp"]
    recall_denominator = totals["tp"] + totals["fn"]
    return {"totals": {**dict(totals), "precision": totals["tp"] / precision_denominator if precision_denominator else None,
                       "recall": totals["tp"] / recall_denominator if recall_denominator else None}, "strata": rows,
            "notes": ["A person may occur in multiple occluder groups; never sum overlapping groups.",
                      "Per-stratum precision is not attributed to ground-truth occlusion; overall FP is reported.",
                      "Wilson intervals are descriptive only: nearby frames/people are correlated.",
                      "Visible box size is not metric distance; far/near is human tagged separately.",
                      "Fully invisible people are not image-level GT; this does not evaluate invisible-person events."]}
