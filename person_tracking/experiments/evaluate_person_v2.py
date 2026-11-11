"""Evaluate a person detector on one exported validation split, without training.

PersonValidator and fixed-threshold matching retain the policy of
runs/annotation_check_20261105/evaluate_baseline.py for v1/v2 comparisons.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time

from person_tracking.common import PACKAGE_DIR, load_config, read_jsonl, write_json
from person_tracking.dataset import read_labels
from person_tracking.detector import select_device

os.environ.setdefault("YOLO_CONFIG_DIR", str(PACKAGE_DIR / ".cache" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PACKAGE_DIR / ".cache" / "matplotlib"))
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import nms
from ultralytics.utils.metrics import ConfusionMatrix


class PersonValidator(DetectionValidator):
    """Class-0, single-label NMS, matching the existing person predictor.

    Ultralytics 8.4.202's default validator does not forward args.classes into
    NMS and enables multi_label; retain the project's explicit person policy.
    """

    def init_metrics(self, model):
        super().init_metrics(model)
        self.names = {0: model.names[0]}
        self.nc = 1
        self.metrics.names = self.names
        self.confusion_matrix = ConfusionMatrix(
            names=self.names, save_matches=self.args.plots and self.args.visualize
        )

    def postprocess(self, preds):
        outputs = nms.non_max_suppression(
            preds, self.args.conf, self.args.iou, classes=[0],
            agnostic=self.args.agnostic_nms, multi_label=False,
            max_det=self.args.max_det, nc=0, end2end=self.end2end,
        )
        return [
            {"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]}
            for x in outputs
        ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xyxy(box):
    cx, cy, width, height = box
    return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2


def iou(a, b):
    a, b = xyxy(a), xyxy(b)
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if intersection else 0.0


def finalize(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        **dict(counts), "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def validation_inputs(dataset: Path):
    """Require the project's exported layout and evaluate exactly its val IDs."""
    import yaml

    manifest = dataset / "exported_samples.jsonl"
    dataset_yaml = dataset / "data.yaml"
    config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    declared_root = Path(config.get("path", dataset))
    if not declared_root.is_absolute():
        declared_root = dataset / declared_root
    if declared_root.resolve() != dataset or config.get("val") != "images/val":
        raise ValueError("Expected an exported dataset whose val is its own images/val directory")
    if config.get("names") not in ({0: "person"}, ["person"]):
        raise ValueError("Expected a single class named person")
    rows = [row for row in read_jsonl(manifest) if row["split"] == "val"]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Validation manifest is empty or contains duplicate IDs")
    records = []
    for row in rows:
        image = dataset / row["image"]
        label = dataset / row["label"]
        if image.parent.resolve() != dataset / "images" / "val":
            raise ValueError(f"Unexpected validation image path: {image}")
        if label.parent.resolve() != dataset / "labels" / "val":
            raise ValueError(f"Unexpected validation label path: {label}")
        if image.stem != row["id"] or label.stem != row["id"]:
            raise ValueError(f"Sample ID does not match image/label filenames: {row['id']}")
        truth = read_labels(label)
        records.append({
            "id": row["id"], "camera_id": row["camera_id"],
            "image": row["image"], "image_sha256": sha256(image),
            "label": row["label"], "label_sha256": sha256(label),
            "ground_truth_boxes": len(truth),
        })
    actual_images = {
        path.name for path in (dataset / "images" / "val").iterdir() if path.is_file()
    }
    if actual_images != {Path(row["image"]).name for row in rows}:
        raise ValueError("Validation directory and exported manifest have different images")
    return rows, records


def main():
    import torch
    import ultralytics
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default=None, help="Defaults to project config (auto)")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
    dataset, model_path, output = args.dataset.resolve(), args.weights.resolve(), args.out.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new --out directory: {output}")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    rows, input_records = validation_inputs(dataset)
    cfg = load_config()
    device = select_device(args.device or cfg["device"])
    options = {
        "data": str(dataset / "data.yaml"), "split": "val", "imgsz": cfg["image_size"],
        "conf": 0.001, "iou": cfg["nms_iou"], "classes": [0], "device": device,
        "quantize": 16 if device != "cpu" else 32, "batch": args.batch, "workers": 0,
        "plots": True, "save_txt": True, "save_conf": True,
        "project": str(output), "name": "predictions", "exist_ok": False,
    }
    provenance = model_path.parent.parent / "training_config.json"
    output.mkdir(parents=True, exist_ok=False)
    configuration = {
        "model": str(model_path), "model_sha256": sha256(model_path),
        "dataset": str(dataset), "dataset_yaml_sha256": sha256(dataset / "data.yaml"),
        "exported_manifest_sha256": sha256(dataset / "exported_samples.jsonl"),
        "val_options": options, "validation_images": len(rows),
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "evaluation_only": True, "training_started_in_evaluation": False,
        "test_split_evaluated": False, "started_at": datetime.now().isoformat(),
        "training_provenance": str(provenance) if provenance.is_file() else None,
        "training_provenance_sha256": sha256(provenance) if provenance.is_file() else None,
        "validator": "PersonValidator: class 0 only, single-label NMS; policy retained from v1 evaluator",
        "prediction_txt_format": "Ultralytics Results.save_txt: normalized xywh and confidence, %g (six significant digits)",
        "numeric_compatibility": "Fixed counts read the same TXT format as the v1 evaluator; near-threshold values can be affected by its rounding.",
        "precision_mode": "quantize=16 selects FP16 compute on CUDA; this is not INT8 export",
        "status": "running",
    }
    write_json(output / "config.json", configuration)
    write_json(output / "validation_inputs.json", input_records)
    started = time.perf_counter()
    try:
        model = YOLO(str(model_path))
        if str(model.names.get(0, "")).lower() != "person":
            raise ValueError("Model class 0 must be person")
        metrics = model.val(validator=PersonValidator, **options)
        prediction_dir = Path(metrics.save_dir) / "labels"
        totals, by_camera, image_metrics = Counter(), defaultdict(Counter), []
        expected_ids = {row["id"] for row in rows}
        unexpected = {path.stem for path in prediction_dir.glob("*.txt")} - expected_ids
        if unexpected:
            raise ValueError(f"Unexpected prediction filenames: {sorted(unexpected)}")
        for row in rows:
            truth = read_labels(dataset / row["label"])
            prediction = prediction_dir / f"{row['id']}.txt"
            predictions = []
            if prediction.is_file():
                for line in prediction.read_text(encoding="utf-8").splitlines():
                    fields = line.split()
                    if len(fields) != 6 or fields[0] != "0":
                        raise ValueError(f"Unexpected person prediction format: {prediction}")
                    if float(fields[5]) >= cfg["detection_conf"]:
                        predictions.append(([float(v) for v in fields[1:5]], float(fields[5])))
            unmatched, tp = set(range(len(truth))), 0
            for box, score in sorted(predictions, key=lambda p: p[1], reverse=True):
                best = max(unmatched, key=lambda i: iou(box, truth[i]), default=None)
                if best is not None and iou(box, truth[best]) >= 0.5:
                    tp += 1
                    unmatched.remove(best)
            counts = {
                "images": 1, "positive_images": int(bool(truth)), "negative_images": int(not truth),
                "ground_truth_boxes": len(truth), "predicted_boxes": len(predictions),
                "tp": tp, "fp": len(predictions) - tp, "fn": len(unmatched),
                "negative_images_with_false_positive": int(not truth and bool(predictions)),
            }
            totals.update(counts)
            by_camera[row["camera_id"]].update(counts)
            image_metrics.append({"id": row["id"], "camera_id": row["camera_id"], **counts})
        # A concurrent annotation or image edit invalidates the evaluation snapshot.
        _, after_records = validation_inputs(dataset)
        if after_records != input_records:
            raise RuntimeError("Validation images/labels changed during evaluation")
        result = {
            "dataset": str(dataset), "split": "val", "model": str(model_path),
            "model_sha256": configuration["model_sha256"],
            "map_metrics": {k: float(v) for k, v in metrics.results_dict.items()},
            "speed_ms_per_image": metrics.speed, "wall_s": time.perf_counter() - started,
            "operating_point": {
                "confidence_threshold": cfg["detection_conf"], "match_iou_threshold": 0.5,
                "matching": "confidence-descending predictions, one-to-one best-IoU ground truth",
                "all": finalize(totals),
                "by_camera": {cam: finalize(counts) for cam, counts in sorted(by_camera.items())},
            },
            "evaluation_only": True, "training_started_in_evaluation": False,
            "training_provenance": configuration["training_provenance"], "test_split_evaluated": False,
            "limitations": [
                "Single-day validation footage; no independent-date or field safety acceptance is implied.",
                "Human label accuracy is not independently certified; static detection does not evaluate tracking IDs.",
                "Ultralytics Precision/Recall use its selected confidence point; operating_point uses fixed confidence 0.10.",
                "Fixed counts retain six-significant-digit prediction TXT rounding used by the v1 evaluator.",
                "GPU batch timing is not a target edge-box throughput measurement.",
            ],
        }
        write_json(output / "metrics.json", result)
        write_json(output / "per_image.json", image_metrics)
        configuration.update(status="completed", completed_at=datetime.now().isoformat())
        write_json(output / "config.json", configuration)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        configuration.update(status="failed", failure=repr(error), elapsed_seconds=time.perf_counter() - started)
        write_json(output / "config.json", configuration)
        raise


if __name__ == "__main__":
    main()
