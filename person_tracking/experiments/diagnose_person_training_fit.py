"""Measure fixed-threshold fit/retention on train images; never an independent test."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path

from person_tracking.common import PACKAGE_DIR, read_jsonl, write_json
from person_tracking.dataset import read_image, read_labels
from person_tracking.experiments.evaluate_person_v2 import finalize, iou, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scope", choices=("new_training_fit_diagnostic", "old_training_retention_diagnostic"), required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    dataset, model_path, output = args.dataset.resolve(), args.weights.resolve(), args.out.resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = [row for row in read_jsonl(dataset / "exported_samples.jsonl") if row["split"] == "train"]
    if not rows:
        raise ValueError("No training images in the exported manifest")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PACKAGE_DIR / ".cache/ultralytics"))
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if model.names.get(0) != "person":
        raise ValueError("Expected class 0 person")
    records, totals, provenance = [], Counter(), []
    for index, row in enumerate(rows, 1):
        image, label = dataset / row["image"], dataset / row["label"]
        before = (sha256(image), sha256(label))
        truth = read_labels(label)
        result = model.predict(read_image(image), imgsz=1280, conf=0.10, iou=0.60,
                               classes=[0], device=args.device, quantize=16 if args.device != "cpu" else 32,
                               verbose=False, save=False)[0]
        predictions = [(list(map(float, box)), float(score))
                       for box, score in zip(result.boxes.xywhn.cpu().tolist(), result.boxes.conf.cpu().tolist())]
        unmatched, tp = set(range(len(truth))), 0
        for box, _ in sorted(predictions, key=lambda item: item[1], reverse=True):
            best = max(unmatched, key=lambda i: iou(box, truth[i]), default=None)
            if best is not None and iou(box, truth[best]) >= 0.50:
                unmatched.remove(best)
                tp += 1
        counts = {"images": 1, "positive_images": int(bool(truth)), "negative_images": int(not truth),
                  "ground_truth_boxes": len(truth), "predicted_boxes": len(predictions),
                  "tp": tp, "fp": len(predictions)-tp, "fn": len(unmatched)}
        totals.update(counts)
        if before != (sha256(image), sha256(label)):
            raise RuntimeError("Diagnostic inputs changed during evaluation")
        provenance.append({"id": row["id"], "image_sha256": before[0], "label_sha256": before[1]})
        records.append({"id": row["id"], "camera_id": row["camera_id"], **counts,
                        "predictions_normalized_xywh_confidence": predictions})
        if index % 100 == 0:
            print(f"Training diagnostic {index}/{len(rows)}", flush=True)
    report = {"evaluated_at": datetime.now().astimezone().isoformat(), "scope": args.scope,
              "dataset": str(dataset), "input_partition": "train", "model": str(model_path),
              "model_sha256": sha256(model_path), "confidence": 0.10, "match_iou": 0.50,
              "nms_iou": 0.60, "imgsz": 1280, "device": args.device, "totals": finalize(totals),
              "independent_validation": False, "test_split_used": False, "training_started": False,
              "limits": "These inputs belong to training partitions. Fit and retention comparisons are diagnostics, not non-workwear generalization or independent acceptance. Raw detector only, without tracking/display filtering/nested deduplication."}
    output.mkdir(parents=True)
    write_json(output / "metrics.json", report)
    write_json(output / "inputs.json", provenance)
    write_json(output / "per_image.json", records)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
