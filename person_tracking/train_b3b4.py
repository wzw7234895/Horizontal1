"""Train the reviewed B3/B4 detector after class and split audits pass."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .common import write_json


def train(dataset, *, output, base_model="yolo11n.pt", epochs=80, image_size=1280,
          batch_size=4, device="0", patience=15, workers=4):
    dataset = Path(dataset).resolve()
    output = Path(output).resolve()
    audit_path = dataset / "audit.json"
    data_yaml = dataset / "data.yaml"
    if not audit_path.is_file() or not data_yaml.is_file():
        raise FileNotFoundError("Expected an exported dataset with audit.json and data.yaml")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("training_ready"):
        raise ValueError("Dataset audit did not mark the reviewed training split ready")
    missing_train = audit.get("missing_classes_in_train", [])
    missing_val = audit.get("missing_classes_in_val", [])
    if missing_train or missing_val:
        raise ValueError(f"Every detector class needs reviewed train and val examples; train={missing_train}, val={missing_val}")
    if output.exists():
        raise FileExistsError(f"Choose a new run directory: {output}")
    if epochs < 1 or image_size < 32 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, patience and image_size must be positive")

    output.mkdir(parents=True)
    config = {"started": datetime.now().isoformat(timespec="seconds"),
              "dataset": str(dataset), "dataset_audit": audit,
              "base_model": base_model, "epochs": epochs, "image_size": image_size,
              "batch_size": batch_size, "device": device, "patience": patience,
              "workers": workers, "model_scope": "head helmet state/color and workwear torso color",
              "training_started": False}
    write_json(output / "training_config.json", config)

    from ultralytics import YOLO

    model = YOLO(base_model)
    config["training_started"] = True
    config["model_names"] = list(model.names.values()) if isinstance(model.names, dict) else model.names
    write_json(output / "training_config.json", config)
    results = model.train(data=str(data_yaml), epochs=epochs, imgsz=image_size,
                          batch=batch_size, device=device, patience=patience,
                          workers=workers, project=str(output), name="ultralytics",
                          exist_ok=False, verbose=True)
    best = Path(results.save_dir) / "weights" / "best.pt"
    last = Path(results.save_dir) / "weights" / "last.pt"
    metrics = {"best_weights": str(best) if best.is_file() else None,
               "last_weights": str(last) if last.is_file() else None,
               "results_csv": str(Path(results.save_dir) / "results.csv"),
               "training_started": True}
    write_json(output / "training_result.json", metrics)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the reviewed B3/B4 PPE detector")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    print(json.dumps(train(args.dataset, output=args.out, base_model=args.model,
        epochs=args.epochs, image_size=args.imgsz, batch_size=args.batch,
        device=args.device, patience=args.patience, workers=args.workers),
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
