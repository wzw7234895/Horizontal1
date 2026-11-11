"""Train an audited person snapshot, retaining an optimizer checkpoint for continuation."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import time

from person_tracking.common import PACKAGE_DIR, load_config, read_jsonl, write_json
from person_tracking.experiments.train_person_static_v1 import file_sha256, normalize_result_metrics, query_gpu_status


def resolve_initialization(weights=None):
    path = Path(weights).resolve() if weights is not None else Path(load_config()["model"])
    if not path.is_file():
        raise FileNotFoundError(f"Pretrained weights not found: {path}")
    return path


def validate_initialization(audit, model_path):
    if audit.get("fresh_public_pretrained_initialization_required"):
        expected = audit.get("required_initialization", {}).get("sha256")
        if not expected or file_sha256(model_path) != expected:
            raise ValueError("Corrected annotations require the clean public initialization bound to this audit")
    required_checkpoint = audit.get("required_checkpoint_initialization")
    if required_checkpoint and file_sha256(model_path) != required_checkpoint.get("sha256"):
        raise ValueError("Specialist fine-tuning requires the checkpoint bound to this dataset audit")


def validate_export_contract(dataset, audit):
    """Ensure YOLO reads exactly the registered partitions, including legacy exports."""
    import yaml

    dataset = Path(dataset).resolve()
    contract = audit.get("export_metadata_sha256", {})
    for name in ("data.yaml", "exported_samples.jsonl"):
        if name in contract and file_sha256(dataset / name) != contract[name]:
            raise ValueError(f"Export metadata changed: {name}")
    routing = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf8"))
    configured_root = Path(routing.get("path", ""))
    if not contract and not configured_root.is_absolute():
        raise ValueError("Legacy export contract requires an absolute YAML path")
    root = (configured_root if configured_root.is_absolute() else dataset / configured_root).resolve()
    if root != dataset or routing.get("names") not in ({0: "person"}, ["person"]):
        raise ValueError("YOLO data routing must use this registered person export")
    rows = read_jsonl(dataset / "exported_samples.jsonl")
    registered = json.loads(Path(audit["fingerprints_file"]).read_text(encoding="utf8"))
    if len({row["id"] for row in rows}) != len(rows) or \
       {(row["id"], row["split"]) for row in rows} != {(row["id"], row["split"]) for row in registered}:
        raise ValueError("Manifest and registered membership differ")
    counts = {}
    for split in ("train", "val", "test"):
        if routing.get(split) != f"images/{split}":
            raise ValueError("YOLO data routing must preserve registered partitions")
        selected = [row for row in rows if row["split"] == split]
        for folder, key, extension in (("images", "image", ".jpg"), ("labels", "label", ".txt")):
            expected = {f"{folder}/{split}/{row['id']}{extension}" for row in selected}
            if {row[key] for row in selected} != expected:
                raise ValueError("Manifest paths differ from registered sample IDs")
            actual = {path.relative_to(dataset).as_posix() for path in (dataset / folder / split).rglob("*") if path.is_file()}
            if actual != expected:
                raise ValueError(f"Unregistered or missing {folder} in {split}")
        counts[split] = len(selected)
    return {"metadata_sha256": {name: file_sha256(dataset / name) for name in ("data.yaml", "exported_samples.jsonl")},
            "partition_membership": counts, "routing_and_directory_membership_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", type=Path, help="Explicit pretrained checkpoint; overrides the tracking configuration")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup-bias-lr", type=float, default=0.1)
    parser.add_argument("--wait-for-idle", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--min-free-mib", type=int, default=3500)
    parser.add_argument("--max-gpu-utilization", type=int, default=10)
    args = parser.parse_args()
    dataset, output = args.dataset.resolve(), args.out.resolve()
    model_path = resolve_initialization(args.weights)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if args.epochs < 1 or args.patience < 0 or args.poll_seconds <= 0 or args.min_free_mib <= 0 or not 0 <= args.max_gpu_utilization <= 100:
        parser.error("Invalid epochs, patience, or polling interval")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if args.batch < 1 or not math.isfinite(args.warmup_bias_lr) or args.warmup_bias_lr < 0:
        parser.error("batch must be positive; warmup-bias-lr must be finite and nonnegative")
    audit = json.loads((dataset / "annotation_audit.json").read_text(encoding="utf8"))
    review_status = audit.get("review_status") or audit.get("source_review_status")
    if audit["status"] != "passed" or not review_status or review_status["pending"] or Path(audit["exported_dataset"]) != dataset:
        raise ValueError("Dataset must be the completed, audited export")
    validate_initialization(audit, model_path)
    fingerprints_path = Path(audit["fingerprints_file"])
    if file_sha256(fingerprints_path) != audit["fingerprints_sha256"]:
        raise ValueError("Audit fingerprints changed")
    for item in json.loads(fingerprints_path.read_text(encoding="utf8")):
        if item["split"] not in ("train", "val", "test"):
            raise ValueError("Unexpected split")
        for kind, extension in (("images", ".jpg"), ("labels", ".txt")):
            path = dataset / kind / item["split"] / (item["id"] + extension)
            expected = item["image_sha256" if kind == "images" else "label_sha256"]
            if file_sha256(path) != expected:
                raise ValueError(f"Export changed since audit: {path}")
    export_contract = validate_export_contract(dataset, audit)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PACKAGE_DIR / ".cache/ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(PACKAGE_DIR / ".cache/matplotlib"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    if args.device != "cpu":
        consecutive = 0
        while consecutive < 2:
            free, total, utilization = query_gpu_status(args.device)
            consecutive = consecutive + 1 if free >= args.min_free_mib and utilization <= args.max_gpu_utilization else 0
            print(f"GPU availability {consecutive}/2: {free}/{total} MiB free; utilization {utilization}%", flush=True)
            if consecutive < 2:
                if not args.wait_for_idle:
                    raise RuntimeError("Pass --wait-for-idle to wait for two availability samples")
                time.sleep(args.poll_seconds)
    import torch
    from ultralytics import YOLO, __version__ as ultralytics_version
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    options = dict(data=str(dataset / "data.yaml"), epochs=args.epochs, patience=args.patience,
        imgsz=1280, batch=args.batch, workers=0, device=args.device, pretrained=True, optimizer="AdamW", lr0=args.learning_rate,
        warmup_bias_lr=args.warmup_bias_lr,
        seed=42, deterministic=True, amp=args.device != "cpu", cache=False, plots=True, save_period=10,
        close_mosaic=10, mosaic=0.25, scale=0.25, project=str(output.parent), name=output.name,
        exist_ok=True, verbose=True)
    output.mkdir(parents=True)
    started = time.perf_counter()
    config = {"model": str(model_path), "model_sha256": file_sha256(model_path),
        "dataset_yaml": str(dataset / "data.yaml"), "dataset_summary": json.loads((dataset / "summary.json").read_text(encoding="utf8")),
        "annotation_audit": audit, "torch": torch.__version__, "ultralytics": ultralytics_version,
        "python": __import__("sys").version, "device": args.device, "hyperparameters": options,
        "process_id": os.getpid(), "model_name": model_path.stem,
        "weights_source": "explicit --weights" if args.weights is not None else "tracking configuration",
        "training_started": True, "training_completed": False, "training_status": "running",
        "gpu_availability_criteria": {"minimum_free_mib": args.min_free_mib, "maximum_utilization_percent": args.max_gpu_utilization, "consecutive_samples": 2},
        "started_at": datetime.now().isoformat(), "test_split_used": False, "test_split_evaluated": False,
        "training_scope": audit.get("training_scope", "Static person detector only; train learns and val selects weights. No tracking, helmet or workwear training."),
        "export_input_contract": export_contract,
        "initialization": f"Pretrained {model_path.stem}; new optimizer, no previous training optimizer restored."}
    write_json(output / "training_config.json", config)
    model = YOLO(str(model_path))
    best_epoch_from_save = None

    def retain_resume(trainer):
        nonlocal best_epoch_from_save
        last = Path(trainer.last)
        if not last.is_file():
            return
        destination = Path(trainer.save_dir) / "weights/resume_latest.pt"
        temporary = destination.with_suffix(".tmp")
        shutil.copy2(last, temporary)
        temporary.replace(destination)
        if trainer.fitness == trainer.best_fitness:
            best_epoch_from_save = int(trainer.epoch) + 1
        write_json(output / "progress.json", {"epoch_completed": int(trainer.epoch) + 1,
            "max_epochs": int(trainer.epochs), "updated_at": datetime.now().isoformat(),
            "elapsed_seconds": time.perf_counter() - started, "validation_metrics": normalize_result_metrics(trainer.metrics),
            "fitness": float(trainer.fitness), "best_fitness": float(trainer.best_fitness), "best_epoch": best_epoch_from_save,
            "resume_checkpoint": str(destination), "test_split_evaluated": False})

    model.add_callback("on_model_save", retain_resume)
    try:
        result = model.train(**options)
        with (output / "results.csv").open(encoding="utf8", newline="") as stream:
            history = list(csv.DictReader(stream))
        best_path, last_path = output / "weights/best.pt", output / "weights/last.pt"
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        # final_eval overwrites best.pt train_results with the complete history.
        # Record the epoch at save time, rather than taking that history's last row.
        if best_epoch_from_save is None:
            raise RuntimeError("No best-checkpoint save event was recorded")
        best_epoch = best_epoch_from_save
        best_history_row = next(row for row in history if int(float(row["epoch"])) == best_epoch)
        final_metrics = {k.strip(): float(v) for k, v in history[-1].items() if k.strip().startswith(("metrics/", "val/"))}
        config.update(training_completed=True, training_status="completed", epochs_completed=len(history),
            early_stopped=len(history) < args.epochs, best_epoch=best_epoch,
            best_epoch_metrics={k.strip(): float(v) for k, v in best_history_row.items() if k.strip().startswith(("metrics/", "val/"))},
            best_epoch_metrics_source="results.csv row for the recorded best-checkpoint save epoch",
            final_epoch_metrics=final_metrics,
            final_epoch_metrics_source="results.csv final epoch; train return separately represents best revalidation",
            train_return_metrics=normalize_result_metrics(result), best_checkpoint=str(best_path),
            best_checkpoint_sha256=file_sha256(best_path), last_checkpoint=str(last_path), last_checkpoint_sha256=file_sha256(last_path),
            resume_checkpoint=str(output / "weights/resume_latest.pt"),
            resume_checkpoint_sha256=file_sha256(output / "weights/resume_latest.pt"), completed_at=datetime.now().isoformat())
    except BaseException as error:
        config.update(training_status="failed", failure=repr(error))
        raise
    finally:
        config["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "training_config.json", config)


if __name__ == "__main__":
    main()
