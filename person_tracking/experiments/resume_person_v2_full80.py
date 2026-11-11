"""Continue an early-stopped audited person run to 80 total epochs."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import time

from person_tracking.common import PACKAGE_DIR, write_json
from person_tracking.experiments.train_person_static_v1 import file_sha256, normalize_result_metrics, query_gpu_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--wait-for-idle", action="store_true")
    parser.add_argument("--min-free-mib", type=int, default=3500)
    parser.add_argument("--max-gpu-utilization", type=int, default=10)
    args = parser.parse_args()
    if args.min_free_mib <= 0 or not 0 <= args.max_gpu_utilization <= 100:
        parser.error("Invalid GPU availability thresholds")
    source, output = args.source_run.resolve(), args.out.resolve()
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((source / "training_config.json").read_text(encoding="utf8"))
    if not config["training_completed"] or not config["early_stopped"]:
        raise ValueError("Source must be a completed early-stopped person run")
    checkpoint = source / "weights/resume_latest.pt"
    if file_sha256(checkpoint) != config["resume_checkpoint_sha256"]:
        raise ValueError("Source optimizer checkpoint changed")
    dataset = Path(config["dataset_yaml"]).parent
    audit = config["annotation_audit"]
    fingerprints_path = Path(audit["fingerprints_file"])
    if file_sha256(fingerprints_path) != audit["fingerprints_sha256"]:
        raise ValueError("Audit fingerprints changed")
    for item in json.loads(fingerprints_path.read_text(encoding="utf8")):
        for kind, extension in (("images", ".jpg"), ("labels", ".txt")):
            path = dataset / kind / item["split"] / (item["id"] + extension)
            key = "image_sha256" if kind == "images" else "label_sha256"
            if file_sha256(path) != item[key]:
                raise ValueError(f"Training snapshot changed: {path}")
    if args.device != "cpu":
        consecutive = 0
        while consecutive < 2:
            free, total, utilization = query_gpu_status(args.device)
            consecutive = consecutive + 1 if free >= args.min_free_mib and utilization <= args.max_gpu_utilization else 0
            print(f"GPU availability {consecutive}/2: {free}/{total} MiB free; utilization {utilization}%", flush=True)
            if consecutive < 2:
                if not args.wait_for_idle:
                    raise RuntimeError("Pass --wait-for-idle to wait for GPU availability")
                time.sleep(15)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PACKAGE_DIR / ".cache/ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(PACKAGE_DIR / ".cache/matplotlib"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    import torch
    from ultralytics import YOLO
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    completed = int(saved["epoch"]) + 1
    if saved.get("optimizer") is None or saved.get("scaler") is None:
        raise ValueError("Optimizer/scaler is missing from continuation checkpoint")
    if completed != config["epochs_completed"] or not 0 < completed < 80:
        raise ValueError("Checkpoint epoch does not match the source run or already reached 80")
    with (source / "results.csv").open(encoding="utf8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != completed or len(saved["train_results"]["epoch"]) != completed:
        raise ValueError("Checkpoint and source history do not match")
    output.mkdir(parents=True)
    (output / "weights").mkdir()
    shutil.copy2(source / "weights/best.pt", output / "weights/best.pt")
    shutil.copy2(source / "results.csv", output / "results.csv")
    shutil.copy2(source / "training_config.json", output / "source_training_config.json")
    prior_seconds = config["elapsed_seconds"]
    best_epoch = config["best_epoch"]
    config = {**config, "source_run": str(source), "resume_checkpoint_input": str(checkpoint),
        "resume_checkpoint_input_sha256": file_sha256(checkpoint), "history_epochs_preserved": completed,
        "source_training_seconds": prior_seconds, "training_completed": False, "training_status": "running",
        "started_at": datetime.now().isoformat(), "early_stopped": False,
        "process_id": os.getpid(),
        "initialization": "Continue the source optimizer/scaler/EMA state; preserve source best and epoch history.",
        "hyperparameters": {**config["hyperparameters"], "patience": 100, "resume": str(checkpoint), "save_dir": str(output)}}
    for key in ("completed_at", "failure", "best_checkpoint_sha256", "last_checkpoint_sha256", "resume_checkpoint_sha256"):
        config.pop(key, None)
    write_json(output / "training_config.json", config)
    started = time.perf_counter()
    model = YOLO(str(checkpoint))

    def retain_resume(trainer):
        nonlocal best_epoch
        temporary = output / "weights/resume_latest.tmp"
        shutil.copy2(trainer.last, temporary)
        temporary.replace(output / "weights/resume_latest.pt")
        if trainer.fitness == trainer.best_fitness:
            best_epoch = int(trainer.epoch) + 1
        write_json(output / "progress.json", {"epoch_completed": int(trainer.epoch) + 1, "max_epochs": 80,
            "best_epoch": best_epoch, "fitness": float(trainer.fitness), "best_fitness": float(trainer.best_fitness),
            "validation_metrics": normalize_result_metrics(trainer.metrics),
            "elapsed_seconds": time.perf_counter() - started, "updated_at": datetime.now().isoformat()})

    model.add_callback("on_model_save", retain_resume)
    try:
        result = model.train(data=str(dataset / "data.yaml"), resume=str(checkpoint), epochs=80,
            patience=100, save_dir=str(output), save_period=10, workers=0, device=args.device,
            plots=True, cache=False, verbose=True)
        with (output / "results.csv").open(encoding="utf8", newline="") as stream:
            history = list(csv.DictReader(stream))
        if len(history) != 80:
            raise RuntimeError(f"Expected 80 total epochs, found {len(history)}")
        best_path, last_path = output / "weights/best.pt", output / "weights/last.pt"
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        best_history_row = next(row for row in history if int(float(row["epoch"])) == best_epoch)
        config.update(training_completed=True, training_status="completed_full_80_epochs", epochs_completed=80,
            best_epoch=best_epoch,
            best_epoch_metrics={k.strip(): float(v) for k, v in best_history_row.items() if k.strip().startswith(("metrics/", "val/"))},
            best_epoch_metrics_source="results.csv row for the recorded best-checkpoint save epoch",
            final_epoch_metrics={k.strip(): float(v) for k, v in history[-1].items() if k.strip().startswith(("metrics/", "val/"))},
            final_epoch_metrics_source="results.csv final row", train_return_metrics=normalize_result_metrics(result),
            best_checkpoint=str(best_path), best_checkpoint_sha256=file_sha256(best_path),
            last_checkpoint=str(last_path), last_checkpoint_sha256=file_sha256(last_path),
            resume_checkpoint=str(output / "weights/resume_latest.pt"),
            resume_checkpoint_sha256=file_sha256(output / "weights/resume_latest.pt"), completed_at=datetime.now().isoformat())
    except BaseException as error:
        config.update(training_status="failed", failure=repr(error))
        raise
    finally:
        config["elapsed_seconds"] = time.perf_counter() - started
        config["combined_training_seconds"] = prior_seconds + config["elapsed_seconds"]
        write_json(output / "training_config.json", config)


if __name__ == "__main__":
    main()
