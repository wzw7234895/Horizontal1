from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from time import perf_counter

from person_tracking.common import PACKAGE_DIR, load_config, write_json
from person_tracking.experiments.train_person_static_v1 import normalize_result_metrics, query_gpu_status


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume the reviewed static-person YOLO11n run from its optimizer checkpoint and finish 80 epochs."
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--run-name", default="person_yolo11n_static_v1_full80_resume_20261105")
    parser.add_argument("--wait-for-idle", action="store_true")
    parser.add_argument("--min-free-mib", type=int, default=3500)
    parser.add_argument("--max-gpu-utilization", type=int, default=10)
    parser.add_argument("--idle-checks", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()

    if args.device.lower() == "cpu":
        parser.error("This continuation is configured for the available RTX 3060; use the original script for CPU runs.")
    if args.min_free_mib < 1 or not 0 <= args.max_gpu_utilization <= 100:
        parser.error("--min-free-mib must be positive and --max-gpu-utilization must be between 0 and 100")
    if args.idle_checks < 1 or args.poll_seconds <= 0:
        parser.error("--idle-checks and --poll-seconds must be positive")

    os.environ.setdefault("YOLO_CONFIG_DIR", str(PACKAGE_DIR / ".cache" / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(PACKAGE_DIR / ".cache" / "matplotlib"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")

    cfg = load_config()
    dataset = PACKAGE_DIR / "data" / "person_reviewed_v1"
    dataset_yaml = dataset / "data.yaml"
    source_run = PACKAGE_DIR / "runs" / "train" / "person_yolo11n_static_v1_20261105"
    resume_checkpoint = source_run / "weights" / "epoch40.pt"
    source_best = source_run / "weights" / "best.pt"
    source_results = source_run / "results.csv"
    output = PACKAGE_DIR / "runs" / "train" / args.run_name

    if output.exists():
        raise FileExistsError(f"Choose a new --run-name; refusing to overwrite {output}")
    for required in (dataset_yaml, resume_checkpoint, source_best, source_results):
        if not required.exists():
            raise FileNotFoundError(required)

    idle_samples = 0
    while idle_samples < args.idle_checks:
        free_mib, total_mib, utilization = query_gpu_status(args.device)
        is_idle = free_mib >= args.min_free_mib and utilization <= args.max_gpu_utilization
        idle_samples = idle_samples + 1 if is_idle else 0
        if idle_samples >= args.idle_checks:
            break
        if not args.wait_for_idle:
            raise RuntimeError(
                f"GPU is not idle: {free_mib}/{total_mib} MiB free, utilization {utilization}%; "
                "pass --wait-for-idle to poll"
            )
        print(
            f"GPU idle check {idle_samples}/{args.idle_checks}: {free_mib}/{total_mib} MiB free, "
            f"utilization {utilization}% (need >= {args.min_free_mib} MiB free and "
            f"<= {args.max_gpu_utilization}%); checking again in {args.poll_seconds:g}s.",
            flush=True,
        )
        time.sleep(args.poll_seconds)

    import torch
    from ultralytics import YOLO, __version__ as ultralytics_version

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    checkpoint = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("optimizer") is None or checkpoint.get("scaler") is None:
        raise RuntimeError(f"Checkpoint is not resumable (optimizer/scaler missing): {resume_checkpoint}")
    if int(checkpoint.get("epoch", -1)) != 40:
        raise RuntimeError(f"Expected the epoch-40 optimizer checkpoint, got epoch={checkpoint.get('epoch')}")
    history = checkpoint.get("train_results") or {}
    history_epochs = len(history.get("epoch", []))
    if history_epochs != 41:
        raise RuntimeError(f"Expected 41 completed epochs in resume history, got {history_epochs}")

    with source_results.open(encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    if len(source_rows) < history_epochs:
        raise RuntimeError("Source results.csv is shorter than the optimizer checkpoint history")
    fields = list(source_rows[0])
    for index, row in enumerate(source_rows[:history_epochs]):
        expected_epoch = int(history["epoch"][index])
        if int(row["epoch"]) != expected_epoch:
            raise RuntimeError(
                f"Source results.csv diverges from resume checkpoint at row {index}: "
                f"csv={row['epoch']} checkpoint={expected_epoch}"
            )

    output.mkdir(parents=True)
    weights_dir = output / "weights"
    weights_dir.mkdir()
    shutil.copy2(source_best, weights_dir / "best.pt")
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(source_rows[:history_epochs])

    dataset_summary = json.loads((dataset / "summary.json").read_text(encoding="utf-8"))
    dataset_summary.pop("training_started", None)
    run_config = {
        "model": str(resume_checkpoint.resolve()),
        "source_pretrained_model": str(Path(cfg["model"]).resolve()),
        "source_run": str(source_run.resolve()),
        "resume_checkpoint_sha256": file_sha256(resume_checkpoint),
        "source_best_checkpoint_sha256": file_sha256(source_best),
        "dataset_yaml": str(dataset_yaml.resolve()),
        "dataset_summary": dataset_summary,
        "ultralytics": ultralytics_version,
        "torch": torch.__version__,
        "device": args.device,
        "training_split": "train",
        "early_stopping_and_model_selection_split": "val",
        "test_split_used": False,
        "test_split_evaluated": False,
        "training_started": False,
        "training_completed": False,
        "training_status": "prepared_to_resume",
        "resume_checkpoint_epoch": 40,
        "history_epochs_preserved": history_epochs,
        "target_total_epochs": 80,
        "early_stopping_patience": 100,
        "gpu_idle_criteria": {
            "minimum_free_mib": args.min_free_mib,
            "maximum_utilization_percent": args.max_gpu_utilization,
            "consecutive_samples": args.idle_checks,
            "poll_seconds": args.poll_seconds,
        },
        "hyperparameters": {
            "epochs": 80,
            "patience": 100,
            "imgsz": 1280,
            "batch": 1,
            "workers": 0,
            "optimizer": "AdamW (state restored from epoch-40 checkpoint)",
            "lr0": 0.001,
            "seed": 42,
            "deterministic": True,
            "amp": True,
            "mosaic": 0.25,
            "scale": 0.25,
            "close_mosaic": 10,
        },
        "notes": [
            "Resume from the epoch-40 optimizer/scaler checkpoint; the saved history contains epochs 1-41.",
            "The original epoch-42 early-stop run is preserved in its original directory; this run replays epoch 42 from the epoch-40 state and continues to epoch 80.",
            "The original best.pt is seeded into the new run so it remains the selected checkpoint unless a later epoch improves validation fitness.",
            "Only train and val are used; test remains sealed.",
            "This is static person detection, not tracking, helmet, hook, workpiece, or height training.",
        ],
        "output_directory": str(output.resolve()),
        "history_csv_rows_copied": history_epochs,
    }
    write_json(output / "training_config.json", run_config)

    started = perf_counter()
    from datetime import datetime
    run_config.update(training_started=True, training_status="running", started_at=datetime.now().isoformat())
    write_json(output / "training_config.json", run_config)
    model = YOLO(str(resume_checkpoint.resolve()))
    try:
        result = model.train(
            data=str(dataset_yaml.resolve()),
            epochs=80,
            patience=100,
            imgsz=1280,
            batch=1,
            workers=0,
            device=args.device,
            resume=str(resume_checkpoint.resolve()),
            save_dir=str(output.resolve()),
            save_period=10,
            plots=True,
            cache=False,
            verbose=True,
        )
    except Exception as error:
        run_config.update(
            training_completed=False,
            elapsed_seconds=perf_counter() - started,
            failure=repr(error),
        )
        write_json(output / "training_config.json", run_config)
        raise

    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"
    if not best_path.is_file() or not last_path.is_file():
        raise FileNotFoundError("Full-80 continuation did not produce both best.pt and last.pt")
    last_checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    final_history = last_checkpoint.get("train_results") or {}
    epochs_completed = len(final_history.get("epoch", []))
    best_metrics = best_checkpoint.get("train_metrics") or {}
    if epochs_completed != 80:
        raise RuntimeError(f"Expected 80 completed epochs, found {epochs_completed}")
    if not best_metrics:
        raise RuntimeError("Best checkpoint did not contain validation metrics")
    final_epoch_metrics = {}
    for key in (
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
    ):
        values = final_history.get(key)
        if values:
            final_epoch_metrics[key] = float(values[-1])
    best_index = min(
        range(len(final_history["epoch"])),
        key=lambda i: abs(final_history["metrics/mAP50(B)"][i] - best_metrics["metrics/mAP50(B)"])
        + abs(final_history["metrics/mAP50-95(B)"][i] - best_metrics["metrics/mAP50-95(B)"]),
    )
    best_epoch = int(final_history["epoch"][best_index])
    run_config.update(
        training_completed=True,
        training_status="completed_full_80_epochs",
        early_stopped=False,
        epochs_completed=epochs_completed,
        best_epoch=best_epoch,
        best_epoch_metrics={k: float(v) for k, v in best_metrics.items() if isinstance(v, (int, float))},
        final_epoch_metrics=final_epoch_metrics,
        final_epoch_metrics_source="last.pt train_results final entry; cross-check against results.csv",
        train_return_metrics=normalize_result_metrics(result),
        elapsed_seconds=perf_counter() - started,
        completed_at=datetime.now().isoformat(),
        best_checkpoint=str(best_path.resolve()),
        best_checkpoint_sha256=file_sha256(best_path),
        last_checkpoint=str(last_path.resolve()),
        last_checkpoint_sha256=file_sha256(last_path),
        resume_history_epochs=epochs_completed,
    )
    write_json(output / "training_config.json", run_config)


if __name__ == "__main__":
    main()
