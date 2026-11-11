from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import time

from person_tracking.common import PACKAGE_DIR, load_config, write_json


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_result_metrics(result) -> dict[str, float]:
    """Normalize metrics returned by Ultralytics, which is a DetMetrics object."""
    values = getattr(result, "results_dict", result)
    if not isinstance(values, dict):
        return {}
    normalized = {}
    for key, value in values.items():
        try:
            normalized[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def gpu_index_for_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized == "cuda":
        return "0"
    if normalized.startswith("cuda:"):
        normalized = normalized.split(":", 1)[1]
    if not normalized.isdigit():
        raise ValueError(f"GPU device must be an index such as 0 or cuda:0, got {device!r}")
    return normalized


def query_gpu_status(device: str) -> tuple[int, int, int]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi was not found; cannot check GPU availability without initializing CUDA")
    result = subprocess.run(
        [executable, "--id", gpu_index_for_device(device),
         "--query-gpu=memory.free,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    rows = list(csv.reader(line for line in result.stdout.splitlines() if line.strip()))
    if len(rows) != 1 or len(rows[0]) != 3:
        raise RuntimeError(f"Unexpected nvidia-smi output: {result.stdout.strip()!r}")
    free_mib, total_mib, utilization = (int(round(float(value.strip()))) for value in rows[0])
    return free_mib, total_mib, utilization


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11n on the reviewed static person dataset.")
    parser.add_argument("--device", default="0", help="Use CUDA device index (default: 0); CPU is also accepted.")
    parser.add_argument("--run-name", default="person_yolo11n_static_v1_20261105")
    parser.add_argument("--wait-for-idle", action="store_true",
                        help="Poll until the GPU has enough free memory and low utilization for two checks.")
    parser.add_argument("--min-free-mib", type=int, default=3500,
                        help="Minimum free GPU memory required before training (default: 3500 MiB).")
    parser.add_argument("--max-gpu-utilization", type=int, default=10,
                        help="Maximum GPU utilization for an idle check (default: 10 percent).")
    parser.add_argument("--idle-checks", type=int, default=2,
                        help="Consecutive idle samples required before training (default: 2).")
    parser.add_argument("--poll-seconds", type=float, default=30,
                        help="Seconds between GPU idle samples (default: 30).")
    args = parser.parse_args()

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
    output_root = PACKAGE_DIR / "runs" / "train"
    output = output_root / args.run_name
    if output.exists():
        raise FileExistsError(f"Choose a new --run-name; refusing to overwrite {output}")
    for required in (dataset_yaml, dataset / "images" / "train", dataset / "images" / "val",
                     dataset / "labels" / "train", dataset / "labels" / "val", Path(cfg["model"])):
        if not required.exists():
            raise FileNotFoundError(required)

    device = args.device
    use_amp = device.lower() != "cpu"
    if device.lower() != "cpu":
        idle_samples = 0
        while True:
            free_mib, total_mib, utilization = query_gpu_status(device)
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
                f"GPU idle check {idle_samples}/{args.idle_checks}: "
                f"{free_mib}/{total_mib} MiB free, utilization {utilization}% "
                f"(need >= {args.min_free_mib} MiB free and <= {args.max_gpu_utilization}%); "
                f"checking again in {args.poll_seconds:g}s.",
                flush=True,
            )
            time.sleep(args.poll_seconds)

    from ultralytics import YOLO, __version__ as ultralytics_version
    import torch
    if device.lower() != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    batch_size = 1  # 6 GiB RTX 3060 Laptop; keep headroom at 1280-pixel input size.
    dataset_summary = json.loads((dataset / "summary.json").read_text(encoding="utf-8"))
    dataset_summary.pop("training_started", None)

    data = {
        "model": str(Path(cfg["model"]).resolve()),
        "model_sha256": file_sha256(Path(cfg["model"])),
        "dataset_yaml": str(dataset_yaml.resolve()),
        "dataset_summary": dataset_summary,
        "ultralytics": ultralytics_version,
        "torch": torch.__version__,
        "device": device,
        "training_split": "train",
        "early_stopping_and_model_selection_split": "val",
        "test_split_used": False,
        "training_started": False,
        "gpu_idle_criteria": None if device.lower() == "cpu" else {
            "minimum_free_mib": args.min_free_mib,
            "maximum_utilization_percent": args.max_gpu_utilization,
            "consecutive_samples": args.idle_checks,
            "poll_seconds": args.poll_seconds,
        },
        "hyperparameters": {
            "epochs": 80,
            "patience": 15,
            "imgsz": 1280,
            "batch": batch_size,
            "workers": 0,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "seed": 42,
            "deterministic": True,
            "amp": use_amp,
            "mosaic": 0.25,
            "scale": 0.25,
            "close_mosaic": 10,
        },
        "notes": [
            "Small pilot fine-tune from public pretrained YOLO11n weights.",
            "Validation set is used for early stopping and model selection; test set stays sealed.",
            "No temporal IDs, workpiece boxes, or B5 event labels are used in this detector training.",
            "A completed checkpoint remains a PyTorch model until exported and validated for the target edge runtime.",
        ],
    }
    output.mkdir(parents=True)
    started = perf_counter()
    from datetime import datetime
    data.update(training_started=True, started_at=datetime.now().isoformat(),
                output_directory=str(output.resolve()))
    write_json(output / "training_config.json", data)

    model = YOLO(str(Path(cfg["model"]).resolve()))
    try:
        result = model.train(
            data=str(dataset_yaml.resolve()),
            epochs=80,
            patience=15,
            imgsz=1280,
            batch=batch_size,
            workers=0,
            device=device,
            pretrained=True,
            optimizer="AdamW",
            lr0=0.001,
            seed=42,
            deterministic=True,
            amp=use_amp,
            cache=False,
            plots=True,
            save_period=10,
            close_mosaic=10,
            mosaic=0.25,
            scale=0.25,
            project=str(output_root.resolve()),
            name=args.run_name,
            exist_ok=True,  # output was verified absent immediately before creation
            verbose=True,
        )
    except Exception as error:
        data.update(training_completed=False, elapsed_seconds=perf_counter() - started,
                    failure=repr(error))
        write_json(output / "training_config.json", data)
        raise
    best_checkpoint = output / "weights" / "best.pt"
    last_checkpoint = output / "weights" / "last.pt"
    data.update(
        training_started=True,
        training_completed=True,
        elapsed_seconds=perf_counter() - started,
        completed_at=datetime.now().isoformat(),
        best_checkpoint=str(best_checkpoint.resolve()),
        best_checkpoint_sha256=file_sha256(best_checkpoint),
        last_checkpoint=str(last_checkpoint.resolve()),
        last_checkpoint_sha256=file_sha256(last_checkpoint),
        result_metrics=normalize_result_metrics(result),
    )
    write_json(output / "training_config.json", data)


if __name__ == "__main__":
    main()
