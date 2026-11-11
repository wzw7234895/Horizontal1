from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from .common import load_config, write_json


def run_doctor(args):
    cfg = load_config(args.config)
    packages, errors = {}, []
    for package in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "scipy", "lap", "PyYAML"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing package: {package}")
    if packages.get("ultralytics") not in (None, "8.4.202"):
        errors.append("This code targets the Ultralytics 8.4.202 tracker API; install requirements.txt")
    device = {"cuda_available": False}
    try:
        import torch
        import torchvision
        import cv2
        device = {"cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
                  "opencv_import_version": cv2.__version__}
        if device["cuda_available"]:
            device["gpu"] = torch.cuda.get_device_name(0)
            device["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except ImportError as exc:
        errors.append(str(exc))
    for camera in cfg["cameras"]:
        if "://" not in camera["source"] and not Path(camera["source"]).is_file():
            errors.append(f"{camera['id']}: local video is missing")
    if not Path(cfg["model"]).is_file():
        errors.append("Weights missing; track/sample will download the official YOLO11n on first use")
    report = {"python": sys.executable, "python_version": platform.python_version(), "packages": packages,
              "device": device, "model": cfg["model"], "cameras": [c["id"] for c in cfg["cameras"]],
              "errors": errors, "ok": not errors}
    if args.out:
        write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)
