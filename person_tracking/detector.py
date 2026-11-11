from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen


def ensure_model(path):
    path = Path(path)
    if path.is_file():
        return path
    if path.name != "yolo11n.pt":
        raise FileNotFoundError(f"Supply a local trained model: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
    temp = path.with_suffix(".download")
    try:
        with urlopen(url, timeout=60) as response, temp.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        if temp.stat().st_size < 1_000_000:
            raise OSError("Model download is incomplete")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def select_device(request):
    import torch
    if request == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if request != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu or install CUDA PyTorch")
    return str(request)


def load_detector(cfg, device=None, image_size=None):
    from ultralytics import YOLO
    actual_device = select_device(device or cfg["device"])
    model = YOLO(str(ensure_model(cfg["model"])))
    if model.names.get(0) != "person":
        raise ValueError("Class 0 must be person; only single-person or COCO-compatible weights are supported")
    return model, {
        "imgsz": image_size or cfg["image_size"], "conf": cfg["detection_conf"],
        "iou": cfg["nms_iou"], "classes": [0], "device": actual_device,
        "quantize": 16 if actual_device != "cpu" else 32, "verbose": False, "save": False,
    }
