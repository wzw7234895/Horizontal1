"""Conservative floor candidates from a locally stored ADE20K model."""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


@lru_cache(maxsize=2)
def _load_model(model_dir: str):
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    directory = Path(model_dir)
    if not (directory / "config.json").is_file() or not (
        (directory / "model.safetensors").is_file()
        or (directory / "pytorch_model.bin").is_file()
    ):
        raise ValueError("地面分割模型目录缺少配置或权重文件")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The following named arguments are not valid.*")
        processor = SegformerImageProcessor.from_pretrained(model_dir, local_files_only=True)
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_dir, local_files_only=True).eval().to(device)
    labels = {int(index): name.lower() for index, name in model.config.id2label.items()}
    floor_ids = [index for index, name in labels.items() if name == "floor"]
    if not floor_ids:
        raise ValueError("分割模型未提供 floor 类别，不能用于地面候选判断")
    return processor, model, device, labels, floor_ids


def _point_decision(floor_probability: np.ndarray, labels: np.ndarray,
                    class_names: dict[int, str], point: np.ndarray,
                    threshold: float) -> dict:
    x, y = np.rint(point).astype(int)
    height, width = floor_probability.shape
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("选点超出地面分割图像")
    radius = max(3, round(min(height, width) * 0.01))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    patch = floor_probability[y0:y1, x0:x1]
    local_support = float(np.mean(patch >= threshold))
    confidence = float(floor_probability[y, x])
    label = class_names.get(int(labels[y, x]), "unknown")
    candidate = (confidence >= threshold and local_support >= 0.8
                 and min(x, y, width - x - 1, height - y - 1) >= radius)
    return {"pixel": [float(point[0]), float(point[1])],
            "status": "ground_candidate_unverified" if candidate else "unknown",
            "floor_probability": confidence,
            "local_floor_fraction": local_support,
            "model_label": label}


def segment_ground(image: np.ndarray, points: np.ndarray | None,
                   model_dir: str | Path, threshold: float = 0.75) -> tuple[dict, np.ndarray]:
    """Return floor candidates, never a confirmed physical ground surface."""
    import torch
    import torch.nn.functional as F

    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("地面分割要求 BGR 图像和 0 到 1 之间的置信度阈值")
    if points is not None:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("地面检查点必须是有限的像素坐标")
    processor, model, device, labels, floor_ids = _load_model(str(Path(model_dir).resolve()))
    inputs = processor(images=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                       return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = logits.softmax(dim=1)
        floor_small = probabilities[:, floor_ids].sum(dim=1)[0].cpu().numpy()
        predicted_small = probabilities.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    width, height = image.shape[1], image.shape[0]
    floor = cv2.resize(floor_small, (width, height), interpolation=cv2.INTER_LINEAR)
    predicted = cv2.resize(predicted_small, (width, height), interpolation=cv2.INTER_NEAREST)
    checks = [] if points is None else [
        _point_decision(floor, predicted, labels, point, threshold) for point in points]
    result = {"status": "diagnostic_only", "method": "segformer_ade20k_floor_candidate",
              "model_dir": str(Path(model_dir).resolve()), "image_size": [image.shape[1], image.shape[0]],
              "floor_class_ids": floor_ids, "threshold": threshold,
              "floor_candidate_fraction": float(np.mean(floor >= threshold)),
              "point_checks": checks,
              "all_points_ground_candidates": bool(checks) and all(
                  check["status"] == "ground_candidate_unverified" for check in checks),
              "limitations": ["ADE20K floor is not validated for this factory or overhead views.",
                              "A floor label does not prove that the visible surface is at ground height.",
                              "Ground candidates cannot establish 3D depth or replace site calibration."]}
    return result, (floor >= threshold).astype(np.uint8) * 255


def save_ground_overlay(image: np.ndarray, mask: np.ndarray, path: str | Path,
                        points: np.ndarray | None = None) -> None:
    if mask.shape != image.shape[:2]:
        raise ValueError("地面掩码尺寸与原图不一致")
    canvas = image.copy()
    tint = canvas.copy()
    tint[mask > 0] = (35, 200, 60)
    canvas = cv2.addWeighted(canvas, 0.55, tint, 0.45, 0)
    if points is not None:
        for index, point in enumerate(points):
            x, y = np.rint(point).astype(int)
            cv2.circle(canvas, (x, y), 9, (0, 255, 255), 2)
            cv2.putText(canvas, chr(ord("A") + index), (x + 12, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg"):
        raise ValueError("地面分割叠加图必须为 PNG 或 JPG")
    ok, encoded = cv2.imencode(suffix, canvas)
    if not ok:
        raise ValueError("地面分割叠加图编码失败")
    encoded.tofile(str(destination))
