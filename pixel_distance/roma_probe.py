"""Offline diagnostic for dense cross-view matches at user-selected pixels."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .auto3d import _fundamental_from_pose
from .geometry import load_image, load_json, save_json
from .pose_state import state_from_nominal_fixed_rig


def _sample(warp, certainty, pixel: np.ndarray, source_shape: tuple[int, int],
            target_shape: tuple[int, int]) -> tuple[np.ndarray, float]:
    import torch
    import torch.nn.functional as F

    height, width = source_shape
    target_height, target_width = target_shape
    grid = torch.tensor([[[[2 * pixel[0] / width - 1,
                            2 * pixel[1] / height - 1]]]],
                        device=warp.device, dtype=warp.dtype)
    mapped = F.grid_sample(warp[0, :, :, 2:].permute(2, 0, 1)[None], grid,
                           align_corners=False)[0, :, 0, 0]
    confidence = F.grid_sample(certainty[0][None, None], grid,
                               align_corners=False)[0, 0, 0, 0]
    location = mapped.detach().float().cpu().numpy()
    return np.array([(location[0] + 1) * target_width / 2,
                     (location[1] + 1) * target_height / 2]), float(confidence)


@lru_cache(maxsize=1)
def _model(resolution: int):
    import torch
    from loguru import logger
    from romatch import roma_outdoor

    logger.disable("romatch")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Local correlation is not supported on non-Linux platforms.*")
        return roma_outdoor(device="cuda", coarse_res=resolution,
                            upsample_res=resolution, symmetric=False,
                            use_custom_corr=False)


def probe_with_entries(reference: np.ndarray, target: np.ndarray, points: np.ndarray,
                       entry_a: dict, entry_b: dict, resolution: int = 448) -> dict:
    import torch
    from PIL import Image

    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError("需要参考图中的 A/B 两个像素点")
    if np.any(points < 0) or np.any(points[:, 0] >= reference.shape[1]) or np.any(points[:, 1] >= reference.shape[0]):
        raise ValueError("A/B 像素点必须在参考图像内")
    if resolution < 224 or resolution % 14:
        raise ValueError("模型分辨率须为不小于 224 的 14 的倍数")
    if not torch.cuda.is_available():
        raise ValueError("完整 RoMa 离线诊断需要 CUDA GPU")
    fundamental = _fundamental_from_pose(entry_a, entry_b)
    model = _model(resolution)
    pil_a = Image.fromarray(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
    pil_b = Image.fromarray(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
    with torch.inference_mode():
        forward_warp, forward_cert = model.match(pil_a, pil_b, device="cuda")
        predicted = [_sample(forward_warp, forward_cert, point,
                             reference.shape[:2], target.shape[:2])
                     for point in points]
        del forward_warp, forward_cert
        reverse_warp, reverse_cert = model.match(pil_b, pil_a, device="cuda")
        checked = []
        for point, (target_pixel, forward_conf) in zip(points, predicted):
            returned, reverse_conf = _sample(reverse_warp, reverse_cert,
                                             target_pixel, target.shape[:2],
                                             reference.shape[:2])
            line = fundamental @ np.array([point[0], point[1], 1.0])
            epipolar_error = abs(float(line @ np.r_[target_pixel, 1.0])) / math.hypot(*line[:2])
            visual_consistent = (forward_conf >= 0.05 and reverse_conf >= 0.05
                                 and np.linalg.norm(returned - point) <= 3)
            nominal_geometry_consistent = epipolar_error <= 3
            checked.append({"reference_pixel": point.tolist(),
                            "target_candidate_pixel": target_pixel.tolist(),
                            "forward_confidence": forward_conf,
                            "reverse_confidence": reverse_conf,
                            "round_trip_error_px": float(np.linalg.norm(returned - point)),
                            "nominal_epipolar_error_px": epipolar_error,
                            "visual_consistent": bool(visual_consistent),
                            "nominal_geometry_consistent": bool(nominal_geometry_consistent)})
    return {"status": "diagnostic_only", "model": "RoMa outdoor",
            "model_resolution": resolution, "points": checked,
            "both_points_visually_consistent": all(p["visual_consistent"] for p in checked),
            "both_points_nominal_geometry_consistent": all(
                p["nominal_geometry_consistent"] for p in checked),
            "height_difference_m": None,
            "note": "候选点尚未验证为同一物理点；名义位姿未标定，不能据此直接给出高度差。"}


def probe(reference: np.ndarray, target: np.ndarray, points: np.ndarray,
          config: dict, reference_id: str, target_id: str,
          resolution: int = 448) -> dict:
    state = state_from_nominal_fixed_rig(
        {reference_id: reference, target_id: target}, config)
    result = probe_with_entries(reference, target, points,
                                state["cameras"][reference_id],
                                state["cameras"][target_id], resolution)
    return {**result, "reference_camera": reference_id,
            "target_camera": target_id}


def main() -> None:
    parser = argparse.ArgumentParser(description="RoMa 跨视角选点离线诊断")
    parser.add_argument("--reference", required=True, choices=["D01", "D02", "D03", "D04"])
    parser.add_argument("--reference-image", required=True)
    parser.add_argument("--target", required=True, choices=["D01", "D02", "D03", "D04"])
    parser.add_argument("--target-image", required=True)
    parser.add_argument("--point-a", required=True)
    parser.add_argument("--point-b", required=True)
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--config", default=str(Path(__file__).with_name("camera_presets.json")))
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.reference == args.target:
        parser.error("需要两个不同的相机")
    try:
        points = np.array([[float(value) for value in arg.split(",")]
                           for arg in (args.point_a, args.point_b)], dtype=float)
        result = probe(load_image(args.reference_image), load_image(args.target_image),
                       points, load_json(args.config), args.reference, args.target,
                       args.resolution)
    except (ValueError, ImportError, RuntimeError, cv2.error) as exc:
        parser.exit(2, f"错误：{exc}\n")
    if args.out:
        save_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
