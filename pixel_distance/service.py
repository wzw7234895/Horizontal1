"""Stable application-facing API for the distance measurement engine."""

import copy
import hashlib
import json
import math
import threading
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .geometry import load_json
from .video_distance import measure_video_views


CAMERA_IDS = ("D01", "D02", "D03", "D04")


def validate_camera_config(config: dict) -> None:
    if not isinstance(config, dict) or not isinstance(config.get("cameras"), dict):
        raise ValueError("配置必须包含 cameras 对象")
    cameras = config["cameras"]
    if set(cameras) != set(CAMERA_IDS):
        raise ValueError("配置必须且只能包含 D01、D02、D03、D04")
    groups: dict[str, list[str]] = {}
    for camera_id, camera in cameras.items():
        group = camera.get("rig")
        if not isinstance(group, str) or not group:
            raise ValueError(f"{camera_id}.rig 必须是非空组名")
        groups.setdefault(group, []).append(camera_id)
        for key, lower, upper in (("height_m", 0.1, 100.0),
                                  ("horizontal_fov_deg", 1.0, 169.0),
                                  ("down_tilt_deg", 0.1, 90.0)):
            try:
                value = float(camera[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{camera_id}.{key} 缺失或无效") from exc
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{camera_id}.{key} 超出有效范围")
        center = np.asarray(camera.get("center_in_rig_m"), dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError(f"{camera_id}.center_in_rig_m 必须是有限的 x,y,z 米坐标")
    if len(groups) != 2 or sorted(map(len, groups.values())) != [2, 2]:
        raise ValueError("必须配置两个固定相机组，每组两台相机")
    if config.get("all_cameras_rigid") is not False:
        raise ValueError("当前接口要求 all_cameras_rigid=false；不允许假设两组相对位置固定")
    offsets = config.get("video_clock_offset_s", {})
    if not isinstance(offsets, dict):
        raise ValueError("video_clock_offset_s 必须是对象")
    for camera_id, value in offsets.items():
        if camera_id not in CAMERA_IDS or not math.isfinite(float(value)):
            raise ValueError("视频时钟偏移仅接受已知相机的有限秒数")


def _config_revision(config: dict) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _finite_point(point, name: str, width: int, height: int) -> list[float]:
    try:
        values = np.asarray(point, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是 [x,y] 像素坐标") from exc
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError(f"{name} 必须是两个有限像素坐标")
    if values[0] < 0 or values[1] < 0 or values[0] >= width or values[1] >= height:
        raise ValueError(f"{name} 超出参考图像范围 {width}x{height}")
    return values.tolist()


class DistanceService:
    """Thread-safe, stateless-per-request wrapper around the measurement engine."""

    def __init__(self, config: dict | str | Path | None = None):
        if config is None:
            config = Path(__file__).with_name("camera_presets.json")
        self._config = copy.deepcopy(load_json(config) if isinstance(config, (str, Path))
                                     else config)
        validate_camera_config(self._config)
        self._lock = threading.RLock()

    def get_config(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._config)

    def get_camera_info(self) -> dict:
        config = self.get_config()
        return {camera_id: {
            "name": camera["name"], "mount": camera["mount"],
            "group": camera["rig"], "height_m": camera["height_m"],
            "horizontal_fov_deg": camera["horizontal_fov_deg"],
            "down_tilt_deg": camera["down_tilt_deg"],
            "roll_deg": camera.get("roll_deg", 0.0),
            "center_in_rig_m": camera["center_in_rig_m"],
            "calibration_status": "nominal_unverified",
        } for camera_id, camera in config["cameras"].items()}

    def update_config(self, config: dict) -> str:
        candidate = copy.deepcopy(config)
        validate_camera_config(candidate)
        with self._lock:
            self._config = candidate
            return _config_revision(candidate)

    def measure_images(self, images: dict[str, np.ndarray], reference_camera: str,
                       point_a, point_b, *, timestamp: str | None = None,
                       matcher: str = "auto", pose_state: dict | None = None,
                       height_alignment: dict | None = None) -> dict:
        request_id = str(uuid.uuid4())
        if not isinstance(images, dict) or set(images) != set(CAMERA_IDS):
            missing = sorted(set(CAMERA_IDS) - set(images or {}))
            extra = sorted(set(images or {}) - set(CAMERA_IDS))
            raise ValueError(f"必须提供四路图像；缺少 {missing}，未知 {extra}")
        if reference_camera not in CAMERA_IDS:
            raise ValueError("reference_camera 必须是 D01、D02、D03 或 D04")
        if matcher not in ("auto", "sift", "loftr", "xfeat"):
            raise ValueError("matcher 仅接受 auto、sift、loftr、xfeat")
        if pose_state is not None and not isinstance(pose_state, dict):
            raise ValueError("pose_state 必须是 JSON 对象")
        if height_alignment is not None and not isinstance(height_alignment, dict):
            raise ValueError("height_alignment 必须是 JSON 对象")
        if timestamp is not None:
            try:
                if datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S").strftime(
                        "%Y-%m-%dT%H:%M:%S") != timestamp:
                    raise ValueError
            except ValueError as exc:
                raise ValueError("timestamp 格式应为 YYYY-MM-DDTHH:MM:SS") from exc
        for camera_id, image in images.items():
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                    or image.ndim != 3 or image.shape[2] != 3
                    or image.shape[0] < 2 or image.shape[1] < 2):
                raise ValueError(f"{camera_id} 图像须为 uint8 BGR 三通道数组")
        reference = images[reference_camera]
        width, height = reference.shape[1], reference.shape[0]
        selected = np.asarray([
            _finite_point(point_a, "point_a", width, height),
            _finite_point(point_b, "point_b", width, height),
        ], dtype=float)
        with self._lock:
            config = copy.deepcopy(self._config)
        result = measure_video_views(
            images, reference_camera, selected, config, matcher=matcher,
            pose_state=pose_state, height_alignment=height_alignment)
        mode = result.get("measurement_mode")
        distance = result.get("distance_m")
        height_delta = result.get("height_difference_m")
        response = {
            "schema_version": "1.0",
            "request_id": request_id,
            "timestamp": timestamp,
            "status": result.get("status", "unknown"),
            "measurement_mode": mode,
            "reference_camera": reference_camera,
            "points_px": {"A": selected[0].tolist(), "B": selected[1].tolist()},
            "distance_m": distance,
            "distance_available": distance is not None and result.get("status") != "unknown",
            "distance_determined": bool(result.get("actual_distance_determined", False)),
            "height_difference_m": height_delta,
            "height_difference_definition": "Z_B - Z_A" if height_delta is not None else None,
            "height_difference_interval_m": result.get("height_difference_interval_m"),
            "height_difference_available": height_delta is not None,
            "height_difference_determined": bool(
                result.get("height_difference_determined", False)),
            "metric_group": result.get("metric_group"),
            "point_support": result.get("point_support"),
            "views": result.get("views", result.get("multiview_attempt", {}).get("views", {})),
            "reason": result.get("reason"),
            "limitations": result.get("limitations", []),
            "config_revision": _config_revision(config),
        }
        return response


def create_app(service: DistanceService | None = None):
    """Create the optional HTTP API; FastAPI is imported only when requested."""
    try:
        from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.concurrency import run_in_threadpool
    except ImportError as exc:
        raise RuntimeError("HTTP 接口依赖未安装，请安装 requirements-api.txt") from exc

    engine = service or DistanceService()
    app = FastAPI(title="Pixel Distance API", version="1.0.0")
    app.state.distance_service = engine

    @app.get("/health")
    async def health():
        return {"status": "ok", "schema_version": "1.0"}

    @app.get("/v1/cameras")
    async def cameras():
        config = engine.get_config()
        return {"schema_version": "1.0", "config_revision": _config_revision(config),
                "cameras": engine.get_camera_info(),
                "video_clock_offset_s": config.get("video_clock_offset_s", {})}

    @app.put("/v1/config")
    async def replace_config(config: dict = Body(...)):
        try:
            revision = engine.update_config(config)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"status": "updated", "config_revision": revision,
                "note": "仅替换当前进程内配置；重启后需由部署配置重新加载。"}

    @app.post("/v1/measure/images")
    async def measure_images(
        reference_camera: str = Form(...),
        point_a_x: float = Form(...), point_a_y: float = Form(...),
        point_b_x: float = Form(...), point_b_y: float = Form(...),
        timestamp: str | None = Form(None), matcher: str = Form("auto"),
        pose_state_json: str | None = Form(None),
        height_alignment_json: str | None = Form(None),
        D01: UploadFile = File(...), D02: UploadFile = File(...),
        D03: UploadFile = File(...), D04: UploadFile = File(...),
    ):
        uploads = {"D01": D01, "D02": D02, "D03": D03, "D04": D04}
        images = {}
        try:
            for camera_id, upload in uploads.items():
                payload = await upload.read(20 * 1024 * 1024 + 1)
                if not payload or len(payload) > 20 * 1024 * 1024:
                    raise ValueError(f"{camera_id} 文件为空或超过 20 MiB")
                image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"{camera_id} 不是有效图像文件")
                images[camera_id] = image
            pose_state = json.loads(pose_state_json) if pose_state_json else None
            alignment = json.loads(height_alignment_json) if height_alignment_json else None
            return await run_in_threadpool(
                engine.measure_images, images, reference_camera,
                [point_a_x, point_a_y], [point_b_x, point_b_y],
                timestamp=timestamp, matcher=matcher,
                pose_state=pose_state, height_alignment=alignment)
        except (ValueError, json.JSONDecodeError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            for upload in uploads.values():
                await upload.close()

    return app
