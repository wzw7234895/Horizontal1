from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "configs" / "cameras.json"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_config(path=DEFAULT_CONFIG, camera_ids=None) -> dict:
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    root = (path.parent / cfg.get("project_root", "../..")).resolve()
    cfg["root"] = root
    for camera in cfg["cameras"]:
        src = camera["source"]
        if "://" not in src:
            camera["source"] = str((root / src).resolve())
    if camera_ids:
        unknown = set(camera_ids) - {c["id"] for c in cfg["cameras"]}
        if unknown:
            raise ValueError(f"Unknown cameras: {sorted(unknown)}")
        cfg["cameras"] = [c for c in cfg["cameras"] if c["id"] in camera_ids]
    ids = [c["id"] for c in cfg["cameras"]]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Camera IDs must be nonempty and unique")
    for key in ("analysis_fps", "lost_buffer_seconds"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for camera in cfg["cameras"]:
        roi = camera.get("roi") or []
        if roi and (len(roi) < 3 or any(len(p) != 2 or any(
                not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1
                for v in p) for p in roi)):
            raise ValueError(f"{camera['id']}: roi must have >=3 normalized [x,y] points")
    cfg["model"] = str((root / cfg["model"]).resolve())
    cfg["tracker"] = str((root / cfg["tracker"]).resolve())
    return cfg


def recording_time(camera: dict, video_time_s: float) -> datetime:
    return datetime.fromisoformat(camera["recording_start"]) + timedelta(
        seconds=video_time_s + camera.get("time_offset_s", 0)
    )


def start_seconds(camera: dict, seconds: float, clock: str | None) -> float:
    if not clock:
        return seconds
    origin = datetime.fromisoformat(camera["recording_start"])
    target = datetime.combine(origin.date(), datetime.strptime(clock, "%H:%M:%S").time())
    offset = (target - origin).total_seconds() - camera.get("time_offset_s", 0)
    if offset < 0:
        raise ValueError(f"{camera['id']}: requested clock precedes this recording")
    return offset
