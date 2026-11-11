"""Public in-process API for person detection and per-camera tracking."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

from .common import DEFAULT_CONFIG, PACKAGE_DIR
from .tracking import run_tracking


def run_person_tracking(
    *,
    output_dir: str | Path,
    config: str | Path = DEFAULT_CONFIG,
    camera_ids: Sequence[str] | None = None,
    weights: str | Path | None = None,
    device: str | None = None,
    image_size: int | None = None,
    analysis_fps: float | None = None,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    max_frames: int | None = None,
    save_video: bool = True,
    preview_width: int = 1280,
    decode_mode: str = "sequential",
    on_frame: Callable[[dict[str, Any]], None] | None = None,
    frame_enricher: Callable[[Any, dict[str, Any]], dict[str, Any] | None] | None = None,
    roi_overrides: dict[str, Sequence[Sequence[float]]] | None = None,
    stop_event: Event | None = None,
) -> dict[str, Any]:
    """Run configured file/RTSP sources and return a versioned run summary.

    ``on_frame`` is called synchronously once for each analyzed frame. The
    callback receives the same record written to that camera's frames.jsonl.
    Keep callbacks short; queue downstream work elsewhere.
    """
    if on_frame is not None and not callable(on_frame):
        raise TypeError("on_frame must be callable or None")
    if frame_enricher is not None and not callable(frame_enricher):
        raise TypeError("frame_enricher must be callable or None")
    if isinstance(camera_ids, str):
        camera_ids = [camera_ids]
    cameras = None if camera_ids is None else list(camera_ids)
    if cameras is not None and (not cameras or any(not isinstance(item, str) or not item for item in cameras)):
        raise ValueError("camera_ids must contain one or more nonempty camera IDs")

    os.environ.setdefault("YOLO_CONFIG_DIR", str(PACKAGE_DIR / ".cache" / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(PACKAGE_DIR / ".cache" / "matplotlib"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    args = SimpleNamespace(
        config=Path(config), cameras=cameras,
        weights=None if weights is None else Path(weights),
        out=Path(output_dir), start=start_seconds, start_clock=None,
        duration=duration_seconds, max_frames=max_frames, fps=analysis_fps,
        device=device, imgsz=image_size, no_video=not save_video,
        preview_width=preview_width, decode_mode=decode_mode,
        on_frame=on_frame, frame_enricher=frame_enricher,
        roi_overrides=roi_overrides, stop_event=stop_event, quiet=True,
    )
    return run_tracking(args)
