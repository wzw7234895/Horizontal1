"""Run the optional HTTP service with ``python -m pixel_distance.api_server``."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from .service import DistanceService, create_app


def main() -> None:
    config_path = Path(os.environ.get(
        "PIXEL_DISTANCE_CONFIG", str(Path(__file__).with_name("camera_presets.json"))))
    app = create_app(DistanceService(config_path))
    uvicorn.run(app, host=os.environ.get("PIXEL_DISTANCE_HOST", "0.0.0.0"),
                port=int(os.environ.get("PIXEL_DISTANCE_PORT", "8000")),
                log_level=os.environ.get("PIXEL_DISTANCE_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
