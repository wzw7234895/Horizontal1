"""Generate a reproducible, explicitly synthetic ground-distance example."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    output = Path(__file__).parent / "demo_output"
    output.mkdir(exist_ok=True)
    image = np.full((600, 800, 3), 245, dtype=np.uint8)
    for x in range(0, 800, 100):
        cv2.line(image, (x, 0), (x, 599), (180, 180, 180), 1)
    for y in range(0, 600, 100):
        cv2.line(image, (0, y), (799, y), (180, 180, 180), 1)
    cv2.putText(image, "SYNTHETIC GROUND GRID: 1 m per square", (25, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (35, 35, 35), 2)
    locations = [(100, 50), (700, 50), (100, 550), (700, 550),
                 (400, 50), (400, 550), (100, 300), (700, 300),
                 (250, 250), (550, 350)]
    anchors = []
    for index, (u, v) in enumerate(locations):
        role = "check" if index >= 8 else "control"
        anchors.append({"id": f"P{index+1:02d}", "role": role,
                        "pixel": [u, v], "xy_m": [u / 100.0, v / 100.0]})
        cv2.circle(image, (u, v), 5, (0, 120, 0) if role == "control" else (0, 130, 230), -1)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode demo image")
    encoded.tofile(str(output / "synthetic_ground.png"))
    (output / "synthetic_points.json").write_text(
        json.dumps({"points": anchors}, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
