"""Draw source pixels and unverified RoMa candidates on four video frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {"A": "#00e5ff", "B": "#ff4d85"}


def _font(size: int) -> ImageFont.FreeTypeFont:
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        return ImageFont.truetype(str(font), size)
    return ImageFont.truetype("DejaVuSans.ttf", size)


def _draw_point(draw: ImageDraw.ImageDraw, point: list[float], label: str,
                detail: str, width: int, height: int) -> None:
    x, y = point
    color = COLORS[label]
    draw.ellipse((x - 20, y - 20, x + 20, y + 20),
                 outline="black", width=10)
    draw.ellipse((x - 20, y - 20, x + 20, y + 20),
                 outline=color, width=6)
    draw.line((x - 30, y, x + 30, y), fill="black", width=8)
    draw.line((x, y - 30, x, y + 30), fill="black", width=8)
    draw.line((x - 30, y, x + 30, y), fill=color, width=4)
    draw.line((x, y - 30, x, y + 30), fill=color, width=4)
    font = _font(30)
    text = f"{label} {detail} ({x:.0f}, {y:.0f})"
    text_width = draw.textbbox((0, 0), text, font=font)[2]
    left = min(max(12, x + 36), width - text_width - 24)
    top = y - 56 if y > height - 100 else y + 30
    top = max(12, min(top, height - 50))
    draw.rectangle((left - 8, top - 4, left + text_width + 8, top + 42),
                   fill=(0, 0, 0, 210))
    draw.text((left, top), text, font=font, fill=color)


def annotate(frames_dir: Path, probes_dir: Path, out_dir: Path) -> list[Path]:
    probes = {}
    for camera_id in ("D01", "D02", "D03"):
        path = probes_dir / f"roma_probe_D04_{camera_id}.json"
        probes[camera_id] = json.loads(path.read_text(encoding="utf-8"))
        if probes[camera_id]["reference_camera"] != "D04" or probes[camera_id]["target_camera"] != camera_id:
            raise ValueError(f"{path} 相机编号与画面不一致")
    reference = [item["reference_pixel"] for item in probes["D03"]["points"]]
    for probe in probes.values():
        if [item["reference_pixel"] for item in probe["points"]] != reference:
            raise ValueError("三路诊断使用的参考选点不一致")
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for camera_id in ("D01", "D02", "D03", "D04"):
        source = frames_dir / f"{camera_id}_20261020_155732.png"
        with Image.open(source) as opened:
            image = opened.convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")
        width, height = image.size
        if camera_id == "D04":
            title = "D04 参考画面：实际选点"
            points = reference
            details = ["选点", "选点"]
        else:
            title = f"{camera_id} 辅助画面：RoMa 候选，均未验证"
            points = [item["target_candidate_pixel"] for item in probes[camera_id]["points"]]
            details = ["候选", "候选"]
        title_font = _font(34)
        title_width = draw.textbbox((0, 0), title, font=title_font)[2]
        draw.rectangle((width - title_width - 48, 20, width - 20, 82),
                       fill=(0, 0, 0, 210))
        draw.text((width - title_width - 34, 29), title, font=title_font, fill="white")
        for label, point, detail in zip(("A", "B"), points, details):
            _draw_point(draw, point, label, detail, width, height)
        destination = out_dir / f"{camera_id}_155732_标注.png"
        image.convert("RGB").save(destination)
        outputs.append(destination)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="标出四路 RoMa 诊断候选点")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--probes-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in annotate(args.frames_dir, args.probes_dir, args.out_dir):
        print(path.resolve())


if __name__ == "__main__":
    main()
