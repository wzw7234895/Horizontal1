"""Decode one video frame in a child process so codec diagnostics stay isolated."""

from __future__ import annotations

import sys

import cv2
import numpy as np


def main() -> int:
    path, index_text = sys.argv[1:]
    index = int(index_text)
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened() or not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise ValueError("video seek failed")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError("video frame decode failed")
        np.save(sys.stdout.buffer, frame, allow_pickle=False)
        return 0
    except (ValueError, cv2.error) as exc:
        print(exc, file=sys.stderr)
        return 2
    finally:
        cap.release()


if __name__ == "__main__":
    raise SystemExit(main())
