from __future__ import annotations

import math
import threading
import time

import cv2


def inspect_source(source: str) -> dict:
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {source}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        if not ok:
            raise OSError(f"Cannot decode first frame: {source}")
        return {"fps": fps, "frame_count": frames, "width": frame.shape[1],
                "height": frame.shape[0], "duration_s": frames / fps if fps > 0 else None,
                "duration_basis": "estimated_frame_count_divided_by_nominal_fps"}
    finally:
        cap.release()


def read_frame_at(cap, target: int, decode_mode="seek"):
    """Read by index. Only sequential mode provides a counted decoded-frame index.

    OpenCV can report the requested index after seeking even when the actual
    decoded-frame count differs. It is not an independent accuracy check.
    """
    if decode_mode not in ("seek", "sequential"):
        raise ValueError("decode_mode must be seek or sequential")
    current = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    if decode_mode == "seek":
        if current != target and not cap.set(cv2.CAP_PROP_POS_FRAMES, target):
            raise OSError(f"Cannot seek to source frame {target}")
        ok, frame = cap.read()
    else:
        if target < current:
            raise ValueError("Sequential sampling requires increasing source frame indexes")
        while current <= target:
            if not cap.grab():
                raise OSError(f"Sequential decode stopped at source frame {current}, requested {target}")
            current += 1
        ok, frame = cap.retrieve()
    decoded = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
    if not ok or decoded != target:
        raise OSError(f"Sample decode failed: requested {target}, decoded {decoded}")
    return frame


class FileReader:
    """Select frames by media timestamps; sequential mode preserves decoder context."""

    def __init__(self, source, analysis_fps=10, start=0, duration=None, decode_mode="sequential"):
        if decode_mode not in ("seek", "sequential"):
            raise ValueError("decode_mode must be seek or sequential")
        self.decode_mode = decode_mode
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise OSError(f"Cannot open video: {source}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.close()
            raise ValueError("File has no valid FPS")
        self.analysis_fps = min(float(analysis_fps), self.fps)
        if (not math.isfinite(self.analysis_fps) or self.analysis_fps <= 0
                or not math.isfinite(start) or start < 0
                or duration is not None and (not math.isfinite(duration) or duration <= 0)):
            self.close()
            raise ValueError("Need finite positive analysis_fps/duration and finite nonnegative start")
        self.start_time = float(start)
        self.end_time = None if duration is None else start + duration
        self.source_frame_basis = "decoded_frame_count"
        if start and decode_mode == "seek":
            if not self.cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000):
                self.close()
                raise OSError("Video seek failed")
            self.source_frame_basis = "opencv_position_estimate_after_seek"
        self.index = int(round(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
        self.next_target = self.start_time
        self.last_time = None
        self.time_basis = "file_pts"
        self.timestamp_fallback_frames = 0
        self.selected_frames = 0
        self.finish_reason = None
        self.finished = False
        self.generation = 0
        if start and decode_mode == "sequential":
            print(f"Sequential decoding to video time {start:.3f}s before analysis; "
                  "this can take time for long recordings.", flush=True)

    def next(self):
        if self.finished:
            return None
        while True:
            if not self.cap.grab():
                self.finished = True
                # VideoCapture cannot reliably distinguish EOF from a decoder
                # failure. Keep that limitation explicit in the run summary.
                self.finish_reason = "eof_or_decode_failure"
                if not self.selected_frames and self.start_time:
                    raise ValueError(f"No decodable frame at or after video time {self.start_time:.3f}s")
                return None
            current = self.index
            self.index += 1
            video_time = float(self.cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            self.time_basis = "file_pts"
            if (not math.isfinite(video_time) or video_time < 0
                    or self.last_time is not None and video_time <= self.last_time):
                video_time = current / self.fps if self.last_time is None else self.last_time + 1 / self.fps
                self.time_basis = "nominal_fps_fallback"
                self.timestamp_fallback_frames += 1
                if self.timestamp_fallback_frames == 1:
                    print("Video timestamps unavailable or non-monotonic; using nominal FPS for affected frames.",
                          flush=True)
            self.last_time = video_time
            if self.end_time is not None and video_time >= self.end_time - 1e-7:
                self.finished = True
                self.finish_reason = "requested_duration"
                return None
            if video_time < self.next_target - 1e-7:
                if self.decode_mode == "sequential" and video_time < self.start_time and self.index % 10000 == 0:
                    print(f"Sequential warm-up: {video_time:.1f}/{self.start_time:.1f}s "
                          f"({self.index} decoded frames)", flush=True)
                continue
            ok, frame = self.cap.retrieve()
            if not ok:
                self.finished = True
                self.finish_reason = "retrieve_failure"
                raise OSError(f"Could not retrieve frame {current}")
            # A timestamp gap skips empty target intervals, without outputting
            # duplicate frames or a burst of closely spaced catch-up samples.
            slot = math.floor((video_time - self.start_time) * self.analysis_fps + 1e-6)
            self.next_target = self.start_time + (slot + 1) / self.analysis_fps
            self.selected_frames += 1
            return frame, current, video_time, self.generation

    def close(self):
        self.cap.release()


class LiveReader:
    """Keep only the newest decoded RTSP frame so slow inference cannot build a queue.

    Times are host arrival times, NOT hardware exposure timestamps.
    A reconnect changes generation and must reset the camera's tracking session.
    """

    def __init__(self, source, analysis_fps=10, start=0, duration=None):
        if start:
            raise ValueError("Live streams do not support seeking")
        self.source = source
        self.analysis_fps = float(analysis_fps)
        if self.analysis_fps <= 0:
            raise ValueError("analysis_fps must be positive")
        self.fps = None
        self.duration = duration
        self.origin = time.monotonic()
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.last_consumed = -1
        self.last_selected_time = -1e9
        self.finished = False
        self.thread = threading.Thread(target=self._decode, daemon=True)
        self.thread.start()

    def _decode(self):
        generation, sequence = -1, 0
        while not self.stop.is_set():
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
            generation += 1
            try:
                while cap.isOpened() and not self.stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    now = time.monotonic() - self.origin
                    if self.duration is not None and now >= self.duration:
                        self.finished = True
                        return
                    with self.lock:
                        self.latest = (frame, sequence, now, generation)
                    sequence += 1
            finally:
                cap.release()
            self.stop.wait(2.0)

    def next(self):
        if self.duration is not None and time.monotonic() - self.origin >= self.duration:
            self.finished = True
            return None
        with self.lock:
            item = self.latest
        if item is None or item[1] == self.last_consumed:
            return None
        if item[2] - self.last_selected_time < 1 / self.analysis_fps - 1e-6:
            return None
        self.last_consumed, self.last_selected_time = item[1], item[2]
        return item

    def close(self):
        self.stop.set()
        self.thread.join(timeout=12)


def open_reader(source, decode_mode="sequential", **kwargs):
    return LiveReader(source, **kwargs) if "://" in source else FileReader(source, decode_mode=decode_mode, **kwargs)
