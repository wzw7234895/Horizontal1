from __future__ import annotations

import json
import math
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from .common import load_config, recording_time, start_seconds, write_json
from .detector import load_detector
from .stability import (BoxSmoother, DisplayConfidence, TrackConfirmation,
                        NESTED_BOX_CONTAINMENT_THRESHOLD, NESTED_BOX_MAX_AREA_RATIO,
                        keep_largest_nested_boxes)
from .video import open_reader


def make_tracker(cfg, fps):
    import yaml
    from ultralytics.trackers.bot_sort import BOTSORT
    options = yaml.safe_load(Path(cfg["tracker"]).read_text(encoding="utf-8"))
    # In pinned Ultralytics 8.4.202 this is a count of analysis frames, and IDs
    # are generated independently inside each tracker (not process-global).
    options["track_buffer"] = max(1, round(fps * cfg["lost_buffer_seconds"]))
    return BOTSORT(SimpleNamespace(**options))


def color_for(track_id):
    return tuple(int(x) for x in np.random.default_rng(track_id).integers(70, 255, size=3))


def draw_dashed_rectangle(frame, top_left, bottom_right, color, thickness=2, dash_length=10):
    """Draw a visibly different rectangle for a short-lived prediction."""
    x1, y1 = top_left
    x2, y2 = bottom_right

    def dashed_line(start, end):
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length <= 0:
            return
        direction = vector / length
        offset = 0.0
        while offset < length:
            a = start + direction * offset
            b = start + direction * min(offset + dash_length, length)
            cv2.line(frame, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                     color, thickness, cv2.LINE_AA)
            offset += dash_length * 2

    dashed_line((x1, y1), (x2, y1))
    dashed_line((x2, y1), (x2, y2))
    dashed_line((x2, y2), (x1, y2))
    dashed_line((x1, y2), (x1, y1))


class CameraState:
    def __init__(self, camera, cfg, args, run_id):
        self.camera, self.cfg, self.args = camera, cfg, args
        self.is_live = "://" in camera["source"]
        if self.is_live and args.start_clock:
            raise ValueError("Live streams do not support --start-clock seeking")
        start = start_seconds(camera, args.start, args.start_clock) if not self.is_live else args.start
        self.reader = open_reader(camera["source"], analysis_fps=args.fps or cfg["analysis_fps"],
                                  start=start, duration=args.duration, decode_mode=args.decode_mode)
        stability = cfg.get("stability", {})
        self.display_min_confidence = float(stability.get("display_min_confidence", 0.2))
        self.display_confidence = DisplayConfidence(
            self.display_min_confidence,
            float(stability.get("display_keep_confidence", self.display_min_confidence)),
        )
        prediction_seconds = float(stability.get("prediction_box_seconds", 0.0))
        if not math.isfinite(prediction_seconds) or prediction_seconds < 0:
            raise ValueError("prediction_box_seconds must be finite and nonnegative")
        self.prediction_box_seconds = prediction_seconds
        self.confirmation = TrackConfirmation(
            hits=int(stability.get("confirm_hits", 3)),
            max_gap_seconds=1.6 / self.reader.analysis_fps,
        )
        self.box_smoother = BoxSmoother(alpha=float(stability.get("box_smoothing_alpha", 0.4)))
        self.output = args.out / camera["id"]
        self.output.mkdir(parents=True)
        self.frames_file = (self.output / "frames.jsonl").open("w", encoding="utf-8")
        self.mot_file = (self.output / "mot.txt").open("w", encoding="utf-8")
        self.tracks_file = (self.output / "tracks.jsonl").open("w", encoding="utf-8")
        self.run_id = run_id
        self.tracker = None
        self.session_number = 0
        self.session_id = None
        self.last_generation = None
        self.last_time = None
        self.local_ids = {}
        self.live_tracks = {}
        self.next_id = 1
        self.frame_count = 0
        self.raw_detection_count = 0
        self.detection_count = 0
        self.suppressed_nested_detection_count = 0
        self.track_observation_count = 0
        self.displayed_detection_count = 0
        self.predicted_detection_count = 0
        self.displayed_prediction_count = 0
        self.total_tracks = 0
        self.video_writer = None
        self.started = time.perf_counter()
        self.last_frame_index = None
        self.first_video_time = None
        self.last_video_time = None

    def reset(self, reason):
        self.flush_tracks(reason)
        self.local_ids.clear()
        self.session_number += 1
        self.session_id = f"{self.run_id}-{self.camera['id']}-{self.session_number}"
        self.tracker = make_tracker(self.cfg, self.reader.analysis_fps)
        self.confirmation.reset()
        self.display_confidence.reset()
        self.box_smoother.reset()

    def finish_track(self, raw_id, reason):
        track = self.live_tracks.pop(raw_id)
        track.pop("trail")
        track["close_reason"] = reason
        self.tracks_file.write(json.dumps(track, ensure_ascii=False) + "\n")
        self.local_ids.pop(raw_id, None)
        self.confirmation.forget(raw_id)
        self.display_confidence.forget(raw_id)
        self.box_smoother.forget(raw_id)

    def flush_tracks(self, reason):
        for raw_id in list(self.live_tracks):
            self.finish_track(raw_id, reason)

    def expire_tracks(self, video_time):
        expired = {raw_id for raw_id, info in self.live_tracks.items()
                   if video_time - info["last_video_time_s"] > self.cfg["lost_buffer_seconds"]}
        if not expired:
            return
        # Live inference can run below the requested FPS. Expire by source/host
        # time as well as tracker-frame counts, BEFORE another detection can
        # re-activate a track that has already exceeded the configured seconds.
        self.tracker.tracked_stracks = [t for t in self.tracker.tracked_stracks if t.track_id not in expired]
        self.tracker.lost_stracks = [t for t in self.tracker.lost_stracks if t.track_id not in expired]
        for raw_id in expired:
            self.finish_track(raw_id, "lost_timeout")

    def prediction_observation(self, raw_id, info, video_time, width, height, polygon):
        """Extrapolate a confirmed track for the configured display-only interval."""
        age = video_time - float(info.get("last_observed_video_time_s", info["last_video_time_s"]))
        if age <= 0 or age > self.prediction_box_seconds:
            return None
        last_box = info.get("last_observed_box")
        if not last_box:
            return None
        previous_box = info.get("previous_observed_box")
        previous_time = info.get("previous_observed_video_time_s")
        last_time = float(info.get("last_observed_video_time_s", info["last_video_time_s"]))
        if previous_box is not None and previous_time is not None and last_time > float(previous_time):
            velocity = (np.asarray(last_box, dtype=np.float32) - np.asarray(previous_box, dtype=np.float32)) / (
                last_time - float(previous_time))
        else:
            velocity = np.zeros(4, dtype=np.float32)
        box = np.asarray(last_box, dtype=np.float32) + velocity * age
        x1, y1, x2, y2 = [float(value) for value in box]
        x1 = min(max(x1, 0.0), max(0.0, width - 1.0))
        y1 = min(max(y1, 0.0), max(0.0, height - 1.0))
        x2 = min(max(x2, x1 + 1.0), float(width))
        y2 = min(max(y2, y1 + 1.0), float(height))
        predicted_box = [x1, y1, x2, y2]
        point = [(x1 + x2) / 2, y2]
        return {"class_id": 0, "label": "person", "confidence": float(info.get("last_observed_confidence", 0.0)),
                "xyxy": predicted_box, "bbox_bottom_center": point, "track_id": info["track_id"],
                "observed": False, "predicted": True, "prediction_age_s": age,
                "prediction_method": "constant_velocity", "track_candidate_id": raw_id,
                "track_confirmed": True, "displayed": True, "display_xyxy": predicted_box,
                "inside_roi": None if polygon is None else bool(
                    cv2.pointPolygonTest(polygon.astype(np.float32), tuple(point), False) >= 0)}

    def update(self, item, result):
        frame, source_frame, video_time, generation = item
        if self.tracker is None or generation != self.last_generation:
            self.reset("new_stream_session")
        elif self.last_time is not None and video_time - self.last_time > self.cfg["lost_buffer_seconds"]:
            self.reset("input_gap")
        self.last_generation, self.last_time = generation, video_time
        self.first_video_time = video_time if self.first_video_time is None else self.first_video_time
        self.last_video_time, self.last_frame_index = video_time, source_frame
        self.expire_tracks(video_time)
        self.confirmation.prune(video_time)
        raw_boxes = result.boxes.cpu().numpy()
        raw_detection_count = len(raw_boxes)
        keep_mask = keep_largest_nested_boxes(raw_boxes.xyxy)
        boxes = raw_boxes[keep_mask]
        suppressed_nested_count = raw_detection_count - len(boxes)
        self.raw_detection_count += raw_detection_count
        self.suppressed_nested_detection_count += suppressed_nested_count
        tracks = self.tracker.update(boxes, frame)
        assignments = {int(t[7]): (int(t[4]), float(t[5])) for t in tracks}
        observations = []
        observed_raw_ids = set()
        height, width = frame.shape[:2]
        roi = self.camera.get("roi") or []
        polygon = np.asarray(roi, np.float32) * [width, height] if roi else None
        for i, (xyxy, confidence) in enumerate(zip(boxes.xyxy, boxes.conf)):
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            point = [(x1 + x2) / 2, y2]
            track_id = None
            candidate_id = None
            confirmed = False
            display_box = None
            if i in assignments:
                raw_id, score = assignments[i]
                observed_raw_ids.add(raw_id)
                candidate_id = raw_id
                confirmed, _ = self.confirmation.update(raw_id, video_time)
                display_box = self.box_smoother.update(raw_id, (x1, y1, x2, y2))
                if confirmed:
                    if raw_id not in self.local_ids:
                        self.local_ids[raw_id] = self.next_id
                        self.next_id += 1
                        self.total_tracks += 1
                    track_id = self.local_ids[raw_id]
                    if raw_id not in self.live_tracks:
                        self.live_tracks[raw_id] = {
                            "camera_id": self.camera["id"], "session_id": self.session_id, "track_id": track_id,
                            "first_video_time_s": video_time, "last_video_time_s": video_time,
                            "observed_frames": 0, "max_observation_gap_s": 0.0, "trail": deque(maxlen=60),
                            "last_observed_box": list(display_box), "previous_observed_box": None,
                            "last_observed_video_time_s": video_time, "previous_observed_video_time_s": None,
                            "last_observed_confidence": float(confidence)}
                    info = self.live_tracks[raw_id]
                    if info.get("last_observed_box") is not None and info.get("last_observed_video_time_s") != video_time:
                        info["previous_observed_box"] = info.get("last_observed_box")
                        info["previous_observed_video_time_s"] = info.get("last_observed_video_time_s")
                    info["last_observed_box"] = list(display_box)
                    info["last_observed_video_time_s"] = video_time
                    info["last_observed_confidence"] = float(confidence)
                    info["max_observation_gap_s"] = max(info["max_observation_gap_s"],
                                                       video_time - info["last_video_time_s"])
                    info["last_video_time_s"] = video_time
                    info["observed_frames"] += 1
                    display_point = ((display_box[0] + display_box[2]) / 2, display_box[3])
                    info["trail"].append(tuple(int(v) for v in display_point))
                    self.mot_file.write(f"{self.frame_count + 1},{track_id},{x1:.2f},{y1:.2f},"
                                        f"{x2-x1:.2f},{y2-y1:.2f},{float(confidence):.5f},-1,-1,-1\n")
            displayed = track_id is not None and self.display_confidence.update(
                candidate_id, float(confidence))
            observations.append({"class_id": 0, "label": "person", "confidence": float(confidence),
                                 "xyxy": [x1, y1, x2, y2], "bbox_bottom_center": point,
                                 "track_id": track_id, "observed": True,
                                 "predicted": False, "prediction_age_s": 0.0, "prediction_method": None,
                                 "track_candidate_id": candidate_id, "track_confirmed": confirmed,
                                 "displayed": displayed,
                                 "display_xyxy": list(display_box) if displayed else None,
                                 "inside_roi": None if polygon is None else bool(
                                     cv2.pointPolygonTest(polygon.astype(np.float32), tuple(point), False) >= 0)})
        if self.prediction_box_seconds > 0:
            for raw_id, info in list(self.live_tracks.items()):
                if raw_id in observed_raw_ids:
                    continue
                predicted = self.prediction_observation(raw_id, info, video_time, width, height, polygon)
                if predicted is not None:
                    observations.append(predicted)
                    self.predicted_detection_count += 1
                    self.displayed_prediction_count += 1
        self.frame_count += 1
        observed_observations = [o for o in observations if o["observed"]]
        self.detection_count += len(observed_observations)
        self.track_observation_count += sum(o["track_id"] is not None for o in observed_observations)
        self.displayed_detection_count += sum(o["displayed"] for o in observed_observations)
        row = {"schema_version": "1.0", "camera_id": self.camera["id"], "session_id": self.session_id,
               "analysis_frame": self.frame_count, "source_frame": source_frame,
               "video_time_s": video_time, "source_fps": self.reader.fps,
               "time_basis": "host_arrival" if self.is_live else self.reader.time_basis,
               "source_frame_basis": "decoder_sequence" if self.is_live else self.reader.source_frame_basis,
               "recording_time_estimate": None if self.is_live else recording_time(self.camera, video_time).isoformat(),
               "width": width, "height": height, "detections": observations}
        row["raw_detection_count"] = raw_detection_count
        row["detections_suppressed_nested"] = suppressed_nested_count
        frame_enricher = getattr(self.args, "frame_enricher", None)
        if frame_enricher is not None:
            enriched = frame_enricher(frame, row)
            if enriched is not None:
                if not isinstance(enriched, dict):
                    raise TypeError("frame_enricher must return a record dict or None")
                row = enriched
        self.frames_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.frame_count % 100 == 0:
            self.frames_file.flush()
        if not self.args.no_video:
            self.render(frame, observations, polygon, video_time)
        on_frame = getattr(self.args, "on_frame", None)
        if on_frame is not None:
            on_frame(row)

    def render(self, frame, observations, polygon, video_time):
        if polygon is not None:
            cv2.polylines(frame, [polygon.astype(np.int32)], True, (0, 200, 255), 2)
        for info in self.live_tracks.values():
            if len(info["trail"]) > 1:
                cv2.polylines(frame, [np.asarray(info["trail"], np.int32)], False,
                              color_for(info["track_id"]), 2)
        for obs in observations:
            if not obs["displayed"]:
                continue
            x1, y1, x2, y2 = [int(v) for v in obs["display_xyxy"]]
            track_id = obs["track_id"]
            predicted = bool(obs.get("predicted"))
            color = (0, 165, 255) if predicted else color_for(track_id)
            text = f"ID {track_id} PRED {obs['prediction_age_s']:.1f}s" if predicted else f"ID {track_id}"
            if predicted:
                draw_dashed_rectangle(frame, (x1, y1), (x2, y2), color, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{text} {obs['confidence']:.2f}", (x1, max(25, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.circle(frame, tuple(int(v) for v in obs["bbox_bottom_center"]), 4, color, -1)
        cv2.putText(frame, f"{self.camera['id']} t={video_time:.2f}s session={self.session_number}",
                    (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        scale = min(1.0, self.args.preview_width / frame.shape[1])
        size = (int(frame.shape[1] * scale) // 2 * 2, int(frame.shape[0] * scale) // 2 * 2)
        if self.video_writer is None:
            self.video_writer = cv2.VideoWriter(str(self.output / "annotated.mp4"),
                                               cv2.VideoWriter_fourcc(*"mp4v"), self.reader.analysis_fps, size)
            if not self.video_writer.isOpened():
                raise OSError("Could not create annotated video")
        self.video_writer.write(cv2.resize(frame, size) if size != (frame.shape[1], frame.shape[0]) else frame)

    def close(self, status):
        self.flush_tracks(status)
        self.reader.close()
        for stream in (self.frames_file, self.mot_file, self.tracks_file):
            stream.close()
        if self.video_writer:
            self.video_writer.release()
        summary = {"camera_id": self.camera["id"], "name": self.camera["name"],
                   "mount": self.camera["mount"], "status": status, "analysed_frames": self.frame_count,
                   "source_fps": self.reader.fps, "selected_analysis_fps": self.reader.analysis_fps,
                   "raw_detections": self.raw_detection_count,
                   "detections": self.detection_count,
                   "detections_suppressed_nested": self.suppressed_nested_detection_count,
                   "tracked_observations": self.track_observation_count,
                   "displayed_detections": self.displayed_detection_count,
                   "predicted_detections": self.predicted_detection_count,
                   "displayed_predictions": self.displayed_prediction_count,
                   "tracks_created": self.total_tracks, "sessions": self.session_number,
                   "first_video_time_s": self.first_video_time, "last_video_time_s": self.last_video_time,
                   "time_basis": "host_arrival" if self.is_live else "file_pts_with_explicit_fps_fallback",
                   "source_frame_basis": "decoder_sequence" if self.is_live else self.reader.source_frame_basis,
                   "timestamp_fallback_frames": 0 if self.is_live else self.reader.timestamp_fallback_frames,
                   "reader_finish_reason": None if self.is_live else self.reader.finish_reason,
                   "last_source_frame": self.last_frame_index, "wall_seconds": time.perf_counter() - self.started}
        write_json(self.output / "summary.json", summary)
        return summary


def run_tracking(args):
    cfg = load_config(args.config, args.cameras)
    roi_overrides = getattr(args, "roi_overrides", None)
    if roi_overrides is not None:
        if not isinstance(roi_overrides, dict):
            raise TypeError("roi_overrides must map camera IDs to normalized polygons")
        cameras_by_id = {camera["id"]: camera for camera in cfg["cameras"]}
        unknown = set(roi_overrides) - set(cameras_by_id)
        if unknown:
            raise ValueError(f"ROI overrides name unknown or unselected cameras: {sorted(unknown)}")
        for camera_id, polygon in roi_overrides.items():
            if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
                raise ValueError(f"{camera_id}: ROI override must have at least three normalized points")
            points = []
            for point in polygon:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError(f"{camera_id}: ROI points must be normalized [x, y] pairs")
                x, y = point
                if (not isinstance(x, (int, float)) or isinstance(x, bool)
                        or not isinstance(y, (int, float)) or isinstance(y, bool)
                        or not math.isfinite(x) or not math.isfinite(y)
                        or not 0 <= x <= 1 or not 0 <= y <= 1):
                    raise ValueError(f"{camera_id}: ROI coordinates must be finite values in [0, 1]")
                points.append([float(x), float(y)])
            cameras_by_id[camera_id]["roi"] = points
    model_override = getattr(args, "weights", None)
    if model_override is not None:
        cfg["model"] = str(Path(model_override).resolve())
    if args.duration is not None and args.duration <= 0:
        raise ValueError("duration must be positive")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        raise ValueError("fps must be finite and positive")
    if args.start < 0 or args.preview_width < 32 or (args.max_frames is not None and args.max_frames <= 0):
        raise ValueError("start >= 0, preview_width >= 32 and max_frames > 0 are required")
    if args.out.exists():
        raise FileExistsError(f"Choose a new --out directory: {args.out}")
    model, predict_args = load_detector(cfg, args.device, args.imgsz)
    args.out.mkdir(parents=True)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    write_json(args.out / "run_config.json", {"run_id": run_id, "model": cfg["model"],
               "model_source": "command_line_override" if model_override is not None else "config",
               "predict_args": predict_args, "config": str(args.config), "started": datetime.now().isoformat(),
               "analysis_fps": args.fps or cfg["analysis_fps"], "lost_buffer_seconds": cfg["lost_buffer_seconds"],
               "stability": cfg.get("stability", {}),
               "nested_box_deduplication": {
                   "enabled": True,
                   "min_small_box_containment": NESTED_BOX_CONTAINMENT_THRESHOLD,
                   "max_small_to_large_area_ratio": NESTED_BOX_MAX_AREA_RATIO},
               "decode_mode": args.decode_mode,
               "roi_source": "api_override" if roi_overrides is not None else "camera_config",
               "tracker": cfg["tracker"], "id_scope": "run/camera/session", "not_training": True,
               "cameras": cfg["cameras"], "tracker_yaml_snapshot": Path(cfg["tracker"]).read_text(encoding="utf-8")})
    states, summaries = [], []
    status = "completed"
    try:
        for camera in cfg["cameras"]:
            states.append(CameraState(camera, cfg, args, run_id))
        active = list(states)
        rounds = 0
        while active:
            stop_event = getattr(args, "stop_event", None)
            if stop_event is not None and stop_event.is_set():
                status = "stopped"
                break
            batch, pending = [], []
            for state in list(active):
                item = state.reader.next()
                if item is not None:
                    batch.append(item[0])
                    pending.append((state, item))
                elif state.reader.finished:
                    active.remove(state)
            if not batch:
                time.sleep(0.01)
                continue
            results = model.predict(batch, **predict_args)
            for (state, item), result in zip(pending, results):
                state.update(item, result)
                if args.max_frames and state.frame_count >= args.max_frames:
                    active.remove(state)
            rounds += 1
            if rounds % 50 == 0 and not getattr(args, "quiet", False):
                print("Progress: " + ", ".join(f"{s.camera['id']}={s.frame_count}" for s in states), flush=True)
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception:
        status = "failed"
        raise
    finally:
        for state in states:
            summaries.append(state.close(status))
        result = {"schema_version": "1.0", "status": status, "run_id": run_id,
                  "output": str(args.out.resolve()), "cameras": summaries,
                  "note": "Execution counts only; accuracy and ID stability have not been measured against human ground truth."}
        write_json(args.out / "summary.json", result)
    if not getattr(args, "quiet", False):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
