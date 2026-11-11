"""B3/B4 PPE and torso-color post-processing for person tracking records."""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import cv2


DEFAULT_COLORS = ("blue", "yellow", "white", "red", "orange", "green", "black", "gray", "other")
DEFAULT_POLICY = Path(__file__).parent / "configs" / "b3b4_attributes.json"


def class_names_for_colors(helmet_colors=DEFAULT_COLORS, workwear_colors=DEFAULT_COLORS):
    names = ["head_no_helmet"]
    names.extend(f"helmet_{color}" for color in helmet_colors)
    names.extend(f"torso_{color}" for color in workwear_colors)
    return names


def _load_yolo(weights):
    from ultralytics import YOLO

    return YOLO(str(weights))


def _box_values(value):
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(v) for v in value]


def _model_predictions(model, frame, confidence, image_size, device):
    if model is None:
        return []
    if callable(model) and not hasattr(model, "predict"):
        result = model(frame)
    else:
        options = {"source": frame, "conf": confidence, "imgsz": image_size, "verbose": False}
        if device is not None:
            options["device"] = device
        result = model.predict(**options)
    if isinstance(result, dict):
        result = [result]
    if not isinstance(result, (list, tuple)):
        result = [result]
    if not result:
        return []
    first = result[0]
    if isinstance(first, dict):
        return [item for item in result if isinstance(item, dict)]

    boxes = getattr(first, "boxes", None)
    if boxes is None:
        return []
    xyxy = getattr(boxes, "xyxy", None)
    conf = getattr(boxes, "conf", None)
    cls = getattr(boxes, "cls", None)
    if xyxy is None or conf is None or cls is None:
        return []
    coords = _box_values(xyxy)
    scores = _box_values(conf)
    classes = [int(v) for v in _box_values(cls)]
    names = getattr(first, "names", getattr(model, "names", {}))
    predictions = []
    for index, (score, class_id) in enumerate(zip(scores, classes)):
        if isinstance(names, dict):
            name = names.get(class_id, str(class_id))
        elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            name = names[class_id]
        else:
            name = str(class_id)
        predictions.append({"class_name": str(name), "confidence": score,
                            "xyxy": coords[index * 4:index * 4 + 4]})
    return predictions


def _parse_class(label):
    name = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    if name == "head_no_helmet":
        return "helmet", "not_worn", None
    if name.startswith("helmet_"):
        return "helmet", "worn", name.removeprefix("helmet_")
    if name.startswith("torso_"):
        return "workwear", "observed", name.removeprefix("torso_")
    return None


def _valid_box(box, width, height):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in box]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
    y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _containment(part_box, person_box):
    x1 = max(part_box[0], person_box[0])
    y1 = max(part_box[1], person_box[1])
    x2 = min(part_box[2], person_box[2])
    y2 = min(part_box[3], person_box[3])
    part_area = max(0.0, part_box[2] - part_box[0]) * max(0.0, part_box[3] - part_box[1])
    if part_area <= 0 or x2 <= x1 or y2 <= y1:
        return 0.0
    return ((x2 - x1) * (y2 - y1)) / part_area


class B3B4Analyzer:
    """Enrich each detected person with B3/B4 attributes and gated alert events.

    The specialized detector is expected to use `head_no_helmet`, `helmet_*`,
    and `torso_*` classes. Recognition runs over the full image; ROI only gates
    final business alerts when `inside_roi` is available on the person record.
    """

    def __init__(self, *, weights=None, allowed_helmet_colors=(), allowed_workwear_colors=(),
                 confidence=0.35, minimum_part_pixels=12, confirm_frames=3,
                 confirm_window_seconds=1.0, image_size=1280, device=None,
                 lost_buffer_seconds=2.5, draw_annotations=True, model=None):
        if not math.isfinite(confidence) or not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        if type(minimum_part_pixels) is not int or minimum_part_pixels < 1:
            raise ValueError("minimum_part_pixels must be a positive integer")
        if type(confirm_frames) is not int or confirm_frames < 1:
            raise ValueError("confirm_frames must be a positive integer")
        if not math.isfinite(confirm_window_seconds) or confirm_window_seconds <= 0:
            raise ValueError("confirm_window_seconds must be positive")
        if type(image_size) is not int or image_size < 32:
            raise ValueError("image_size must be an integer >= 32")
        if not math.isfinite(lost_buffer_seconds) or lost_buffer_seconds <= 0:
            raise ValueError("lost_buffer_seconds must be positive")
        self.confidence = float(confidence)
        self.minimum_part_pixels = minimum_part_pixels
        self.confirm_frames = confirm_frames
        self.confirm_window_seconds = float(confirm_window_seconds)
        self.image_size = image_size
        self.device = device
        self.lost_buffer_seconds = float(lost_buffer_seconds)
        self.draw_annotations = bool(draw_annotations)
        self.allowed_helmet_colors = {str(c).strip().lower() for c in allowed_helmet_colors}
        self.allowed_workwear_colors = {str(c).strip().lower() for c in allowed_workwear_colors}
        self.model = model
        self.weights = None if weights in (None, "") else Path(weights).resolve()
        if self.model is None and self.weights is not None:
            if not self.weights.is_file():
                raise FileNotFoundError(self.weights)
            self.model = _load_yolo(self.weights)
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            names = list(names.values())
        if isinstance(names, (list, tuple)) and names:
            normalized = {str(name).strip().lower().replace("-", "_").replace(" ", "_")
                          for name in names}
            if ("head_no_helmet" not in normalized
                    or not any(name.startswith("helmet_") for name in normalized)
                    or not any(name.startswith("torso_") for name in normalized)):
                raise ValueError("B3/B4 weights must include head_no_helmet, helmet_<color>, and torso_<color> classes")
        self.history = {}
        self.active_alerts = set()
        self.track_last_seen = {}

    @classmethod
    def from_config(cls, path=DEFAULT_POLICY, *, weights=None, model=None):
        path = Path(path).resolve()
        config = json.loads(path.read_text(encoding="utf-8-sig"))
        checkpoint = weights if weights is not None else config.get("weights")
        if checkpoint and not Path(checkpoint).is_absolute():
            checkpoint = Path(checkpoint).resolve()
        return cls(weights=checkpoint, model=model,
                   allowed_helmet_colors=config.get("allowed_helmet_colors", []),
                   allowed_workwear_colors=config.get("allowed_workwear_colors", []),
                   confidence=float(config.get("confidence", 0.35)),
                   minimum_part_pixels=int(config.get("minimum_part_pixels", 12)),
                   confirm_frames=int(config.get("confirm_frames", 3)),
                   confirm_window_seconds=float(config.get("confirm_window_seconds", 1.0)),
                   image_size=int(config.get("image_size", 1280)),
                   device=config.get("device"),
                   lost_buffer_seconds=float(config.get("lost_buffer_seconds", 2.5)),
                   draw_annotations=bool(config.get("draw_annotations", True)))

    def _predictions(self, frame):
        predictions = []
        height, width = frame.shape[:2]
        for item in _model_predictions(self.model, frame, self.confidence,
                                       self.image_size, self.device):
            parsed = _parse_class(item.get("class_name", item.get("label", "")))
            if parsed is None:
                continue
            try:
                score = float(item.get("confidence", item.get("conf", 0.0)))
            except (TypeError, ValueError):
                continue
            box = _valid_box(item.get("xyxy", item.get("box")), width, height)
            if (box is None or not math.isfinite(score) or score < self.confidence
                    or box[2] - box[0] < self.minimum_part_pixels
                    or box[3] - box[1] < self.minimum_part_pixels):
                continue
            predictions.append({"kind": parsed[0], "state": parsed[1], "color": parsed[2],
                                "class_name": str(item.get("class_name", item.get("label"))),
                                "confidence": score, "xyxy": box})
        return predictions

    def _assign(self, predictions, persons):
        assigned = {id(person): [] for person in persons}
        for prediction in predictions:
            candidates = []
            for person in persons:
                person_box = _valid_box(person.get("xyxy"), 1e12, 1e12)
                if person_box is None:
                    continue
                containment = _containment(prediction["xyxy"], person_box)
                if containment >= 0.6:
                    candidates.append((containment, person))
            candidates.sort(key=lambda item: item[0], reverse=True)
            if not candidates:
                continue
            if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.1:
                continue
            assigned[id(candidates[0][1])].append(prediction)
        return assigned

    @staticmethod
    def _best_part(assigned, kind):
        candidates = [item for item in assigned if item["kind"] == kind]
        if not candidates:
            return None, "part_not_detected"
        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        if (len(candidates) > 1 and candidates[0]["class_name"] != candidates[1]["class_name"]
                and candidates[1]["confidence"] >= candidates[0]["confidence"] * 0.85):
            return None, "ambiguous_part_detections"
        return candidates[0], None

    def _attribute(self, detection, kind, allowed_colors):
        if detection is None:
            return {"status": "unknown", "color": None, "confidence": None,
                    "part_xyxy": None, "reason": "part_not_detected"}
        if kind == "helmet":
            if detection["state"] == "not_worn":
                return {"status": "non_compliant", "helmet_state": "not_worn", "color": None,
                        "confidence": detection["confidence"], "part_xyxy": detection["xyxy"],
                        "reason": None}
            color = detection["color"]
            compliant = color in allowed_colors if allowed_colors else None
            return {"status": ("compliant" if compliant else "non_compliant") if compliant is not None else "unknown",
                    "helmet_state": "worn", "color": color, "confidence": detection["confidence"],
                    "part_xyxy": detection["xyxy"],
                    "reason": None if compliant is not None else "helmet_color_policy_not_configured"}
        color = detection["color"]
        compliant = color in allowed_colors if allowed_colors else None
        return {"status": ("compliant" if compliant else "non_compliant") if compliant is not None else "unknown",
                "color": color, "confidence": detection["confidence"], "part_xyxy": detection["xyxy"],
                "reason": None if compliant is not None else "workwear_color_policy_not_configured"}

    def _confirm(self, key, capability, attribute, video_time):
        outcome = attribute["status"]
        if outcome not in {"compliant", "non_compliant"}:
            self.history.pop((key, capability), None)
            return False
        signature = (outcome, attribute.get("helmet_state"), attribute.get("color"))
        history_key = (key, capability)
        state = self.history.setdefault(history_key, {"values": deque(maxlen=self.confirm_frames), "time": None})
        if state["time"] is not None and video_time - state["time"] > self.confirm_window_seconds:
            state["values"].clear()
        state["values"].append(signature)
        state["time"] = video_time
        return len(state["values"]) == self.confirm_frames and len(set(state["values"])) == 1

    def _gate_alert(self, record, person, capability, attribute, confirmed):
        track_id = person.get("track_id")
        alert_key = (record.get("camera_id"), record.get("session_id"), track_id, capability)
        if person.get("inside_roi") is False:
            self.active_alerts.discard(alert_key)
            self.history.pop((alert_key[:3], capability), None)
            return False, False, "outside_warning_roi"
        if person.get("inside_roi") is not True:
            return False, False, "warning_roi_not_configured"
        if track_id is None:
            return False, False, "track_id_unavailable"
        if attribute["status"] != "non_compliant":
            if confirmed and attribute["status"] == "compliant":
                self.active_alerts.discard(alert_key)
            return alert_key in self.active_alerts, False, None
        if not confirmed:
            return alert_key in self.active_alerts, False, "awaiting_temporal_confirmation"
        is_new = alert_key not in self.active_alerts
        self.active_alerts.add(alert_key)
        return True, is_new, None

    def __call__(self, frame, record):
        persons = list(record.get("detections", []))
        observed = [person for person in persons if person.get("observed") is True
                    and person.get("predicted") is not True]
        predictions = self._predictions(frame) if observed and self.model is not None else []
        assignments = self._assign(predictions, observed)
        frame_alerts = []
        video_time = float(record.get("video_time_s", 0.0))

        for person in persons:
            if person.get("observed") is not True or person.get("predicted") is True:
                person["b3b4"] = {"evaluated": False, "scope": "full_frame",
                                   "reason": "predicted_or_unobserved_person"}
                continue

            matched = assignments.get(id(person), [])
            helmet_part, helmet_reason = self._best_part(matched, "helmet") if self.model is not None else (None, "model_not_configured")
            torso_part, torso_reason = self._best_part(matched, "workwear") if self.model is not None else (None, "model_not_configured")
            helmet = self._attribute(helmet_part, "helmet", self.allowed_helmet_colors)
            workwear = self._attribute(torso_part, "workwear", self.allowed_workwear_colors)
            if helmet_part is None:
                helmet["reason"] = helmet_reason
            if torso_part is None:
                workwear["reason"] = torso_reason

            camera_id = record.get("camera_id")
            session_id = record.get("session_id")
            track_id = person.get("track_id")
            track_key = (camera_id, session_id, track_id)
            if track_id is not None:
                self.track_last_seen[track_key] = video_time
            helmet_confirmed = self._confirm(track_key, "B3", helmet, video_time) if track_id is not None else False
            workwear_confirmed = self._confirm(track_key, "B4", workwear, video_time) if track_id is not None else False
            helmet_active, helmet_new, helmet_gate_reason = self._gate_alert(
                record, person, "B3", helmet, helmet_confirmed)
            workwear_active, workwear_new, workwear_gate_reason = self._gate_alert(
                record, person, "B4", workwear, workwear_confirmed)
            helmet["confirmed"] = helmet_confirmed
            helmet["alert_active"] = helmet_active
            helmet["alert_gate_reason"] = helmet_gate_reason
            workwear["confirmed"] = workwear_confirmed
            workwear["alert_active"] = workwear_active
            workwear["alert_gate_reason"] = workwear_gate_reason
            person["b3b4"] = {"evaluated": self.model is not None, "scope": "full_frame",
                               "region_state": ("inside_warning_roi" if person.get("inside_roi") is True
                                                else "outside_warning_roi" if person.get("inside_roi") is False
                                                else "warning_roi_not_configured"),
                               "helmet": helmet, "workwear_torso": workwear}
            if helmet_new:
                frame_alerts.append(self._alert(record, person, "B3", helmet))
            if workwear_new:
                frame_alerts.append(self._alert(record, person, "B4", workwear))
            if self.draw_annotations:
                self._draw(frame, person, helmet_part, torso_part, helmet, workwear)

        self._expire_stale(record, video_time)
        record["b3b4_scope"] = "full_frame"
        record["b3b4_alerts"] = frame_alerts
        record["b3b4_active_alerts"] = [
            {"camera_id": camera, "session_id": session, "track_id": track, "rule": capability}
            for camera, session, track, capability in sorted(self.active_alerts, key=str)
            if camera == record.get("camera_id") and session == record.get("session_id")
        ]
        return record

    @staticmethod
    def _draw(frame, person, helmet_part, torso_part, helmet, workwear):
        overlays = (
            (helmet_part, (40, 215, 255),
             f"B3 {helmet.get('helmet_state', 'unknown')} {helmet.get('color') or ''}"),
            (torso_part, (50, 220, 120),
             f"B4 torso {workwear.get('color') or 'unknown'}"),
        )
        for part, color, text in overlays:
            if part is None:
                continue
            x1, y1, x2, y2 = [int(round(value)) for value in part["xyxy"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text.strip(), (x1, max(18, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        box = person.get("xyxy")
        if box is not None:
            x1, y1, _, _ = [int(round(value)) for value in box]
            status = f"B3:{helmet['status']} B4:{workwear['status']}"
            cv2.putText(frame, status, (x1, max(36, y1 - 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    def _expire_stale(self, record, video_time):
        camera_id, session_id = record.get("camera_id"), record.get("session_id")
        replaced_sessions = {key[1] for key in self.track_last_seen
                             if key[0] == camera_id and key[1] != session_id}
        for key in list(self.track_last_seen):
            if key[0] == camera_id and key[1] in replaced_sessions:
                self.track_last_seen.pop(key, None)
                for history_key in [item for item in self.history if item[0] == key]:
                    self.history.pop(history_key, None)
                for alert_key in [item for item in self.active_alerts if item[:3] == key]:
                    self.active_alerts.discard(alert_key)
        for key, last_seen in list(self.track_last_seen.items()):
            if (key[0] == camera_id and key[1] == session_id
                    and video_time - last_seen > self.lost_buffer_seconds):
                self.track_last_seen.pop(key, None)
                for history_key in [item for item in self.history if item[0] == key]:
                    self.history.pop(history_key, None)
                for alert_key in [item for item in self.active_alerts if item[:3] == key]:
                    self.active_alerts.discard(alert_key)

    @staticmethod
    def _alert(record, person, capability, attribute):
        if capability == "B3":
            code = "B3_NO_HELMET" if attribute.get("helmet_state") == "not_worn" else "B3_HELMET_COLOR_NONCOMPLIANT"
        else:
            code = "B4_WORKWEAR_TORSO_COLOR_NONCOMPLIANT"
        return {"rule": capability, "code": code, "camera_id": record.get("camera_id"),
                "session_id": record.get("session_id"), "track_id": person.get("track_id"),
                "video_time_s": record.get("video_time_s"), "confidence": attribute.get("confidence"),
                "color": attribute.get("color"), "evidence_xyxy": attribute.get("part_xyxy")}


def create_analyzer(path=DEFAULT_POLICY, *, weights=None, model=None):
    """Load a B3/B4 analyzer from JSON policy, optionally overriding its weights."""
    return B3B4Analyzer.from_config(path, weights=weights, model=model)
