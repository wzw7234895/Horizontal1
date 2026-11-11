from __future__ import annotations

import math


NESTED_BOX_CONTAINMENT_THRESHOLD = 0.90
NESTED_BOX_MAX_AREA_RATIO = 0.65


def keep_largest_nested_boxes(xyxy,
                              containment_threshold=NESTED_BOX_CONTAINMENT_THRESHOLD,
                              max_area_ratio=NESTED_BOX_MAX_AREA_RATIO):
    """Keep the largest box when a much smaller box is almost fully inside it."""
    if not 0 < containment_threshold <= 1:
        raise ValueError("containment_threshold must be in (0, 1]")
    if not 0 < max_area_ratio < 1:
        raise ValueError("max_area_ratio must be in (0, 1)")

    boxes = [tuple(float(value) for value in box) for box in xyxy]
    if any(len(box) != 4 or not all(math.isfinite(value) for value in box) for box in boxes):
        raise ValueError("xyxy must contain four finite coordinates per box")

    areas = [max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) for box in boxes]
    keep = [False] * len(boxes)
    retained = []

    for index in sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True):
        small_area = areas[index]
        suppressed = False
        if small_area > 0:
            small = boxes[index]
            for larger_index in retained:
                large_area = areas[larger_index]
                if small_area / large_area > max_area_ratio:
                    continue
                large = boxes[larger_index]
                intersection_width = max(0.0, min(small[2], large[2]) - max(small[0], large[0]))
                intersection_height = max(0.0, min(small[3], large[3]) - max(small[1], large[1]))
                if intersection_width * intersection_height / small_area >= containment_threshold:
                    suppressed = True
                    break
        if not suppressed:
            keep[index] = True
            if small_area > 0:
                retained.append(index)
    return keep


class TrackConfirmation:
    """Require consecutive observations before showing a newly created track."""

    def __init__(self, hits=3, max_gap_seconds=0.12):
        if hits < 1 or not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
            raise ValueError("hits and max_gap_seconds must be positive")
        self.hits = hits
        self.max_gap_seconds = max_gap_seconds
        self._states = {}

    def update(self, track_id, timestamp):
        previous = self._states.get(track_id)
        if previous is not None and previous[2]:
            self._states[track_id] = (timestamp, previous[1], True)
            return True, previous[1]
        if previous is None or timestamp <= previous[0] or timestamp - previous[0] > self.max_gap_seconds:
            count = 1
        else:
            count = previous[1] + 1
        confirmed = count >= self.hits
        self._states[track_id] = (timestamp, count, confirmed)
        return confirmed, count

    def forget(self, track_id):
        self._states.pop(track_id, None)

    def prune(self, timestamp):
        stale = [track_id for track_id, state in self._states.items()
                 if not state[2] and timestamp - state[0] > self.max_gap_seconds]
        for track_id in stale:
            self._states.pop(track_id, None)

    def reset(self):
        self._states.clear()


class DisplayConfidence:
    """Use separate start/keep thresholds for observed, confirmed tracks."""

    def __init__(self, start_threshold, keep_threshold):
        if not (math.isfinite(start_threshold) and math.isfinite(keep_threshold)
                and 0 <= keep_threshold <= start_threshold <= 1):
            raise ValueError("Display thresholds must satisfy 0 <= keep <= start <= 1")
        self.start_threshold = start_threshold
        self.keep_threshold = keep_threshold
        self._visible = set()

    def update(self, track_id, confidence):
        threshold = self.keep_threshold if track_id in self._visible else self.start_threshold
        if math.isfinite(confidence) and confidence >= threshold:
            self._visible.add(track_id)
            return True
        self._visible.discard(track_id)
        return False

    def forget(self, track_id):
        self._visible.discard(track_id)

    def reset(self):
        self._visible.clear()


class BoxSmoother:
    """Smooth displayed boxes only; never change detector or MOT coordinates."""

    def __init__(self, alpha=0.4, reset_distance_in_heights=1.5):
        if not 0 < alpha <= 1 or reset_distance_in_heights <= 0:
            raise ValueError("alpha and reset distance must be positive")
        self.alpha = alpha
        self.reset_distance_in_heights = reset_distance_in_heights
        self._boxes = {}

    def update(self, track_id, box):
        current = tuple(float(value) for value in box)
        if len(current) != 4 or not all(math.isfinite(value) for value in current):
            raise ValueError("box must contain four finite coordinates")
        previous = self._boxes.get(track_id)
        if previous is None:
            self._boxes[track_id] = current
            return current

        old_cx = (previous[0] + previous[2]) / 2
        old_cy = (previous[1] + previous[3]) / 2
        new_cx = (current[0] + current[2]) / 2
        new_cy = (current[1] + current[3]) / 2
        old_height = max(1.0, previous[3] - previous[1])
        if math.hypot(new_cx - old_cx, new_cy - old_cy) > old_height * self.reset_distance_in_heights:
            smoothed = current
        else:
            smoothed = tuple(
                (1 - self.alpha) * old + self.alpha * new
                for old, new in zip(previous, current)
            )
        self._boxes[track_id] = smoothed
        return smoothed

    def forget(self, track_id):
        self._boxes.pop(track_id, None)

    def reset(self):
        self._boxes.clear()
