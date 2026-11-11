import numpy as np

from person_tracking.b3b4 import B3B4Analyzer, class_names_for_colors


class AttributeModel:
    def __init__(self, predictions):
        self.predictions = predictions
        self.calls = []

    def predict(self, source, **options):
        self.calls.append(options)
        return self.predictions


def record(*, inside_roi=True, observed=True):
    return {
        "camera_id": "D02",
        "session_id": "session-1",
        "video_time_s": 1.0,
        "width": 640,
        "height": 480,
        "detections": [{
            "label": "person",
            "xyxy": [100, 50, 300, 450],
            "track_id": 4,
            "observed": observed,
            "predicted": not observed,
            "inside_roi": inside_roi,
        }],
    }


def test_taxonomy_builds_attribute_detector_classes():
    assert class_names_for_colors(["blue", "yellow"], ["orange"]) == [
        "head_no_helmet", "helmet_blue", "helmet_yellow", "torso_orange",
    ]


def test_no_model_returns_unknown_and_does_not_claim_compliance():
    analyzer = B3B4Analyzer()
    row = analyzer(np.zeros((480, 640, 3), dtype=np.uint8), record())
    person = row["detections"][0]
    assert person["b3b4"]["evaluated"] is False
    assert person["b3b4"]["helmet"]["status"] == "unknown"
    assert person["b3b4"]["workwear_torso"]["status"] == "unknown"
    assert row["b3b4_alerts"] == []


def test_whole_frame_recognition_runs_without_roi_but_alerts_are_gated():
    model = AttributeModel([{
        "class_name": "helmet_yellow",
        "confidence": 0.92,
        "xyxy": [155, 60, 205, 115],
    }, {
        "class_name": "torso_red",
        "confidence": 0.90,
        "xyxy": [135, 170, 265, 330],
    }])
    analyzer = B3B4Analyzer(model=model, allowed_helmet_colors=["blue"],
                            allowed_workwear_colors=["blue"], confirm_frames=1)
    row = analyzer(np.zeros((480, 640, 3), dtype=np.uint8), record(inside_roi=None))
    person = row["detections"][0]
    assert person["b3b4"]["scope"] == "full_frame"
    assert person["b3b4"]["helmet"]["color"] == "yellow"
    assert person["b3b4"]["helmet"]["status"] == "non_compliant"
    assert person["b3b4"]["workwear_torso"]["color"] == "red"
    assert person["b3b4"]["workwear_torso"]["status"] == "non_compliant"
    assert person["b3b4"]["helmet"]["alert_gate_reason"] == "warning_roi_not_configured"
    assert row["b3b4_alerts"] == []
    assert model.calls[0]["imgsz"] == 1280


def test_explicit_no_helmet_and_torso_violation_alert_after_confirmation():
    model = AttributeModel([{
        "class_name": "head_no_helmet",
        "confidence": 0.95,
        "xyxy": [155, 60, 205, 115],
    }, {
        "class_name": "torso_red",
        "confidence": 0.91,
        "xyxy": [135, 170, 265, 330],
    }])
    analyzer = B3B4Analyzer(model=model, allowed_workwear_colors=["blue"], confirm_frames=3)
    emitted = []
    for index in range(3):
        current = record(inside_roi=True)
        current["video_time_s"] = index / 10
        row = analyzer(np.zeros((480, 640, 3), dtype=np.uint8), current)
        emitted.extend(row["b3b4_alerts"])
    assert {alert["code"] for alert in emitted} == {
        "B3_NO_HELMET", "B4_WORKWEAR_TORSO_COLOR_NONCOMPLIANT",
    }
    assert len(emitted) == 2
    assert len(analyzer(np.zeros((480, 640, 3), dtype=np.uint8), {
        **record(inside_roi=True), "video_time_s": 0.4,
    })["b3b4_alerts"]) == 0


def test_unobserved_prediction_is_not_used_as_new_evidence():
    model = AttributeModel([{
        "class_name": "head_no_helmet",
        "confidence": 0.99,
        "xyxy": [155, 60, 205, 115],
    }])
    analyzer = B3B4Analyzer(model=model, confirm_frames=1)
    row = analyzer(np.zeros((480, 640, 3), dtype=np.uint8), record(inside_roi=True, observed=False))
    assert row["detections"][0]["b3b4"]["evaluated"] is False
    assert row["b3b4_alerts"] == []


def test_policy_template_does_not_mark_any_color_compliant_by_default():
    analyzer = B3B4Analyzer(model=AttributeModel([{
        "class_name": "helmet_blue", "confidence": 0.9, "xyxy": [155, 60, 205, 115],
    }]))
    row = analyzer(np.zeros((480, 640, 3), dtype=np.uint8), record(inside_roi=True))
    helmet = row["detections"][0]["b3b4"]["helmet"]
    assert helmet["helmet_state"] == "worn"
    assert helmet["color"] == "blue"
    assert helmet["status"] == "unknown"
    assert helmet["reason"] == "helmet_color_policy_not_configured"


def test_roi_false_prevents_alert_and_clears_confirmation_history():
    model = AttributeModel([{
        "class_name": "head_no_helmet", "confidence": 0.95, "xyxy": [155, 60, 205, 115],
    }])
    analyzer = B3B4Analyzer(model=model, confirm_frames=1)
    row = analyzer(np.zeros((480, 640, 3), dtype=np.uint8), record(inside_roi=False))
    person = row["detections"][0]
    assert person["b3b4"]["helmet"]["status"] == "non_compliant"
    assert person["b3b4"]["helmet"]["alert_gate_reason"] == "outside_warning_roi"
    assert row["b3b4_alerts"] == []
    assert analyzer.active_alerts == set()
