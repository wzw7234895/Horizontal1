import pytest

from person_tracking.experiments.complete_person_v3 import PauseRequested, ensure_not_paused, read_json, relative_improvement


def test_launch_json_with_powershell_utf8_bom_can_be_read(tmp_path):
    launch = tmp_path / "launch.json"
    launch.write_text('{"pid": 123}', encoding="utf-8-sig")
    assert read_json(launch) == {"pid": 123}


def test_pause_prevents_followup_after_initial_training_already_completed(tmp_path):
    initial, continuation = tmp_path / "initial", tmp_path / "continuation"
    initial.mkdir()
    continuation.mkdir()
    (initial / "training_config.json").write_text('{"training_status":"completed"}', encoding="utf8")
    marker = initial / "PAUSE_REQUESTED"
    ensure_not_paused([initial, continuation], marker)
    (continuation / "training_config.json").write_text('{"training_status":"paused_by_user"}', encoding="utf8")
    with pytest.raises(PauseRequested):
        ensure_not_paused([initial, continuation], marker)
    marker.touch()
    with pytest.raises(PauseRequested):
        ensure_not_paused([initial], marker)


def metrics(precision, recall, f1, map50):
    return {"operating_point": {"all": {"precision": precision, "recall": recall, "f1": f1}},
            "map_metrics": {"metrics/mAP50(B)": map50, "metrics/mAP50-95(B)": map50 / 2}}


def test_higher_map_cannot_hide_a_recall_regression_in_continuation_decision():
    baseline = metrics(0.6, 0.7, 0.64, 0.7)
    candidate = metrics(0.95, 0.6, 0.74, 0.8)
    checks = relative_improvement(baseline, candidate)
    assert not all(checks.values())
    assert checks["recall_not_lower"] is False


def test_equal_result_requires_continuation_and_joint_improvement_does_not():
    baseline = metrics(0.6, 0.7, 0.64, 0.7)
    assert not all(relative_improvement(baseline, baseline).values())
    assert all(relative_improvement(baseline, metrics(0.7, 0.8, 0.75, 0.8)).values())


def test_lower_localization_map_requires_continuation_even_if_fixed_pr_improved():
    baseline = metrics(0.3, 0.4, 0.34, 0.4)
    candidate = metrics(0.7, 0.5, 0.58, 0.6)
    candidate["map_metrics"]["metrics/mAP50-95(B)"] = 0.1
    checks = relative_improvement(baseline, candidate)
    assert checks["map50_not_lower"]
    assert not checks["map50_95_not_lower"]
    assert not all(checks.values())
