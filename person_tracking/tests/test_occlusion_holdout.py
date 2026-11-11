"""Synthetic contract tests; these numbers are never field performance claims."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import urllib.error
import urllib.request

import numpy as np
import pytest

from person_tracking.common import write_json
from person_tracking.dataset import save_jpeg
from person_tracking.occlusion import (
    OcclusionStore, group_keys, match_frame, recall_report, sha256, validate_objects,
)
from person_tracking.occlusion_benchmark import (
    evaluate, freeze, make_server, protocol_for, verify_frozen,
)
from person_tracking.occlusion_history import (
    allowed_at, assert_isolation, blocked_intervals, evidence_unchanged,
    history_inventory, subtract_intervals, timestamp,
)


def person(box=(.5, .5, .12, .5), **changes):
    value = {"box": list(box), "occlusion": "light", "occluders": ["load"],
             "distance": "near", "truncated": False}
    value.update(changes)
    return value


def prediction(box, confidence=.9):
    return {"box": list(box), "confidence": confidence}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf8")


@pytest.fixture
def holdout(tmp_path):
    history = tmp_path / "historical_data"
    runs = tmp_path / "historical_runs"
    runs.mkdir()
    write_jsonl(history / "v1" / "samples.jsonl", [{
        "id": "old", "source": str(tmp_path / "old.mp4"), "camera_id": "D01",
        "recording_time_estimate": "2026-10-20T15:00:00", "split": "train",
    }])
    root = tmp_path / "review"
    (root / "images").mkdir(parents=True)
    rows = []
    for index in range(2):
        image = root / "images" / f"sample_{index}.jpg"
        save_jpeg(image, np.full((90, 160, 3), 40 + index, dtype=np.uint8))
        rows.append({"id": f"sample_{index}", "camera_id": "D02", "clip_id": "clip_D02",
                     "source": str(tmp_path / "new.mp4"), "source_frame": index,
                     "video_time_s": float(index), "recording_time_estimate": f"2026-10-20T16:29:{15+index:02d}",
                     "width": 160, "height": 90, "image": f"images/sample_{index}.jpg",
                     "image_sha256": sha256(image), "split": "occlusion_holdout"})
    write_jsonl(root / "samples.jsonl", rows)
    write_json(root / "summary.json", {"status": "completed", "kind": "occlusion_holdout"})
    write_json(root / "protocol.json", protocol_for("temporal_gap", 30))
    write_json(root / "history_audit.json", history_inventory(history, runs))
    write_json(root / "sampling_plan.json", {"selection": "uniform clock sampling"})
    write_json(root / "reviews.json", {})
    return root, history, runs


def test_one_to_one_matching_counts_duplicate_and_missed_person():
    left = person((.2, .5, .1, .4))
    right = person((.8, .5, .1, .4))
    result = match_frame([left, right], [
        prediction(left["box"], .8), prediction(left["box"], .9),
        prediction(right["box"], .09), prediction((.5, .5, .1, .2), .6),
    ], [])
    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 1)
    assert result["predictions_above_threshold"] == 3
    assert result["objects"][0]["confidence"] == .9
    assert result["objects"][1]["matched"] is False


def test_one_prediction_does_not_detect_two_overlapping_people():
    a, b = person(), person((.51, .5, .12, .5))
    result = match_frame([a, b], [prediction(a["box"])], [])
    assert (result["tp"], result["fn"]) == (1, 1)


def test_valid_truth_has_priority_over_ignore_region():
    obj = person()
    result = match_frame([obj], [prediction(obj["box"])], [[.5, .5, 1., 1.]])
    assert (result["tp"], result["ignored_predictions"], result["fn"]) == (1, 0, 0)


def test_ignore_uses_prediction_area_and_preserves_other_false_positive():
    result = match_frame([], [prediction((.2, .5, .03, .05)), prediction((.8, .5, .03, .05))],
                         [[.2, .5, .3, .8]])
    assert (result["ignored_predictions"], result["fp"]) == (1, 1)


@pytest.mark.parametrize("changes", [
    {"occlusion": None}, {"distance": None}, {"occlusion": "none", "occluders": ["load"]},
    {"occlusion": "heavy", "occluders": []}, {"occluders": ["load", "load"]},
    {"truncated": "false"}, {"box": [.01, .5, .2, .4]},
])
def test_missing_or_inconsistent_human_attributes_rejected(changes):
    with pytest.raises(ValueError):
        validate_objects([person(**changes)])


def test_multi_occluder_and_unknown_labels_remain_explicit():
    assert validate_objects([person(occlusion="unknown", occluders=["load", "structure"],
                                    distance="unknown")])[0]["distance"] == "unknown"
    with pytest.raises(ValueError, match="Duplicate"):
        validate_objects([person(), person()])


def test_far_small_requires_human_far_tag_and_visible_size():
    near_small = person((.5, .5, .04, 40/720), distance="near")
    far_large = person((.5, .5, .1, 80/720), distance="far")
    far_small = person((.5, .5, .04, 40/720), distance="far")
    target = ("small_target", "far_and_visible_small")
    assert target not in group_keys(near_small, "D01")
    assert target not in group_keys(far_large, "D01")
    assert target in group_keys(far_small, "D01")
    assert ("size", "visible_h_32_to64") in group_keys(far_small, "D01")


def test_recall_zero_truth_is_unverified_not_zero_or_perfect():
    report = recall_report([], protocol_for("temporal_gap", 30))
    assert report["totals"]["recall"] is None
    assert report["totals"]["precision"] is None
    for row in report["strata"]:
        assert row["recall"] is None and row["coverage"] == "not_verified"
        assert row["recall_target_met"] is None


def test_overlapping_groups_do_not_duplicate_overall_denominator():
    obj = person(occlusion="heavy", occluders=["load", "structure"], distance="far")
    match = match_frame([obj], [], [])
    frames = [{"id": "a", "camera_id": "D03", "clip_id": "clip1", **match}]
    report = recall_report(frames, protocol_for("temporal_gap", 30))
    assert report["totals"]["fn"] == 1
    rows = {(r["category"], r["stratum"]): r for r in report["strata"]}
    assert rows["occluder", "load"]["fn"] == 1
    assert rows["occluder", "structure"]["fn"] == 1
    assert rows["occlusion", "heavy"]["recall"] == 0
    assert rows["occlusion", "heavy"]["coverage"] == "limited"
    assert rows["occlusion", "heavy"]["recall_target_met"] is None


def test_historical_exclusion_covers_other_camera_and_old_splits(holdout):
    _, historical, runs = holdout
    write_jsonl(historical / "abandoned_v2" / "exported_samples.jsonl", [{
        "id": "oldval", "camera_id": "D04", "source": "old_other_camera.mp4", "split": "val",
        "recording_time_estimate": "2026-10-20T16:29:30", "split_group": "not-an-iso-group",
    }])
    inv = history_inventory(historical, runs)
    with pytest.raises(ValueError, match="Historical temporal overlap"):
        assert_isolation([{"id": "newD02", "camera_id": "D02", "recording_time_estimate": "2026-10-20T16:29:40"}],
                         inv, protocol_for("temporal_gap", 30))
    assert not allowed_at(timestamp("2026-10-20T16:29:40"), inv, "temporal_gap", 30)


def test_strict_group_and_temporal_gap_are_distinct(holdout):
    _, historical, runs = holdout
    inv = history_inventory(historical, runs)
    same_group_far_from_sample = timestamp("2026-10-20T15:05:00")
    assert allowed_at(same_group_far_from_sample, inv, "temporal_gap", 30)
    assert not allowed_at(same_group_far_from_sample, inv, "strict_group", 30)
    assert not allowed_at(timestamp("2026-10-20T15:10:30"), inv, "strict_group", 30)
    assert allowed_at(timestamp("2026-10-20T15:11:01"), inv, "strict_group", 30)


def test_nested_full_video_inference_is_excluded_globally(holdout):
    _, historical, runs = holdout
    write_jsonl(runs / "inference" / "full" / "D01" / "frames.jsonl", [
        {"recording_time_estimate": "2026-10-20T15:11:40"},
        {"recording_time_estimate": "2026-10-20T16:20:29.223"},
    ])
    inv = history_inventory(historical, runs)
    assert not allowed_at(timestamp("2026-10-20T16:09:30"), inv, "temporal_gap", 30)
    start, end = timestamp("2026-10-20T16:20:00"), timestamp("2026-10-20T16:22:00")
    allowed = subtract_intervals(start, end, blocked_intervals(inv, "temporal_gap", 30))
    assert allowed == [[pytest.approx(timestamp("2026-10-20T16:20:59.223")), end]]


def test_store_distinguishes_ignore_only_from_confirmed_negative(holdout):
    root, _, _ = holdout
    store = OcclusionStore(root)
    with pytest.raises(ValueError, match="显式确认"):
        store.save(0, "reviewed", [], [[.5, .5, .1, .1]])
    saved = store.save(0, "reviewed", [], [[.5, .5, .1, .1]], confirm_empty=True)
    assert saved["confirmed_no_evaluable_person"] is True
    assert saved["confirmed_empty"] is False
    saved = store.save(1, "reviewed", [], confirm_empty=True)
    assert saved["confirmed_empty"] is True
    assert OcclusionStore(root).status()["pending"] == 0


def test_exclusions_need_reasons_but_can_discard_incomplete_boxes(holdout):
    root, _, _ = holdout
    store = OcclusionStore(root)
    with pytest.raises(ValueError, match="原因"):
        store.save(0, "excluded", [person(occlusion=None)])
    result = store.save(0, "excluded", [person(occlusion=None)], reason="反光导致无法确定人员")
    assert result["status"] == "excluded" and result["objects"] == []


def test_pending_review_cannot_create_frozen_output(holdout, tmp_path):
    root, _, _ = holdout
    store = OcclusionStore(root)
    store.save(0, "reviewed", [person()])
    out = tmp_path / "frozen"
    with pytest.raises(ValueError, match="pending"):
        freeze(root, out)
    assert not out.exists()
    with pytest.raises(ValueError, match="Freeze human"):
        verify_frozen(root)


def test_freeze_locks_snapshot_preserves_source_and_rejects_tamper(holdout, tmp_path):
    root, _, _ = holdout
    store = OcclusionStore(root)
    store.save(0, "reviewed", [person()])
    store.save(1, "reviewed", [], confirm_empty=True)
    output = tmp_path / "frozen"
    result = freeze(root, output)
    assert result["evaluation_only"] is True
    frozen = verify_frozen(output)
    assert frozen.status()["frozen"] is True
    with pytest.raises(ValueError, match="冻结"):
        frozen.save(0, "reviewed", [person()])
    store.save(0, "reviewed", [person(occlusion="moderate")])
    assert verify_frozen(output).item(0)["objects"][0]["occlusion"] == "light"
    (output / "protocol.json").write_text("{}", encoding="utf8")
    with pytest.raises(ValueError, match="Frozen evidence changed"):
        verify_frozen(output)


def test_frozen_image_change_is_detected(holdout, tmp_path):
    root, _, _ = holdout
    store = OcclusionStore(root)
    for index in range(2):
        store.save(index, "reviewed", [person()])
    output = tmp_path / "frozen"
    freeze(root, output)
    (output / "images/sample_0.jpg").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="image changed"):
        verify_frozen(output)


def test_new_training_exposure_invalidates_prepared_holdout(holdout, tmp_path):
    root, history, _ = holdout
    store = OcclusionStore(root)
    for index in range(2):
        store.save(index, "reviewed", [person()])
    write_jsonl(history / "new_train" / "samples.jsonl", [{
        "id": "new_exposure", "source": "same_day.mp4", "camera_id": "D03",
        "recording_time_estimate": "2026-10-20T16:29:15", "split": "train",
    }])
    with pytest.raises(ValueError, match="Historical temporal overlap"):
        freeze(root, tmp_path / "frozen")


def test_changed_historical_evidence_is_not_silently_accepted(holdout):
    _, history, runs = holdout
    before = history_inventory(history, runs)
    path = history / "v1/samples.jsonl"
    path.write_text(path.read_text(encoding="utf8") + "\n", encoding="utf8")
    after = history_inventory(history, runs)
    with pytest.raises(ValueError, match="Historical evidence changed"):
        evidence_unchanged(before, after)


def test_http_roundtrip_has_no_detector_proposals_and_persists_attributes(holdout):
    root, _, _ = holdout
    server = make_server(root, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/item?index=0") as response:
            item = json.load(response)
        assert item["objects"] == []
        with urllib.request.urlopen(base + "/api/image?index=0") as response:
            assert response.headers.get_content_type() == "image/jpeg"
            assert response.read().startswith(b"\xff\xd8")
        payload = {"index": 0, "status": "reviewed", "objects": [person(occlusion="heavy", occluders=["structure"])],
                   "ignore_regions": [], "confirm_empty": False}
        request = urllib.request.Request(base + "/api/review", json.dumps(payload).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["status"]["reviewed"] == 1
        with urllib.request.urlopen(base + "/api/item?index=0") as response:
            assert json.load(response)["objects"][0]["occluders"] == ["structure"]
        request = urllib.request.Request(base + "/api/review", json.dumps(payload).encode(),
                                         {"Content-Type": "application/json", "Origin": "http://unrelated.example"})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request)
        assert exc.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("development_recheck", [False, True])
def test_evaluation_pipeline_writes_real_artifacts_from_human_gt(holdout, tmp_path, monkeypatch, development_recheck):
    """Stub the detector, not the evaluator/labels; artifact counts must match."""
    root, _, _ = holdout
    store = OcclusionStore(root)
    for index in range(2):
        store.save(index, "reviewed", [person(occlusion="heavy", occluders=["structure"])])
    frozen = tmp_path / "frozen"
    freeze(root, frozen)
    weight = tmp_path / "model" / "weights" / "best.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"synthetic detector stub checkpoint")

    class TensorStub:
        def __init__(self, values):
            self.values = values

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    class DetectorStub:
        names = {0: "person"}

        def __init__(self, _):
            self.calls = 0

        def predict(self, image, **options):
            assert image.shape[:2] == (720, 1280)
            assert options["conf"] == .1 and options["classes"] == [0]
            # First human-labelled person is found; second is genuinely missed.
            boxes = [[.44, .25, .56, .75]] if self.calls == 0 else []
            scores = [.9] if self.calls == 0 else []
            self.calls += 1
            return [SimpleNamespace(boxes=SimpleNamespace(
                xyxyn=TensorStub(boxes), conf=TensorStub(scores)))]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=DetectorStub, __version__="test-stub"))
    out = tmp_path / "evaluation"
    evaluate(SimpleNamespace(dataset=frozen, out=out, weights=[weight], device="cpu",
                             development_recheck=development_recheck))
    comparison = json.loads((out / "comparison.json").read_text(encoding="utf8"))
    metrics_path = Path(comparison[0]["metrics"])
    metrics = json.loads(metrics_path.read_text(encoding="utf8"))
    assert (metrics["totals"]["tp"], metrics["totals"]["fn"], metrics["totals"]["recall"]) == (1, 1, .5)
    assert metrics["training_started"] is False
    assert metrics["development_recheck"] is development_recheck
    assert metrics["current_history_rechecked"] is not development_recheck
    if development_recheck:
        assert metrics["independence_scope"] == "development_recheck_after_prior_model_comparisons"
    assert (metrics_path.parent / "per_person.jsonl").is_file()
    assert (metrics_path.parent / "predictions.jsonl").is_file()
    assert (metrics_path.parent / "recall_by_group.csv").read_text(encoding="utf-8-sig").startswith("category,")
    assert "未验证" in (metrics_path.parent / "report.md").read_text(encoding="utf8")


def make_reviewed_frozen(holdout, output):
    root, _, _ = holdout
    store = OcclusionStore(root)
    for index in range(len(store.rows)):
        store.save(index, "reviewed", [person(occlusion="heavy", occluders=["load"])])
    freeze(root, output)
    return output


@pytest.mark.parametrize("tampered_file", ["protocol.json", "images/sample_0.jpg"])
def test_development_recheck_preserves_frozen_integrity_after_later_exposure(holdout, tmp_path, tampered_file):
    _, _, runs = holdout
    frozen = make_reviewed_frozen(holdout, tmp_path / "frozen")
    write_jsonl(runs / "later_replay" / "frames.jsonl", [
        {"recording_time_estimate": "2026-10-20T16:29:15"},
    ])
    with pytest.raises(ValueError, match="Historical temporal overlap"):
        verify_frozen(frozen)
    assert verify_frozen(frozen, development_recheck=True).frozen
    (frozen / tampered_file).write_bytes(b"{}" if tampered_file.endswith(".json") else b"tampered")
    with pytest.raises(ValueError, match="changed"):
        verify_frozen(frozen, development_recheck=True)


def register_completed_evaluation(evaluation_root, dataset, name="completed"):
    """Real comparison/metric structure, with deliberately synthetic counts."""
    output = evaluation_root / name
    metrics = output / "model_1_stub" / "metrics.json"
    freeze_path = dataset / "freeze.json"
    write_json(metrics, {
        "dataset": str(dataset.resolve()), "freeze_sha256": sha256(freeze_path),
        "model": "synthetic_test_detector.pt", "model_sha256": "test-only",
        "totals": {"tp": 1, "fn": 1, "fp": 0, "recall": .5},
        "evaluation_only": True, "training_started": False,
    })
    comparison = output / "comparison.json"
    write_json(comparison, [{"model": "synthetic_test_detector.pt", "metrics": str(metrics.resolve()),
                             "totals": {"tp": 1, "fn": 1, "fp": 0, "recall": .5}}])
    return comparison, metrics


def test_completed_holdout_exposure_blocks_new_other_camera_and_records_evidence(holdout, tmp_path):
    _, historical, runs = holdout
    frozen = make_reviewed_frozen(holdout, tmp_path / "already_evaluated")
    evaluation = tmp_path / "evaluation"
    comparison, metrics = register_completed_evaluation(evaluation, frozen)
    inv = history_inventory(historical, runs, evaluation_root=evaluation)
    row = {"id": "new_D04", "camera_id": "D04", "recording_time_estimate": "2026-10-20T16:29:30"}
    with pytest.raises(ValueError, match="Historical temporal overlap"):
        assert_isolation([row], inv, protocol_for("temporal_gap", 30))
    assert not allowed_at(timestamp("2026-10-20T16:29:44"), inv, "temporal_gap", 30)
    assert allowed_at(timestamp("2026-10-20T16:29:47"), inv, "temporal_gap", 30)
    assert not allowed_at(timestamp("2026-10-20T16:25:00"), inv, "strict_group", 30)
    for path in (comparison, metrics, frozen / "samples.jsonl", frozen / "freeze.json"):
        assert inv["evidence_sha256"][str(path.resolve())] == sha256(path)


def test_incomplete_evaluation_without_comparison_does_not_claim_exposure(holdout, tmp_path):
    _, historical, runs = holdout
    evaluation = tmp_path / "evaluation"
    write_json(evaluation / "interrupted" / "model_1" / "metrics.json", {
        "dataset": str(tmp_path / "nonexistent_unfinished_dataset"), "evaluation_only": True,
    })
    inv = history_inventory(historical, runs, evaluation_root=evaluation)
    assert allowed_at(timestamp("2026-10-20T16:29:30"), inv, "temporal_gap", 30)


def test_current_frozen_dataset_can_be_reverified_after_its_completed_comparison(holdout, tmp_path):
    _, historical, runs = holdout
    frozen = make_reviewed_frozen(holdout, tmp_path / "frozen")
    evaluation = tmp_path / "evaluation"
    register_completed_evaluation(evaluation, frozen)
    normal = history_inventory(historical, runs, evaluation_root=evaluation)
    assert not allowed_at(timestamp("2026-10-20T16:29:15"), normal, "temporal_gap", 30)
    same_dataset = history_inventory(historical, runs, evaluation_root=evaluation,
                                     ignore_evaluation_datasets=(frozen,))
    assert allowed_at(timestamp("2026-10-20T16:29:15"), same_dataset, "temporal_gap", 30)
    assert verify_frozen(frozen).frozen is True


def test_same_source_review_revision_can_freeze_after_prior_model_comparison(holdout, tmp_path):
    root, _, _ = holdout
    frozen = make_reviewed_frozen(holdout, tmp_path / "frozen_v1")
    register_completed_evaluation(tmp_path / "evaluation", frozen)
    # Correct the original human review, then freeze the next revision. Its
    # freeze.source family is the same; it must not be mistaken for a new holdout.
    OcclusionStore(root).save(0, "reviewed", [person(occlusion="moderate", occluders=["load"])])
    revised = tmp_path / "frozen_v2"
    freeze(root, revised)
    assert verify_frozen(revised).item(0)["objects"][0]["occlusion"] == "moderate"
    assert json.loads((revised / "freeze.json").read_text(encoding="utf8"))["source"] == str(root.resolve())


def test_default_evaluation_root_is_found_without_new_video_argument(holdout, tmp_path):
    _, historical, runs = holdout
    frozen = make_reviewed_frozen(holdout, tmp_path / "prior_frozen")
    register_completed_evaluation(historical.parent / "evaluation", frozen)
    inv = history_inventory(historical, runs)
    assert not allowed_at(timestamp("2026-10-20T16:29:20"), inv, "temporal_gap", 30)


@pytest.mark.parametrize("missing", ["metrics", "samples", "freeze"])
def test_completed_evaluation_missing_registered_evidence_fails_closed(holdout, tmp_path, missing):
    _, historical, runs = holdout
    frozen = make_reviewed_frozen(holdout, tmp_path / "prior_frozen")
    comparison, metrics = register_completed_evaluation(tmp_path / "evaluation", frozen)
    inv = history_inventory(historical, runs)
    path = {"metrics": metrics, "samples": frozen / "samples.jsonl", "freeze": frozen / "freeze.json"}[missing]
    path.unlink()
    # A fresh scan must not silently skip this completed comparison. Either
    # missing-file or explicit validation errors are appropriate fail-closed behavior.
    with pytest.raises((ValueError, FileNotFoundError)):
        history_inventory(historical, runs)


@pytest.mark.parametrize("changed", ["comparison", "metrics", "samples", "freeze"])
def test_completed_evaluation_evidence_changes_cannot_be_silently_accepted(holdout, tmp_path, changed):
    _, historical, runs = holdout
    frozen = make_reviewed_frozen(holdout, tmp_path / "prior_frozen")
    comparison, metrics = register_completed_evaluation(tmp_path / "evaluation", frozen)
    before = history_inventory(historical, runs)
    path = {"comparison": comparison, "metrics": metrics,
            "samples": frozen / "samples.jsonl", "freeze": frozen / "freeze.json"}[changed]
    path.write_text(path.read_text(encoding="utf8") + "\n", encoding="utf8")
    try:
        after = history_inventory(historical, runs)
    except ValueError:
        # Per-comparison checksum verification may reject the change earlier.
        return
    with pytest.raises(ValueError, match="Historical evidence changed"):
        evidence_unchanged(before, after)


def test_removed_completed_comparison_is_detected_against_saved_inventory(holdout, tmp_path):
    _, historical, runs = holdout
    frozen = make_reviewed_frozen(holdout, tmp_path / "prior_frozen")
    comparison, _ = register_completed_evaluation(tmp_path / "evaluation", frozen)
    before = history_inventory(historical, runs)
    comparison.unlink()
    after = history_inventory(historical, runs)
    with pytest.raises(ValueError, match="Historical evidence changed"):
        evidence_unchanged(before, after)
