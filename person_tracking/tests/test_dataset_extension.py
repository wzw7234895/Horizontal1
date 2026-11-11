from collections import Counter

from person_tracking.experiments.expand_person_dataset import eligible_group, select_additions


def camera():
    return {"recording_start": "2026-10-20T15:00:00", "time_offset_s": 0}


def test_extension_rejects_test_clock_disagreement_and_nearby_old_frames():
    groups = {"2026-10-20T15:00:00": "train", "2026-10-20T15:10:00": "test"}
    assert eligible_group(camera(), 120, 3000, 25, groups, []) == "2026-10-20T15:00:00"
    assert eligible_group(camera(), 720, 18000, 25, groups, []) is None
    assert eligible_group(camera(), 120, 18000, 25, groups, []) is None
    assert eligible_group(camera(), 120, 3000, 25, groups, [122]) is None
    assert eligible_group(camera(), 30, 750, 25, groups, []) is None


def test_validation_selection_is_independent_of_model_predictions():
    original = [{"id": "old", "camera_id": "D01", "split": "train"}]
    candidates = []
    for split in ("train", "val"):
        for i in range(30):
            candidates.append({"id": f"{split}_{i}", "camera_id": "D01", "split": split,
                               "split_group": split, "video_time_s": i*15,
                               "proposal_count": int(i%3 == 0), "detector_disagreement": bool(i%2),
                               "near_reviewed_train_person": False})
    first = select_additions(candidates, original, target_total=12, val_per_camera=4)
    changed = [{**r, "proposal_count": 12-r["proposal_count"],
                "detector_disagreement": not r["detector_disagreement"]} for r in candidates]
    second = select_additions(changed, original, target_total=12, val_per_camera=4)
    assert {r["id"] for r in first if r["split"] == "val"} == {r["id"] for r in second if r["split"] == "val"}
    assert len(first) == 11
    assert Counter(r["split"] for r in first) == {"train": 7, "val": 4}
