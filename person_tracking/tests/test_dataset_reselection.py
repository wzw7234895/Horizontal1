from person_tracking.experiments.prepare_person_v3 import select_grouped_splits


def test_reselection_keeps_cameras_together_and_ignores_predictions_and_labels():
    rows = [{"id": f"g{i}_c{c}_{j}", "split_group": f"g{i}", "camera_id": f"D0{c}",
             "proposal_count": i, "labels": []}
            for i in range(6) for c in range(1, 5) for j in range(i + 1)]
    first = select_grouped_splits(rows, seed=20261106)
    changed = [{**r, "proposal_count": 999, "labels": [[0.1, 0.2, 0.3, 0.4]]} for r in rows]
    assert select_grouped_splits(changed, seed=20261106) == first
    assert select_grouped_splits(list(reversed(rows)), seed=20261106) == first
    assert set(first.values()) == {"train", "val", "test"}
    for split in ("train", "val", "test"):
        assert {r["camera_id"] for r in rows if first[r["split_group"]] == split} == {"D01", "D02", "D03", "D04"}
