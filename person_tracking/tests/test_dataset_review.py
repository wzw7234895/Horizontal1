import json
import threading
import urllib.error
import urllib.request
from datetime import datetime

import numpy as np
import pytest

from person_tracking.common import write_json
from person_tracking.dataset import assign_splits, group_for_time, read_labels, save_jpeg, validate_boxes, write_labels
from person_tracking.review import ReviewStore, export_dataset, make_server


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    for folder in ("images", "proposals/labels", "reviewed_labels"):
        (root / folder).mkdir(parents=True)
    rows = []
    for i, split in enumerate(("train", "val", "test", "train")):
        row = {"id": f"sample_{i}", "camera_id": "D01", "split": split, "split_group": f"g{i}",
               "recording_time_estimate": f"2026-10-20T15:{i*10:02d}:30", "proposal_count": 1,
               "width": 160, "height": 100, "image": f"images/sample_{i}.jpg",
               "proposal_label": f"proposals/labels/sample_{i}.txt"}
        save_jpeg(root / row["image"], np.zeros((100, 160, 3), np.uint8))
        write_labels(root / row["proposal_label"], [[0.5, 0.5, 0.2, 0.6]])
        rows.append(row)
    (root / "samples.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    write_json(root / "summary.json", {"status": "completed"})
    return root


def test_time_group_and_boundary_gap_prevent_same_moment_cross_split():
    times = [datetime(2026, 10, 20, 15, minute, second) for minute, second in
             [(1, 0), (3, 3), (8, 59), (11, 0), (21, 0), (31, 0), (41, 0), (51, 0)]]
    groups = [group_for_time(t) for t in times]
    split = assign_splits(groups)
    assert groups[0] == groups[1] == groups[2]
    assert len({split[g] for g in groups[:3]}) == 1
    assert set(split.values()) == {"train", "val", "test"}
    assert group_for_time(datetime(2026, 10, 20, 15, 0, 59)) is None
    assert group_for_time(datetime(2026, 10, 20, 15, 9, 0)) is None
    assert assign_splits(groups) == split


def test_unreviewed_proposals_never_export(dataset, tmp_path):
    out = tmp_path / "export"
    with pytest.raises(ValueError, match="No reviewed"):
        export_dataset(dataset, out)
    assert not out.exists()
    assert ReviewStore(dataset).status()["reviewed"] == 0


def test_required_complete_review_blocks_partial_export(dataset, tmp_path):
    store = ReviewStore(dataset)
    for index in (0, 1, 2):
        store.save(index, "reviewed", [[0.5, 0.5, 0.2, 0.6]])
    write_json(dataset / "summary.json", {"status": "completed", "require_complete_review": True})
    output = tmp_path / "incomplete_export"
    with pytest.raises(ValueError, match="pending annotations"):
        export_dataset(dataset, output)
    assert not output.exists()


def test_negative_requires_explicit_review_and_does_not_change_proposals(dataset):
    store = ReviewStore(dataset)
    with pytest.raises(ValueError, match="confirm_empty"):
        store.save(0, "reviewed", [])
    assert not store.label_path(store.rows[0]).exists()
    store.save(0, "reviewed", [], confirm_empty=True)
    assert read_labels(store.label_path(store.rows[0])) == []
    assert len(read_labels(dataset / store.rows[0]["proposal_label"])) == 1
    assert ReviewStore(dataset).status()["confirmed_negative_images"] == 1


def test_inherited_test_annotation_is_readable_but_cannot_be_modified(dataset):
    store = ReviewStore(dataset)
    store.save(2, "reviewed", [[0.5, 0.5, 0.2, 0.6]])
    rows = store.rows
    rows[2]["annotation_locked"] = True
    (dataset / "samples.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    locked = ReviewStore(dataset)
    original_label = locked.label_path(rows[2]).read_bytes()
    original_reviews = locked.review_path.read_bytes()
    assert locked.item(2)["annotation_locked"] is True
    with pytest.raises(ValueError, match="sealed"):
        locked.save(2, "reviewed", [], confirm_empty=True)
    assert locked.label_path(rows[2]).read_bytes() == original_label
    assert locked.review_path.read_bytes() == original_reviews


def test_reopened_test_is_editable_and_pending_reselection_cannot_export(dataset, tmp_path):
    store = ReviewStore(dataset)
    store.save(2, "reviewed", [[0.5, 0.5, 0.2, 0.6]])
    rows = store.rows
    rows[2]["annotation_locked"] = False
    (dataset / "samples.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    write_json(dataset / "summary.json", {"status": "completed", "split_selection_status": "pending_reselection", "test_set_sealed": False})
    reopened = ReviewStore(dataset)
    assert reopened.item(2)["annotation_locked"] is False
    reopened.save(2, "reviewed", [[0.4, 0.5, 0.2, 0.4]])
    assert read_labels(reopened.label_path(rows[2])) == [[0.4, 0.5, 0.2, 0.4]]
    assert reopened.status()["split_selection_status"] == "pending_reselection"
    assert reopened.status()["test_set_sealed"] is False
    with pytest.raises(ValueError, match="reselect"):
        export_dataset(dataset, tmp_path / "export")
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("box", [[0.5, 0.5, 0, 0.2], [0.05, 0.5, 0.2, 0.3], [float('nan'), 0.5, 0.2, 0.3]])
def test_invalid_annotation_rejected(box):
    with pytest.raises(ValueError):
        validate_boxes([box])


def test_export_reviewed_only_preserves_held_out_splits(dataset, tmp_path):
    store = ReviewStore(dataset)
    for i in (0, 1, 2):
        store.save(i, "reviewed", [[0.5, 0.5, 0.2, 0.6]])
    output = tmp_path / "export"
    result = export_dataset(dataset, output)
    assert result["reviewed_images_exported"] == 3
    assert result["omitted_unreviewed_or_excluded"] == 1
    assert result["by_split"] == {"train": 1, "val": 1, "test": 1}
    assert (output / "data.yaml").is_file()
    for split in ("train", "val", "test"):
        assert len(list((output / "images" / split).glob("*.jpg"))) == 1
        assert len(list((output / "labels" / split).glob("*.txt"))) == 1
    for line in (output / "exported_samples.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert (output / row["image"]).is_file() and (output / row["label"]).is_file()
        assert row["review_status"] == "reviewed"
    (output / "images/train/sample_0.jpg").write_bytes(b"edited export")
    assert (dataset / "images/sample_0.jpg").read_bytes() != b"edited export"


def test_group_leakage_is_rejected(dataset, tmp_path):
    path = dataset / "samples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["split_group"] = rows[0]["split_group"]
    path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="multiple splits"):
        export_dataset(dataset, tmp_path / "export")


def test_review_http_reads_image_and_saves_label(dataset):
    server = make_server(dataset, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/items") as response:
            data = json.load(response)
        assert data["status"]["pending"] == 4
        with urllib.request.urlopen(base + "/api/image?index=0") as response:
            assert response.headers.get_content_type() == "image/jpeg"
            assert response.read()[:2] == b"\xff\xd8"
        payload = {"index": 0, "status": "reviewed", "boxes": [[0.5, 0.5, 0.2, 0.6]]}
        request = urllib.request.Request(base + "/api/review", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["status"]["reviewed"] == 1
        with urllib.request.urlopen(base + "/api/item?index=0") as response:
            assert json.load(response)["boxes"] == [[0.5, 0.5, 0.2, 0.6]]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
