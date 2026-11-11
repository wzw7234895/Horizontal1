import json

import cv2
import numpy as np
import pytest

from person_tracking.b3b4_data import B3B4ReviewStore, _taxonomy, _validate_boxes, export_yolo_dataset
from person_tracking.common import write_json


def test_taxonomy_has_explicit_part_and_color_classes(tmp_path):
    policy = tmp_path / "policy.json"
    write_json(policy, {"candidate_label_colors": ["blue", "white"]})
    taxonomy = _taxonomy(policy)
    assert taxonomy["classes"] == ["head_no_helmet", "helmet_blue", "helmet_white",
                                   "torso_blue", "torso_white"]
    assert taxonomy["customer_policy_confirmed"] is False


def test_box_validation_requires_known_class_and_image_bounds():
    assert _validate_boxes([{"class_name": "helmet_blue", "xyxy": [1, 2, 10, 12]}],
                           20, 30, {"helmet_blue"}) == [
                               {"class_name": "helmet_blue", "xyxy": [1.0, 2.0, 10.0, 12.0]}]
    with pytest.raises(ValueError, match="inside the image"):
        _validate_boxes([{"class_name": "helmet_blue", "xyxy": [1, 2, 21, 12]}],
                        20, 30, {"helmet_blue"})
    with pytest.raises(ValueError, match="class_name"):
        _validate_boxes([{"class_name": "person", "xyxy": [1, 2, 10, 12]}],
                        20, 30, {"helmet_blue"})


def make_review_data(root):
    (root / "images").mkdir(parents=True)
    (root / "proposals" / "labels").mkdir(parents=True)
    rows = []
    classes = ["head_no_helmet", "helmet_blue", "torso_blue"]
    annotations = {}
    for index, split in enumerate(("train", "val", "test")):
        image_name = f"D02_{index}.jpg"
        image_path = root / "images" / image_name
        assert cv2.imwrite(str(image_path), np.zeros((32, 48, 3), dtype=np.uint8))
        sample_id = f"D02_{index}"
        rows.append({"id": sample_id, "camera_id": "D02", "split": split,
                     "split_group": f"group_{split}", "recording_time_estimate": f"2026-11-10T12:0{index}:00",
                     "image": f"images/{image_name}", "width": 48, "height": 32,
                     "proposal_count": 0, "proposal_label": f"proposals/labels/{sample_id}.txt"})
        (root / rows[-1]["proposal_label"]).write_text("", encoding="utf-8")
        annotations[sample_id] = {"status": "reviewed", "frame_complete": True,
                                  "boxes": [{"class_name": name, "xyxy": [2, 2, 12, 14]}
                                            for name in classes]}
    (root / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    write_json(root / "summary.json", {"status": "completed"})
    write_json(root / "b3b4_taxonomy.json", {"classes": classes})
    write_json(root / "b3b4_annotations.json", annotations)


def test_export_preserves_time_groups_and_excludes_test_from_training_readiness(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    make_review_data(source)
    store = B3B4ReviewStore(source)
    assert store.status()["reviewed"] == 3
    output = tmp_path / "export"
    result = export_yolo_dataset(source, output)
    assert result["training_ready"] is True
    assert result["boxes_by_split"]["val"] == {
        "head_no_helmet": 1, "helmet_blue": 1, "torso_blue": 1,
    }
    assert (output / "exported_samples.jsonl").is_file()
    with pytest.raises(FileExistsError):
        export_yolo_dataset(source, output)


def test_review_save_requires_full_frame_confirmation(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    make_review_data(source)
    store = B3B4ReviewStore(source)
    with pytest.raises(ValueError, match="Confirm the whole frame"):
        store.save(0, "reviewed", False, [])
