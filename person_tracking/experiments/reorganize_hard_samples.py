"""Register hard samples and summarize grouped development validation.

The registry keeps training candidates, pending review images, and validation
evidence in separate roles.  Image-level condition tags from the training
selection are retained as hints; they are never promoted to person-level
ground truth.  The script does not copy images, change labels, train models,
or modify a sealed test split.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from person_tracking.common import PACKAGE_DIR, read_jsonl, write_json
from person_tracking.occlusion import group_keys, wilson


 # Keep defaults relative to the project root.  On this Windows workspace,
 # resolving a path containing the Chinese project name before opening it can
 # convert the current directory to a legacy-codepage string.
PROJECT_ROOT = Path("person_tracking")
DEFAULT_V12_EXPORT = PROJECT_ROOT / "data" / "person_reviewed_v12_occlusion_augmented_20261108"
DEFAULT_V13_D03 = PROJECT_ROOT / "data" / "person_v13_d03_hard_review_20261108"
DEFAULT_V13_D04 = PROJECT_ROOT / "data" / "person_v13_d04_hard_review_20261108"
DEFAULT_HOLDOUT = PROJECT_ROOT / "evaluation" / "occlusion_20261107" / "frozen_v1"
DEFAULT_METRICS = (PROJECT_ROOT / "evaluation" / "occlusion_20261107" /
                   "results_v12_v1_dev_recheck_20261108" /
                   "model_1_person_yolo11n_static_v12_occlusion_augmented_20261108" /
                   "metrics.json")


CONDITION_MAP = {
    "俯视": "overhead",
    "斜俯视": "oblique_overhead",
    "工件或吊具遮挡": "load_occlusion",
    "结构遮挡": "structure_occlusion",
    "结构局部遮挡": "structure_occlusion",
    "人员交叠": "person_overlap",
    "边缘截断": "edge_truncation",
    "边缘进出": "edge_entry_exit",
    "远处小目标": "far_small_target",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def review_map(root: Path) -> dict:
    path = root / "reviews.json"
    return read_json(path) if path.is_file() else {}


def line_count(path: Path) -> int | None:
    if not path or not path.is_file():
        return None
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def normalize_conditions(values) -> list[str]:
    result = []
    for value in values or []:
        normalized = CONDITION_MAP.get(str(value), str(value))
        if normalized not in result:
            result.append(normalized)
    return sorted(result)


def time_group(row: dict) -> str | None:
    return row.get("split_group") or row.get("window_id") or row.get("evaluation_group")


def base_registry_row(row: dict, *, role: str, status: str, root: Path,
                      source_kind: str, training_eligible: bool,
                      validation_eligible: bool, independent: bool) -> dict:
    conditions = normalize_conditions(row.get("target_conditions"))
    if not conditions and row.get("candidate_type"):
        conditions = [str(row["candidate_type"])]
    label = row.get("label") or row.get("reviewed_label")
    label_count = line_count(root / label) if label else None
    return {
        "id": row["id"],
        "role": role,
        "status": status,
        "source_kind": source_kind,
        "camera_id": row.get("camera_id"),
        "split": row.get("split"),
        "split_group": time_group(row),
        "recording_time_estimate": row.get("recording_time_estimate"),
        "video_time_s": row.get("video_time_s"),
        "source": row.get("source"),
        "image": row.get("image"),
        "label": label,
        "confirmed_person_boxes": label_count,
        "target_conditions": row.get("target_conditions", []),
        "normalized_conditions": conditions,
        "candidate_type": row.get("candidate_type"),
        "suspected_unassociated_detection": row.get("suspected_unassociated_detection"),
        "suspected_short_track": row.get("suspected_short_track"),
        "event_group": row.get("event_group"),
        "window_id": row.get("window_id"),
        "condition_status": row.get("condition_status", "not_collected"),
        "training_eligible": training_eligible,
        "validation_eligible": validation_eligible,
        "independent_validation": independent,
        "sealed_test": row.get("split") == "test",
        "source_dataset": str(root),
    }


def load_training_candidates(v12_export: Path) -> list[dict]:
    rows = read_jsonl(v12_export / "exported_samples.jsonl")
    # The exported v12 manifest carries event_group only for the 132 hard
    # multi-event additions.  Baseline train/val/test images are excluded.
    additions = [row for row in rows if row.get("event_group")]
    result = []
    for row in additions:
        result.append(base_registry_row(
            row, role="train_hard_candidate", status="reviewed", root=v12_export,
            source_kind="v12_multievent_training", training_eligible=True,
            validation_eligible=False, independent=False))
    return result


def load_hard_review(dataset: Path) -> list[dict]:
    rows = read_jsonl(dataset / "samples.jsonl")
    reviews = review_map(dataset)
    result = []
    for row in rows:
        if row.get("annotation_locked") is not False:
            continue
        review = reviews.get(row["id"], {})
        status = review.get("status", "pending")
        role = ("train_hard_candidate" if status == "reviewed" else
                "excluded_candidate" if status == "excluded" else "pending_review")
        label = f"reviewed_labels/{row['id']}.txt" if status == "reviewed" else None
        copied = {**row, "label": label}
        item = base_registry_row(
            copied, role=role, status=status, root=dataset,
            source_kind="v13_hard_review", training_eligible=status == "reviewed",
            validation_eligible=False, independent=False)
        item["confirmed_empty"] = bool(review.get("confirmed_empty", False))
        item["review_box_count"] = review.get("box_count")
        item["reviewed_at"] = review.get("reviewed_at")
        item["hard_reasons"] = [
            reason for reason, enabled in (
                ("detector_candidate", row.get("candidate_type") == "detector_candidate_review"),
                ("no_detection_candidate", row.get("candidate_type") == "no_detection_review"),
                ("unassociated_detection", bool(row.get("suspected_unassociated_detection"))),
                ("short_track", bool(row.get("suspected_short_track"))),
            ) if enabled
        ]
        result.append(item)
    return result


def load_validation_people(holdout: Path, metrics_path: Path | None) -> tuple[list[dict], dict]:
    rows = read_jsonl(holdout / "samples.jsonl")
    reviews = review_map(holdout)
    per_person = {}
    if metrics_path and metrics_path.is_file():
        metrics_root = metrics_path.parent
        per_path = metrics_root / "per_person.jsonl"
        if per_path.is_file():
            per_person = {row["id"]: row for row in read_jsonl(per_path)}
    people = []
    image_registry = []
    for row in rows:
        review = reviews.get(row["id"], {})
        status = review.get("status", "pending")
        image_registry.append(base_registry_row(
            row, role="dev_holdout", status=status, root=holdout,
            source_kind="frozen_same_day_holdout", training_eligible=False,
            validation_eligible=status == "reviewed", independent=False))
        if status != "reviewed":
            continue
        result = per_person.get(row["id"], {})
        result_objects = result.get("objects", [])
        for index, obj in enumerate(review.get("objects", [])):
            matched = result_objects[index].get("matched") if index < len(result_objects) else None
            people.append({
                "id": row["id"], "camera_id": row.get("camera_id"),
                "clip_id": row.get("clip_id"),
                "evaluation_group": row.get("evaluation_group", row.get("clip_id")),
                "recording_time_estimate": row.get("recording_time_estimate"),
                "box": obj["box"], "occlusion": obj.get("occlusion"),
                "occluders": obj.get("occluders", []),
                "distance": obj.get("distance"), "truncated": obj.get("truncated"),
                "matched": matched,
                "tp": 1 if matched is True else 0 if matched is False else None,
                "fn": 1 if matched is False else 0 if matched is True else None,
                "metrics_source": str(metrics_path) if metrics_path and metrics_path.is_file() else None,
                "independence_scope": "development_recheck_same_day",
            })
    metadata = {
        "rows": rows, "reviews": reviews, "image_registry": image_registry,
        "metrics": read_json(metrics_path) if metrics_path and metrics_path.is_file() else None,
    }
    return people, metadata


def registry_groups(registry: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in registry:
        conditions = row["normalized_conditions"] or ["unclassified"]
        for condition in conditions:
            groups[(row["role"], condition)].append(row)
    output = []
    for (role, condition), rows in sorted(groups.items()):
        output.append({
            "role": role, "group": condition, "images": len(rows),
            "reviewed": sum(r["status"] == "reviewed" for r in rows),
            "pending": sum(r["status"] == "pending" for r in rows),
            "excluded": sum(r["status"] == "excluded" for r in rows),
            "training_eligible": sum(bool(r["training_eligible"]) for r in rows),
            "cameras": ",".join(sorted({str(r["camera_id"]) for r in rows if r["camera_id"]})),
            "time_groups": len({r["split_group"] for r in rows if r["split_group"]}),
            "confirmed_person_boxes": sum(r["confirmed_person_boxes"] or 0 for r in rows),
            "person_level_truth": all(r["condition_status"] == "human_person_level" for r in rows),
        })
    return output


def validation_groups(people: list[dict], protocol: dict) -> list[dict]:
    groups = defaultdict(lambda: {"tp": 0, "fn": 0, "images": set(), "clips": set()})
    for person in people:
        if person["matched"] is None:
            continue
        obj = {"box": person["box"], "occlusion": person["occlusion"],
               "occluders": person["occluders"], "distance": person["distance"],
               "truncated": person["truncated"]}
        for category, name in group_keys(obj, person["camera_id"], protocol.get("small_visible_height_720p", 64)):
            group = groups[(category, name)]
            group["tp"] += person["tp"] or 0
            group["fn"] += person["fn"] or 0
            group["images"].add(person["id"])
            group["clips"].add(person["evaluation_group"])
    rows = []
    min_people = protocol.get("min_people_per_stratum", 30)
    min_clips = protocol.get("min_clips_per_stratum", 3)
    for (category, name), group in sorted(groups.items()):
        n = group["tp"] + group["fn"]
        supported = n >= min_people and len(group["clips"]) >= min_clips
        rows.append({
            "category": category, "stratum": name, "tp": group["tp"], "fn": group["fn"],
            "ground_truth_people": n, "positive_images": len(group["images"]),
            "positive_clips": len(group["clips"]),
            "recall": group["tp"] / n if n else None,
            "wilson95_descriptive": wilson(group["tp"], n),
            "coverage": "supported_counts" if supported else "limited" if n else "not_verified",
            "recall_target_met": (group["tp"] / n >= protocol.get("target_recall", 0.85)
                                  if supported and n else None),
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8-sig")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict, groups: list[dict], validation: list[dict]) -> None:
    lines = ["# 困难样本重组与分组验证登记", "",
             "本登记只组织样本和验证证据，不启动训练、不修改 v12 权重、不解封原 test。",
             "", "## 样本角色", "",
             "| 角色 | 数量 | 可用于训练 | 可用于验证 | 说明 |",
             "|---|---:|---:|---:|---|"]
    descriptions = {
        "train_hard_candidate": "人工复核通过的同日困难训练候选；不能作为独立验收集。",
        "pending_review": "尚未完成逐图人工复核；不得训练或计入真值。",
        "excluded_candidate": "人工排除；不得训练或验证。",
        "dev_holdout": "冻结开发诊断集；当前与历史同日记录有重叠，不是独立验收。",
    }
    for role, count in sorted(summary["roles"].items()):
        lines.append(f"| {role} | {count['images']} | {count['training_eligible']} | {count['validation_eligible']} | {descriptions.get(role, '')} |")
    lines += ["", "## 训练候选分组", "",
              "这些条件来自图像级筛选提示或候选来源；v12 多事件标签没有逐人员遮挡属性，不能直接当作 Recall 分母。", "",
              "| 角色 | 组 | 图像 | 已复核 | 待复核 | 排除 | 人员框 | 时间片 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in groups:
        lines.append(f"| {row['role']} | {row['group']} | {row['images']} | {row['reviewed']} | {row['pending']} | {row['excluded']} | {row['confirmed_person_boxes']} | {row['time_groups']} |")
    lines += ["", "## 开发分组验证", "",
              "固定协议使用 conf=0.10、匹配 IoU=0.50、1280×720 输入，仅统计人工可见人员框；完全不可见人员不计入逐帧 Recall。", "",
              "| 类别 | 分组 | TP | FN | 真值框 | 正样本帧 | 时间片 | Recall | 覆盖 |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in validation:
        recall = "未验证" if row["recall"] is None else f"{row['recall']:.2%}"
        lines.append(f"| {row['category']} | {row['stratum']} | {row['tp']} | {row['fn']} | {row['ground_truth_people']} | {row['positive_images']} | {row['positive_clips']} | {recall} | {row['coverage']} |")
    lines += ["", "## 结论和后续", "",
              "当前分组结果只能用于定位漏检来源；单组未达到至少 30 个真值人员框和 3 个独立时间片时标记为 limited。正式验收需新日期或新作业录像，且采样时不向标注人员展示模型候选框。",
              "", f"生成摘要：`{summary['generated_at']}`。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def organize(args) -> dict:
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Choose a new --out directory: {out}")
    out.mkdir(parents=True)
    v12 = Path(args.v12_export)
    registry = load_training_candidates(v12)
    for dataset in args.v13_review:
        registry.extend(load_hard_review(Path(dataset)))
    holdout = Path(args.holdout)
    metrics_path = Path(args.metrics) if args.metrics else None
    people, holdout_meta = load_validation_people(holdout, metrics_path)
    registry.extend(holdout_meta["image_registry"])
    registry.sort(key=lambda row: (row["role"], row.get("camera_id") or "", row["id"]))
    groups = registry_groups(registry)
    protocol_path = args.holdout / "protocol.json"
    protocol = read_json(protocol_path) if protocol_path.is_file() else {
        "min_people_per_stratum": 30, "min_clips_per_stratum": 3,
        "target_recall": 0.85, "small_visible_height_720p": 64,
    }
    validation = validation_groups(people, protocol)
    roles = {}
    for row in registry:
        item = roles.setdefault(row["role"], {"images": 0, "training_eligible": 0, "validation_eligible": 0})
        item["images"] += 1
        item["training_eligible"] += int(row["training_eligible"])
        item["validation_eligible"] += int(row["validation_eligible"])
    metrics = holdout_meta.get("metrics") or {}
    pending_count = roles.get("pending_review", {}).get("images", 0)
    limitations = [
        "v12 target_conditions are image-level selection hints, not person-level truth.",
        "The frozen holdout has only five person boxes and one evaluation clip in the current recheck.",
        "Same-day development results do not establish new-date or new-operation generalization.",
    ]
    if pending_count:
        limitations.insert(1, f"{pending_count} hard-review images remain pending and are unusable for training or truth.")
    else:
        limitations.insert(1, "All D03/D04 hard-review images are finalized; only reviewed additions are training eligible.")
    summary = {
        "status": "completed", "generated_at": __import__("datetime").datetime.now().isoformat(),
        "registry_images": len(registry), "validation_people": len(people), "roles": roles,
        "group_rows": len(groups), "validation_group_rows": len(validation),
        "v12_export": str(v12), "v13_review_datasets": [str(Path(p)) for p in args.v13_review],
        "holdout": str(holdout), "metrics": str(metrics_path) if metrics_path else None,
        "training_started": False, "v12_weight_modified": False, "original_test_modified": False,
        "independence_scope": metrics.get("independence_scope", "development_recheck_same_day"),
        "independent_validation": False,
        "limitations": limitations,
    }
    (out / "hard_sample_registry.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in registry), encoding="utf-8")
    (out / "validation_people.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in people), encoding="utf-8")
    write_csv(out / "hard_sample_groups.csv", groups)
    write_csv(out / "validation_groups.csv", validation)
    write_json(out / "summary.json", summary)
    write_report(out / "report.md", summary, groups, validation)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--v12-export", type=Path, default=DEFAULT_V12_EXPORT)
    parser.add_argument("--v13-review", type=Path, nargs="*", default=[DEFAULT_V13_D03, DEFAULT_V13_D04])
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    args = parser.parse_args(argv)
    organize(args)


if __name__ == "__main__":
    main()
