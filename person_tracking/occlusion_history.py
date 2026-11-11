"""Conservative, cross-camera temporal exclusion of historical model exposure."""
from __future__ import annotations

from bisect import bisect_left
from datetime import datetime, timedelta
import json
import math
from pathlib import Path

from .occlusion import sha256


def timestamp(value):
    when = datetime.fromisoformat(value)
    if when.tzinfo is not None:
        raise ValueError("Use the local naive recording clock consistently")
    return (when - datetime(1970, 1, 1)).total_seconds()


def iso(seconds):
    return (datetime(1970, 1, 1) + timedelta(seconds=seconds)).isoformat()


def merge_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Interval ends before it starts")
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def subtract_intervals(start, end, blocked):
    cursor, allowed = start, []
    for lo, hi in merge_intervals(blocked):
        if hi <= cursor or lo >= end:
            continue
        if lo > cursor:
            allowed.append([cursor, min(lo, end)])
        cursor = max(cursor, hi)
    if cursor < end:
        allowed.append([cursor, end])
    return allowed


def _evaluation_family(dataset):
    """Follow frozen-copy provenance so a holdout can be rechecked in place."""
    current, visited, evidence = Path(dataset).resolve(), set(), {}
    while True:
        if current in visited:
            raise ValueError(f"Cyclic frozen dataset provenance: {dataset}")
        visited.add(current)
        frozen = current / "freeze.json"
        if not frozen.is_file():
            return current, evidence
        evidence[str(frozen)] = sha256(frozen)
        metadata = json.loads(frozen.read_text(encoding="utf8"))
        source = metadata.get("source")
        if not source:
            raise ValueError(f"Frozen dataset has no source provenance: {frozen}")
        source = Path(source)
        current = (source if source.is_absolute() else current / source).resolve()


def history_inventory(data_root, runs_root, evaluation_root=None, ignore_evaluation_datasets=()):
    """All versions and all splits, including abandoned/corrected annotations.

    Only project data manifests are accepted as samples. Run sample snapshots
    are accepted when their real camera/source metadata are present. Fixture
    rows without a recording source are not evidence of actual model exposure.
    """
    data_root, runs_root = Path(data_root).resolve(), Path(runs_root).resolve()
    evaluation_root = Path(evaluation_root).resolve() if evaluation_root is not None else data_root.parent / "evaluation"
    ignored_families = {_evaluation_family(p)[0] for p in ignore_evaluation_datasets}
    if not data_root.is_dir() or not runs_root.is_dir():
        raise ValueError("Historical data and run roots must exist")
    sample_files = sorted(set(data_root.rglob("*samples*.jsonl")) | set(runs_root.rglob("*samples*.jsonl")))
    frame_files = sorted(runs_root.rglob("frames.jsonl"))
    config_files = sorted(runs_root.glob("train/*/training_config.json"))
    if not sample_files:
        raise ValueError("No historical sample manifests found")
    evidence, points, sources, groups, inference, skipped = {}, set(), set(), set(), [], 0

    def consume_samples(path, require_provenance=False):
        nonlocal skipped
        evidence[str(path)] = sha256(path)
        accepted = 0
        with path.open(encoding="utf8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("source") or not row.get("camera_id") or not row.get("recording_time_estimate"):
                    if require_provenance:
                        raise ValueError(f"Completed evaluation has a sample without real recording provenance: {path}")
                    skipped += 1
                    continue
                value = timestamp(row["recording_time_estimate"])
                points.add(value)
                groups.add(math.floor(value / 600) * 600)
                sources.add(str(Path(row["source"]).resolve()))
                accepted += 1
        if require_provenance and not accepted:
            raise ValueError(f"Completed evaluation has an empty sample manifest: {path}")
        return accepted

    for path in sample_files:
        consume_samples(path)
    for path in frame_files:
        evidence[str(path)] = sha256(path)
        lo = hi = None
        with path.open(encoding="utf8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("recording_time_estimate"):
                    continue
                value = timestamp(row["recording_time_estimate"])
                lo = value if lo is None else min(lo, value)
                hi = value if hi is None else max(hi, value)
        if lo is not None:
            inference.append({"path": str(path), "start": lo, "end": hi})
    for path in config_files:
        evidence[str(path)] = sha256(path)
        cfg = json.loads(path.read_text(encoding="utf8"))
        dataset = cfg.get("dataset_yaml")
        if dataset and not (Path(dataset).parent / "exported_samples.jsonl").is_file():
            raise ValueError(f"Historical training manifest is missing: {path}")

    # A completed holdout comparison has already been used to inspect model
    # quality. It must not be silently sampled again as a new independent set.
    # During freeze/re-evaluation only that same holdout family is exempted;
    # unrelated completed evaluations remain historical exposure.
    evaluations, ignored_evaluations, seen_datasets = [], [], {}
    for comparison in sorted(evaluation_root.rglob("comparison.json")) if evaluation_root.is_dir() else []:
        entries = json.loads(comparison.read_text(encoding="utf8"))
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Invalid completed evaluation index: {comparison}")
        for entry in entries:
            metrics_value = entry.get("metrics") if isinstance(entry, dict) else None
            if not metrics_value:
                raise ValueError(f"Completed evaluation has no metrics path: {comparison}")
            metrics_path = Path(metrics_value)
            metrics_path = (metrics_path if metrics_path.is_absolute() else comparison.parent / metrics_path).resolve()
            if not metrics_path.is_file():
                raise ValueError(f"Completed evaluation metrics are missing: {metrics_path}")
            metrics = json.loads(metrics_path.read_text(encoding="utf8"))
            dataset_value = metrics.get("dataset")
            if not dataset_value:
                raise ValueError(f"Completed evaluation has no dataset provenance: {metrics_path}")
            dataset = Path(dataset_value)
            dataset = (dataset if dataset.is_absolute() else metrics_path.parent / dataset).resolve()
            frozen = dataset / "freeze.json"
            expected_freeze = metrics.get("freeze_sha256")
            if not expected_freeze or not frozen.is_file():
                raise ValueError(f"Completed evaluation frozen provenance is missing: {metrics_path}")
            if sha256(frozen) != expected_freeze:
                raise ValueError(f"Completed evaluation frozen provenance changed: {frozen}")
            family, frozen_evidence = _evaluation_family(dataset)
            record = {"comparison": str(comparison), "metrics": str(metrics_path),
                      "dataset": str(dataset), "family": str(family)}
            if family in ignored_families:
                ignored_evaluations.append(record)
                continue
            manifest = dataset / "samples.jsonl"
            if not manifest.is_file():
                raise ValueError(f"Completed evaluation sample manifest is missing: {manifest}")
            evidence[str(comparison)] = sha256(comparison)
            evidence[str(metrics_path)] = sha256(metrics_path)
            evidence.update(frozen_evidence)
            if dataset not in seen_datasets:
                seen_datasets[dataset] = consume_samples(manifest, require_provenance=True)
            evaluations.append({**record, "sample_manifest": str(manifest), "sample_count": seen_datasets[dataset]})
    if not points:
        raise ValueError("Historical manifests contained no real recording timestamps")
    return {"data_root": str(data_root), "runs_root": str(runs_root), "evaluation_root": str(evaluation_root),
            "evidence_sha256": evidence, "sample_times": sorted(points),
            "used_groups": sorted(groups), "sources": sorted(sources),
            "inference_intervals": inference, "ignored_rows_without_source_clock": skipped,
            "completed_evaluations": evaluations, "ignored_same_family_evaluations": ignored_evaluations,
            "policy": "Exclude all historical splits/versions, inference spans and completed holdout evaluations on the common recording clock."}


def blocked_intervals(history, mode, guard_seconds):
    if mode not in ("strict_group", "temporal_gap") or not math.isfinite(guard_seconds) or guard_seconds < 0:
        raise ValueError("Invalid temporal isolation policy")
    blocks = [[value - guard_seconds, value + guard_seconds] for value in history["sample_times"]]
    blocks += [[item["start"] - guard_seconds, item["end"] + guard_seconds]
               for item in history["inference_intervals"]]
    if mode == "strict_group":
        blocks += [[start, start + 600] for start in history["used_groups"]]
    return merge_intervals(blocks)


def allowed_at(value, history, mode, guard_seconds):
    points = history["sample_times"]
    index = bisect_left(points, value)
    if any(abs(points[j] - value) < guard_seconds + 1e-6 for j in (index-1, index) if 0 <= j < len(points)):
        return False
    if any(item["start"] - guard_seconds <= value <= item["end"] + guard_seconds
           for item in history["inference_intervals"]):
        return False
    if mode == "strict_group":
        group = math.floor(value / 600) * 600
        phase = value - group
        return group not in set(history["used_groups"]) and 60 <= phase < 540
    return True


def assert_isolation(rows, history, protocol):
    for row in rows:
        if not allowed_at(timestamp(row["recording_time_estimate"]), history, protocol["mode"], protocol["guard_seconds"]):
            raise ValueError(f"Historical temporal overlap: {row['id']}")


def evidence_unchanged(recorded, current):
    # New history is allowed only after all holdout timestamps have been
    # rechecked; modifications to previously audited evidence fail closed.
    for path, digest in recorded["evidence_sha256"].items():
        if current["evidence_sha256"].get(path) != digest:
            raise ValueError(f"Historical evidence changed or disappeared: {path}")
