"""Finish one launched v3 run: val baseline, optional full-80 continuation, and val results."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from person_tracking.common import write_json
from person_tracking.experiments.train_person_static_v1 import file_sha256


class PauseRequested(RuntimeError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def ensure_not_paused(run_paths, marker):
    if marker.exists():
        raise PauseRequested(f"Pause marker exists: {marker}")
    for run in run_paths:
        config_path = run / "training_config.json"
        if config_path.is_file() and read_json(config_path).get("training_status") == "paused_by_user":
            raise PauseRequested(f"Training paused by user: {run}")


def relative_improvement(baseline, candidate):
    a, b = baseline["operating_point"]["all"], candidate["operating_point"]["all"]
    return {
        "precision_not_lower": b["precision"] >= a["precision"],
        "recall_not_lower": b["recall"] >= a["recall"],
        "f1_improved": b["f1"] > a["f1"],
        "map50_not_lower": candidate["map_metrics"]["metrics/mAP50(B)"] >= baseline["map_metrics"]["metrics/mAP50(B)"],
        "map50_95_not_lower": candidate["map_metrics"]["metrics/mAP50-95(B)"] >= baseline["map_metrics"]["metrics/mAP50-95(B)"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    launch = read_json(args.launch)
    run = Path(launch["output"])
    watched_runs = [run]
    pause_marker = run / "PAUSE_REQUESTED"
    argv = launch["arguments"]
    dataset = Path(argv[argv.index("--dataset") + 1])
    record = {"status": "waiting_for_training", "created_at": datetime.now().isoformat(),
        "initial_run": str(run), "dataset": str(dataset), "test_split_evaluated": False,
        "decision_policy": "After early stop, require fixed P/R not lower, F1 higher, and both mAP50 and mAP50-95 not lower than public baseline on the same corrected val; otherwise continue to 80 total epochs.",
        "quality_scope": "Relative development comparison, not an absolute field-acceptance threshold.",
        "pause_marker": str(pause_marker)}
    write_json(output / "status.json", record)

    def execute(module, options, name):
        ensure_not_paused(watched_runs, pause_marker)
        command = [sys.executable, "-u", "-m", module, *map(str, options)]
        record.update(status=name, updated_at=datetime.now().isoformat(), active_command=command)
        write_json(output / "status.json", record)
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
        with (output / f"{name}.stdout.log").open("xb") as stdout, (output / f"{name}.stderr.log").open("xb") as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            record["active_child_pid"] = child.pid
            write_json(output / "status.json", record)
            try:
                while child.poll() is None:
                    ensure_not_paused(watched_runs, pause_marker)
                    time.sleep(1)
                code = child.returncode
            except PauseRequested:
                if child.poll() is None:
                    child.terminate()
                    child.wait()
                raise
        record.pop("active_child_pid", None)
        if code:
            raise RuntimeError(f"{name} failed with exit {code}; see saved logs")

    def evaluate(weights, name):
        execute("person_tracking.experiments.evaluate_person_v2", ["--weights", weights, "--dataset", dataset,
            "--out", output / name, "--device", "0", "--batch", "4"], name)
        return read_json(output / name / "metrics.json")

    try:
        process = psutil.Process(launch["pid"])
        initial_create_time = process.create_time()
        if "person_tracking.experiments.train_person_v2" not in process.cmdline():
            raise RuntimeError("Launch PID is not the expected training process")
        while True:
            ensure_not_paused(watched_runs, pause_marker)
            cfg_path = run / "training_config.json"
            config = read_json(cfg_path) if cfg_path.is_file() else {}
            if config.get("training_status") == "paused_by_user":
                record.update(status="paused_with_training", completed_at=datetime.now().isoformat())
                write_json(output / "status.json", record)
                return
            if config.get("training_completed"):
                break
            if config.get("training_status") == "failed":
                raise RuntimeError(f"Initial training failed: {config.get('failure')}")
            if not process.is_running() or process.create_time() != initial_create_time:
                time.sleep(1)
                config = read_json(cfg_path) if cfg_path.is_file() else {}
                if not config.get("training_completed"):
                    raise RuntimeError("Training process exited before reporting completion; evaluation not started")
                break
            time.sleep(10)
        # Wait for the initial process to release GPU memory before validation.
        process.wait()
        ensure_not_paused(watched_runs, pause_marker)
        config = read_json(run / "training_config.json")
        initialization = Path(config["model"])
        if file_sha256(initialization) != config["model_sha256"]:
            raise RuntimeError("Recorded public initialization changed before baseline evaluation")
        baseline = evaluate(initialization, "public_pretrained_on_v3_val")
        early = evaluate(run / "weights/best.pt", "initial_best_on_v3_val")
        checks = relative_improvement(baseline, early)
        ensure_not_paused(watched_runs, pause_marker)
        record["early_comparison_checks"] = checks
        if config["early_stopped"] and not all(checks.values()):
            continued = run.with_name(run.name + "_full80")
            watched_runs.append(continued)
            execute("person_tracking.experiments.resume_person_v2_full80", ["--source-run", run, "--out", continued,
                "--device", "0", "--wait-for-idle", "--min-free-mib", "3400", "--max-gpu-utilization", "20"], "continue_full80")
            run = continued
            candidate = evaluate(run / "weights/best.pt", "full80_best_on_v3_val")
            last = evaluate(run / "weights/last.pt", "full80_last_on_v3_val")
        else:
            candidate = early
            last = evaluate(run / "weights/last.pt", "initial_last_on_v3_val")
        config = read_json(run / "training_config.json")
        result = {"status": "completed", "completed_at": datetime.now().isoformat(), "selected_run": str(run),
            "selected_checkpoint": str(run / "weights/best.pt"), "epochs_completed": config["epochs_completed"],
            "best_epoch": config["best_epoch"], "baseline": baseline, "best": candidate, "last": last,
            "comparison_checks": relative_improvement(baseline, candidate), "test_split_evaluated": False,
            "limitations": ["Development val also selected checkpoints; not an independent test estimate.",
                "Single day and a partly model-assisted source sample pool; no field acceptance claim.",
                "No old finetuned model used as an unseen-holdout baseline."]}
        write_json(output / "comparison.json", result)
        record.update(status="completed", selected_checkpoint=result["selected_checkpoint"], completed_at=result["completed_at"])
        record.pop("active_command", None)
        write_json(output / "status.json", record)
    except PauseRequested as error:
        if process.is_running() and process.create_time() == initial_create_time:
            process.terminate()
            process.wait()
        record.pop("active_child_pid", None)
        record.update(status="paused_by_user", reason=str(error), updated_at=datetime.now().isoformat())
        write_json(output / "status.json", record)
    except BaseException as error:
        record.update(status="failed", failure=repr(error), updated_at=datetime.now().isoformat())
        write_json(output / "status.json", record)
        raise


if __name__ == "__main__":
    main()
