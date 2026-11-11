from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .common import DEFAULT_CONFIG, PACKAGE_DIR, load_config, write_json


def main():
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PACKAGE_DIR / ".cache" / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(PACKAGE_DIR / ".cache" / "matplotlib"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "occlusion":
        from .occlusion_benchmark import main as run_occlusion
        return run_occlusion(sys.argv[2:])
    parser = argparse.ArgumentParser(description="Crane person detection / tracking / video dataset tools")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Inspect real video metadata")
    inspect.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    inspect.add_argument("--cameras", nargs="+")
    inspect.add_argument("--out", type=Path, required=True)
    track = sub.add_parser("track", help="Run person detection and independent per-camera tracking")
    track.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    track.add_argument("--cameras", nargs="+")
    track.add_argument("--weights", type=Path, help="Optional model checkpoint override; config remains unchanged")
    track.add_argument("--out", type=Path, required=True)
    track.add_argument("--start", type=float, default=0, help="Seconds from each file start")
    track.add_argument("--start-clock", help="Approximate common HH:MM:SS (filename clock, not exposure sync)")
    track.add_argument("--duration", type=float)
    track.add_argument("--max-frames", type=int)
    track.add_argument("--fps", type=float)
    track.add_argument("--device", help="auto, cpu or CUDA device index, e.g. 0")
    track.add_argument("--imgsz", type=int)
    track.add_argument("--no-video", action="store_true")
    track.add_argument("--preview-width", type=int, default=1280)
    track.add_argument("--decode-mode", choices=("sequential", "seek"), default="sequential",
                       help="Files: sequential (default) preserves decoder context; seek is faster but may warn or locate inaccurately")
    doctor = sub.add_parser("doctor", help="Check environment, GPU, input files and model")
    doctor.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    doctor.add_argument("--out", type=Path)
    sample = sub.add_parser("sample", help="Sample saved videos and create unverified person proposals")
    sample.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sample.add_argument("--cameras", nargs="+")
    sample.add_argument("--out", type=Path, required=True)
    sample.add_argument("--interval", type=float, default=30)
    sample.add_argument("--start", type=float, default=0)
    sample.add_argument("--start-clock")
    sample.add_argument("--duration", type=float)
    sample.add_argument("--group-seconds", type=int, default=600)
    sample.add_argument("--guard-seconds", type=int, default=60)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--device")
    sample.add_argument("--imgsz", type=int)
    sample.add_argument("--batch-size", type=int, default=4)
    sample.add_argument("--no-proposals", action="store_true")
    sample.add_argument("--decode-mode", choices=("sequential", "seek"), default="sequential",
                        help="Files: sequential (default) selects by video timestamps without seeking; seek is an optional fast start")
    review = sub.add_parser("review", help="Review/add/remove candidate boxes in a local browser")
    review.add_argument("--dataset", type=Path, required=True)
    review.add_argument("--port", type=int, default=8765)
    dataset_status = sub.add_parser("dataset-status", help="Show actual label review progress")
    dataset_status.add_argument("--dataset", type=Path, required=True)
    export = sub.add_parser("export", help="Export only reviewed YOLO person labels, without training")
    export.add_argument("--dataset", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    hard_review = sub.add_parser("review-hard-negatives", help="Prepare full-frame hard-negative review candidates")
    hard_review.add_argument("--base", type=Path, required=True,
                             help="Completed, fixed-split reviewed annotations")
    hard_review.add_argument("--base-export", type=Path,
                             help="Matching exported dataset for the locked review baseline")
    hard_review.add_argument("--inference", type=Path, required=True,
                             help="Completed inference run containing per-camera frames.jsonl files")
    hard_review.add_argument("--out", type=Path, required=True,
                             help="New review dataset directory")
    hard_review.add_argument("--interval-seconds", type=int, default=10)
    hard_review.add_argument("--empty-interval-seconds", type=int, default=60)
    hard_review.add_argument("--min-confidence", type=float, default=0.2)
    hard_complete = sub.add_parser("finalize-hard-negatives", help="Audit reviewed hard negatives and export a new dataset")
    hard_complete.add_argument("--dataset", type=Path, required=True)
    hard_complete.add_argument("--audit-out", type=Path, required=True)
    hard_complete.add_argument("--export-out", type=Path, required=True)
    organize_hard = sub.add_parser("organize-hard-samples",
                                   help="Register hard samples and summarize grouped development validation")
    organize_hard.add_argument("--out", type=Path, required=True)
    organize_hard.add_argument("--v12-export", type=Path)
    organize_hard.add_argument("--v13-review", type=Path, nargs="*")
    organize_hard.add_argument("--holdout", type=Path)
    organize_hard.add_argument("--metrics", type=Path)
    merge_hard = sub.add_parser("merge-hard-training",
                                help="Merge reviewed D03/D04 additions into the v12 training snapshot")
    merge_hard.add_argument("--base", type=Path, required=True)
    merge_hard.add_argument("--supplement", type=Path, nargs="+", required=True)
    merge_hard.add_argument("--out", type=Path, required=True)
    merge_hard.add_argument("--initialization", type=Path, required=True)
    corrections = sub.add_parser("finalize-corrections", help="Audit completed label corrections and freeze historical splits")
    corrections.add_argument("--dataset", type=Path, required=True)
    corrections.add_argument("--snapshot", type=Path, required=True)
    corrections.add_argument("--audit-out", type=Path, required=True)
    corrections.add_argument("--export-out", type=Path, required=True)
    corrections.add_argument("--initialization", type=Path, help="Clean public checkpoint to require for the new training run")
    occlusion = sub.add_parser("occlusion", help="Independent person occlusion sampling / review / Recall evaluation", add_help=False)
    occlusion.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "inspect":
        from .video import inspect_source
        cfg = load_config(args.config, args.cameras)
        rows = [{"camera_id": c["id"], "name": c["name"], "mount": c["mount"],
                 "source": c["source"], **inspect_source(c["source"])} for c in cfg["cameras"]]
        write_json(args.out, rows)
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    elif args.command == "track":
        from .tracking import run_tracking
        run_tracking(args)
    elif args.command == "doctor":
        from .doctor import run_doctor
        run_doctor(args)
    elif args.command == "sample":
        from .dataset import run_sampling
        run_sampling(args)
    elif args.command == "review":
        from .review import run_review
        run_review(args)
    elif args.command == "dataset-status":
        from .review import ReviewStore
        print(json.dumps(ReviewStore(args.dataset).status(), ensure_ascii=False, indent=2))
    elif args.command == "export":
        from .review import export_dataset
        print(json.dumps(export_dataset(args.dataset, args.out), ensure_ascii=False, indent=2))
    elif args.command == "review-hard-negatives":
        from .experiments.prepare_person_hard_negative_review import prepare_review
        if args.interval_seconds <= 0 or args.empty_interval_seconds <= 0:
            parser.error("review sampling intervals must be positive")
        if not 0 <= args.min_confidence <= 1:
            parser.error("--min-confidence must be between 0 and 1")
        prepare_review(args.base, args.inference, args.out, args.interval_seconds,
                       args.empty_interval_seconds, args.min_confidence, args.base_export)
    elif args.command == "finalize-hard-negatives":
        from .experiments.complete_person_v5 import complete_dataset
        complete_dataset(args.dataset, args.audit_out, args.export_out)
    elif args.command == "organize-hard-samples":
        from .experiments.reorganize_hard_samples import organize, DEFAULT_HOLDOUT, DEFAULT_METRICS, DEFAULT_V12_EXPORT, DEFAULT_V13_D03, DEFAULT_V13_D04
        args.v12_export = args.v12_export or DEFAULT_V12_EXPORT
        args.v13_review = args.v13_review if args.v13_review is not None else [DEFAULT_V13_D03, DEFAULT_V13_D04]
        args.holdout = args.holdout or DEFAULT_HOLDOUT
        args.metrics = args.metrics or DEFAULT_METRICS
        organize(args)
    elif args.command == "merge-hard-training":
        from .experiments.merge_hard_training import merge
        merge(args.base, args.supplement, args.out, args.initialization)
    elif args.command == "finalize-corrections":
        from .experiments.complete_person_correction import complete_correction
        result = complete_correction(args.dataset, args.snapshot, args.audit_out, args.export_out, args.initialization)
        print(json.dumps({key: result[key] for key in (
            "status", "exported_dataset", "inherited_confirmed_empty_images", "excluded_images", "by_split")},
            ensure_ascii=False, indent=2))
    elif args.command == "occlusion":
        from .occlusion_benchmark import main as run_occlusion
        run_occlusion(args.arguments)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}") from None
