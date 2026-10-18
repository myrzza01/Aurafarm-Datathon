"""CLI for preparing, training and running the local video detector."""
import argparse
import json
import sys
from pathlib import Path


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Train and run a local video violation detector")
    root = Path(__file__).resolve().parent
    packaged = root / "weights" / "detector.pt"
    default_checkpoint = packaged if packaged.is_file() else root / "artifacts" / "cycle" / "selected.pt"
    if (root / "weights" / "reviewed_detector.pt").is_file():
        default_checkpoint = root / "weights" / "reviewed_detector.pt"
    if (root / "weights" / "hard_negative_detector.pt").is_file():
        default_checkpoint = root / "weights" / "hard_negative_detector.pt"
    commands = parser.add_subparsers(dest="command", required=True)
    weights = commands.add_parser("init-weights", help="Download initial encoder once; all later commands use local weights")
    weights.add_argument("--output", type=Path, default=Path("artifacts/pretrained/encoder.pt"))
    prepare = commands.add_parser("prepare", help="Read labels, decode video locally, split by video")
    prepare.add_argument("--data-dir", required=True, type=Path)
    prepare.add_argument("--labels", type=Path)
    prepare.add_argument("--output", type=Path, default=Path("artifacts/data"))
    prepare.add_argument("--allow-missing", action="store_true")
    prepare.add_argument("--groups-csv", type=Path)
    prepare.add_argument("--confirm-group-review", action="store_true",
                         help="Explicit source-owner confirmation of relationships in --groups-csv")
    prepare.add_argument("--accept-label-revisions", action="store_true",
                         help="Accept reviewed dev/test label revisions, archive prior manifest and invalidate test qualification")
    prepare.add_argument("--seed", type=int, default=42)
    reviewed = commands.add_parser("prepare-reviewed", help="Import the merged human review workbook with explicit provenance")
    reviewed.add_argument("--workbook", required=True, type=Path)
    reviewed.add_argument("--videos", required=True, type=Path)
    reviewed.add_argument("--output", required=True, type=Path)
    reviewed.add_argument("--confirm-action-notes", action="store_true", help="H/J are final human action annotations")
    reviewed.add_argument("--allow-weak-original", action="store_true", help="Retain original binary labels with weak provenance")
    reviewed.add_argument("--seed", type=int, default=42)
    training = commands.add_parser("train", help="Train encoder and temporal classifier on video pixels")
    training.add_argument("--manifest", type=Path, default=Path("artifacts/data/manifest.json"))
    training.add_argument("--output", type=Path, default=Path("artifacts/cycle/temporal"))
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--head-type", choices=["temporal", "linear"], default="temporal")
    training.add_argument("--encoder-weights", type=Path, default=Path("artifacts/pretrained/encoder.pt"))
    training.add_argument("--samples-per-epoch", type=int, default=80)
    training.add_argument("--seed", type=int, help="Training seed only; does not change locked video splits")
    training.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    predict = commands.add_parser("predict", help="Analyze a new video using weights only")
    predict.add_argument("--video", required=True, type=Path)
    predict.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    predict.add_argument("--output", type=Path, default=Path("artifacts/prediction.json"))
    predict.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    predict.add_argument("--threshold", type=float)
    evaluation = commands.add_parser("evaluate", help="Evaluate a frozen checkpoint without tuning parameters")
    evaluation.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    evaluation.add_argument("--manifest", type=Path, default=Path("artifacts/data/manifest.json"))
    evaluation.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    evaluation.add_argument("--output", type=Path, default=Path("artifacts/cycle/evaluation.json"))
    evaluation.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    comparison = commands.add_parser("compare", help="Compare temporal model against the trained linear baseline")
    comparison.add_argument("--temporal", type=Path, default=Path("artifacts/cycle/temporal"))
    comparison.add_argument("--baseline", type=Path, default=Path("artifacts/cycle/linear"))
    comparison.add_argument("--output", type=Path, default=Path("artifacts/cycle/comparison.json"))
    args = parser.parse_args()
    try:
        if args.command == "init-weights":
            from cashwatch.train import init_weights
            init_weights(args.output)
        elif args.command == "prepare":
            from cashwatch.data import prepare as prepare_data
            prepare_data(args.data_dir, args.output, args.labels, args.allow_missing, args.groups_csv, args.seed,
                         group_review_confirmed=args.confirm_group_review,
                         accept_label_revisions=args.accept_label_revisions)
        elif args.command == "prepare-reviewed":
            from cashwatch.data import prepare_reviewed
            manifest = prepare_reviewed(args.workbook, args.videos, args.output, seed=args.seed,
                allow_weak_original=args.allow_weak_original,
                human_action_notes_confirmed=args.confirm_action_notes)
            print(json.dumps({"prepared_unique_videos": len(manifest["records"]), "manifest": str(args.output / "manifest.json")}))
        elif args.command == "train":
            from cashwatch.train import train
            report = train(args.manifest, args.output, args.epochs, args.batch_size, args.device,
                           args.samples_per_epoch, args.head_type, args.encoder_weights, args.seed)
            print(json.dumps({"training_complete": True, "selected_epoch": report["selected_epoch"],
                              "checkpoint": str(args.output / "best.pt")}, ensure_ascii=False))
        elif args.command == "evaluate":
            from cashwatch.train import evaluate_checkpoint
            report = evaluate_checkpoint(args.checkpoint, args.manifest, args.split, args.output, args.device)
            print(json.dumps({"split": args.split, "video": report["metrics"]["video"],
                              "events": report["metrics"]["events"]}))
        elif args.command == "compare":
            from cashwatch.train import compare_reports
            report = compare_reports(args.temporal, args.baseline, args.output)
            print(json.dumps({"temporal_video": report["temporal"]["video"],
                              "linear_video": report["linear_mean_pool_baseline"]["video"],
                              "constant_video": report["constant_majority_baseline"],
                              "selected_model": report["selected_model"]}))
        else:
            from cashwatch.inference import VideoAnalyzer
            analyzer = VideoAnalyzer(args.checkpoint, args.device, args.threshold)
            result, diagnostics = analyzer.analyze(args.video)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            args.output.with_suffix(".metrics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            # Do not print evidence images or scene content into logs/cloud chat.
            print(json.dumps({"saved": str(args.output), **{key: diagnostics[key] for key in
                ["elapsed_sec", "video_duration_sec", "events", "sampled_frames", "model_sec", "peak_vram_allocated_mb"]}}, ensure_ascii=False))
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
