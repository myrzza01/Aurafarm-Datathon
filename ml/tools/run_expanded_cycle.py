"""Predeclared multi-seed comparison. Test evaluation is a separate, gated phase."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cashwatch.data import file_hash, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def spread(values):
    present = [float(value) for value in values if value is not None]
    return {"values": values, "defined_runs": len(present),
            "mean": statistics.mean(present) if present else None,
            "std_population": statistics.pstdev(present) if present else None,
            "min": min(present) if present else None, "max": max(present) if present else None}


def aggregate(reports):
    return {scope: {key: spread([report["dev"][scope][key] for report in reports]) for key in keys}
            for scope, keys in {"video": ["tp", "fp", "fn", "tn", "precision", "recall", "f1"],
                "events": ["matched", "missed", "extra", "precision", "recall", "f1", "mae_sec"]}.items()}


def rank(aggregated):
    e, v = aggregated["events"], aggregated["video"]
    mae = e["mae_sec"]["mean"]
    return (e["f1"]["mean"], v["f1"]["mean"], -e["missed"]["mean"],
            -e["extra"]["mean"], -(mae if mae is not None else float("inf")))


def save_locked_selection(output, chosen, config, selection):
    """Validate every existing lock before changing any selected artifact."""
    lock_path = output / "selection_lock.json"
    selected = output / "selected.pt"
    if lock_path.exists():
        old = read(lock_path)
        for key in ["selected_model", "selected_sha256", "plan_sha256", "checkpoints"]:
            if old[key] != selection[key]:
                raise ValueError("Cannot change frozen selection; start another experiment")
        if not selected.is_file() or file_hash(selected) != old["selected_sha256"]:
            raise ValueError("Selected checkpoint was changed after locking")
        if read(output / "selected_config.json") != config:
            raise ValueError("Selected configuration was changed after locking")
        return
    if selected.exists() or (output / "selected_config.json").exists():
        raise FileExistsError("Uncommitted selected artifacts exist; use a fresh experiment")
    shutil.copyfile(chosen, selected)
    write_json(output / "selected_config.json", config)
    write_json(lock_path, selection)


def verify_locked_weights(output, baseline):
    lock = read(output / "selection_lock.json")
    if file_hash(output / "selected.pt") != lock["selected_sha256"]:
        raise ValueError("Selected model changed after locking")
    if file_hash(output / "plan.json") != lock["plan_sha256"]:
        raise ValueError("Predeclared protocol changed after locking")
    if file_hash(baseline) != read(output / "plan.json")["baseline_sha256"]:
        raise ValueError("Frozen baseline changed")
    for name, expected in lock["checkpoints"].items():
        if file_hash(output / "runs" / name / "best.pt") != expected:
            raise ValueError("Candidate changed after selection locking")
    return lock


def lock_plan(output, manifest_path, baseline):
    manifest = read(manifest_path)
    plan = {"manifest_sha256": file_hash(manifest_path), "baseline_sha256": file_hash(baseline),
        "seeds": [42, 43, 44], "representative_seed": 42,
        "heads": ["linear", "temporal"], "epochs": 20, "samples_per_epoch": 80, "batch_size": 8,
        "selection": "mean dev event F1, mean video F1, fewer missed, fewer extra, lower matched MAE; linear wins ties",
        "replacement": "Candidate aggregate and fixed seed 42 must each strictly outrank the frozen baseline on dev",
        "test_used_for_selection": False, "split_seed": manifest["seed"],
        "test_scoring_requires_confirmed_group_review": True,
        "source_hashes": {str(p.relative_to(ROOT)): file_hash(p)
                          for p in sorted((ROOT / "cashwatch").glob("*.py"))},
        "memberships": {split: [r["id"] for r in manifest["records"] if r["split"] == split]
                        for split in ["train", "dev", "test"]}}
    target = output / "plan.json"
    if target.exists():
        old = read(target)
        old.pop("created_utc", None)
        if old != plan:
            raise ValueError("Experiment inputs changed; choose a new output directory")
    else:
        write_json(target, {**plan, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        shutil.copyfile(manifest_path, output / "manifest.json")
    return plan


def training_phase(args):
    from cashwatch.train import train, evaluate_checkpoint
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    plan = lock_plan(output, args.manifest, args.baseline)
    reports, sampling = {}, None
    for head in plan["heads"]:
        reports[head] = []
        for seed in plan["seeds"]:
            run = output / "runs" / f"{head}_seed{seed}"
            if not (run / "report.json").exists():
                run.mkdir(parents=True, exist_ok=True)
                with (run / "console.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                    train(args.manifest, run, plan["epochs"], plan["batch_size"], args.device,
                          plan["samples_per_epoch"], head, args.encoder, seed)
            report = read(run / "report.json")
            if report["config"]["seed"] != seed or read(run / "manifest.json") != read(args.manifest):
                raise ValueError("Existing run does not match the locked protocol")
            current = read(run / "sampling.json")
            if sampling is not None and sampling != current:
                raise ValueError("Heads/seeds received different input frames or windows")
            sampling = current
            reports[head].append(report)
            print(json.dumps({"head": head, "seed": seed, "epoch": report["selected_epoch"],
                "dev_video": report["dev"]["video"], "dev_events": report["dev"]["events"]}), flush=True)
    baseline_path = output / "frozen_baseline_dev.json"
    baseline = evaluate_checkpoint(args.baseline, args.manifest, "dev", baseline_path, args.device)
    aggregates = {head: aggregate(runs) for head, runs in reports.items()}
    frozen = aggregate([{"dev": baseline["metrics"]}])
    candidate = "temporal" if rank(aggregates["temporal"]) > rank(aggregates["linear"]) else "linear"
    representative = aggregate([reports[candidate][0]])
    replace = rank(aggregates[candidate]) > rank(frozen) and rank(representative) > rank(frozen)
    chosen = output / "runs" / f"{candidate}_seed42" / "best.pt" if replace else args.baseline
    if replace:
        config = reports[candidate][0]["config"]
    else:
        import torch
        config = torch.load(chosen, map_location="cpu", weights_only=True)["config"]
    selected_model = candidate if replace else "frozen_7_video_linear"
    selection = {"selected_model": selected_model, "candidate": candidate, "replaced_baseline": replace,
        "source_checkpoint": str(chosen.resolve()), "selected_sha256": file_hash(chosen),
        "locked_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "representative_seed": 42,
        "test_used": False, "plan_sha256": file_hash(output / "plan.json"),
        "checkpoints": {f"{head}_seed{seed}": file_hash(output / "runs" / f"{head}_seed{seed}" / "best.pt")
                        for head in plan["heads"] for seed in plan["seeds"]}}
    save_locked_selection(output, chosen, config, selection)
    manifest = read(args.manifest)
    write_json(output / "comparison.json", {"dataset": {split: sum(r["split"] == split for r in manifest["records"])
        for split in ["train", "dev", "test"]}, "same_input_frames_verified": True,
        "seeds": plan["seeds"], "frozen_baseline": frozen, "models": aggregates,
        "selection": read(output / "selection_lock.json"), "test_metrics": None,
        "test_status": "reserved_unscored_pending_source_group_review",
        "limitations": ["Only two development videos, used for selection and thresholds",
            "Point annotations do not establish complete event boundaries",
            "Between-seed spread is not a confidence interval for new-video performance"]})
    print(json.dumps({"selected_model": selected_model, "baseline_replaced": replace,
                      "same_input_frames": True, "test_scored": False}), flush=True)


def benchmark_phase(args):
    import torch
    from cashwatch.inference import VideoAnalyzer
    output = args.output
    lock = verify_locked_weights(output, args.baseline)
    manifest = read(args.manifest)
    # Timing only, after selection is locked. Never inspect/save held-out scores,
    # event counts or labels, and never feed timings back into model selection.
    longest = max(manifest["records"], key=lambda r: r["duration_sec"])
    if file_hash(longest["path"]) != longest["sha256"]:
        raise ValueError("Benchmark video changed after preparation")
    checkpoints = {"frozen_baseline": args.baseline,
        **{head: output / "runs" / f"{head}_seed42" / "best.pt" for head in ["linear", "temporal"]}}
    results = {}
    for name, checkpoint in checkpoints.items():
        loaded = time.perf_counter()
        analyzer = VideoAnalyzer(checkpoint, args.device)
        load_seconds = time.perf_counter() - loaded
        trials = []
        for index in range(3):
            result, diagnostics = analyzer.analyze(longest["path"])
            # Images remain transient local values. Never serialize them into benchmark logs.
            del result
            trials.append({key: diagnostics[key] for key in ["elapsed_sec", "decode_sec", "model_sec",
                "evidence_sec", "peak_vram_allocated_mb", "peak_vram_reserved_mb"]})
        results[name] = {"load_sec": load_seconds, "first_request": trials[0],
            "warm_requests": trials[1:], "timing_spread": {key: spread([trial[key] for trial in trials])
                for key in trials[0]}, "checkpoint_sha256": file_hash(checkpoint)}
        del analyzer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(json.dumps({"benchmark": name, "first_request_sec": trials[0]["elapsed_sec"],
                          "warm_mean_sec": statistics.mean(t["elapsed_sec"] for t in trials[1:])}), flush=True)
    write_json(output / "benchmark.json", {"video_id": longest["id"], "duration_sec": longest["duration_sec"],
        "is_longest_available_labelled_video": True,
        "test_video_used_for_timing_only": longest["split"] == "test",
        "test_quality_metrics_computed": False, "predictions_logged": False,
        "timings_used_for_model_selection": False, "models": results,
        "selection_sha256": lock["selected_sha256"], "full_http_request_sec": None,
        "cold_process_start_sec": None, "note": "In-process model load and local inference only; no HTTP backend supplied."})


def test_phase(args):
    from cashwatch.train import evaluate_checkpoint
    manifest = read(args.manifest)
    if manifest.get("group_review", {}).get("status") != "confirmed":
        raise ValueError("Source-group review is pending. Keep reserved test unscored until provenance is confirmed")
    if (manifest.get("label_revisions_pending_review")
            or manifest.get("test_reservation", {}).get("qualification_status") == "invalidated_by_label_revision"):
        raise ValueError("Annotation revisions invalidated the test protocol")
    output = args.output
    lock = verify_locked_weights(output, args.baseline)
    original = read(output / "manifest.json")
    old_records = {r["id"]: r for r in original["records"]}
    if set(old_records) != {r["id"] for r in manifest["records"]}:
        raise ValueError("Dataset membership changed after training; start a reviewed experiment")
    for record in manifest["records"]:
        before = old_records[record["id"]]
        if any(record[key] != before[key] for key in ["sha256", "split", "label", "event_times"]):
            raise ValueError("Video, split or annotations changed after training")
        if file_hash(record["path"]) != record["sha256"]:
            raise ValueError("Source video changed after preparation")
    for left, right in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        if ({r["group"] for r in manifest["records"] if r["split"] == left}
                & {r["group"] for r in manifest["records"] if r["split"] == right}):
            raise ValueError("Reviewed source groups reveal cross-split leakage; test invalid")
    destination = output / "test_evaluation"
    if destination.exists():
        raise FileExistsError("Test already evaluated; do not use repeated test results for tuning")
    targets = {"frozen_baseline": args.baseline,
        **{name: output / "runs" / name / "best.pt" for name in lock["checkpoints"]}}
    destination.mkdir()
    for name, checkpoint in targets.items():
        evaluate_checkpoint(checkpoint, args.manifest, "test", destination / f"{name}.json", args.device)
    write_json(destination / "status.json", {"test_used_for_selection": False,
        "selection_lock_sha256": file_hash(output / "selection_lock.json"),
        "warning": "Any subsequent tuning informed by these results requires a new independent test."})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["train", "benchmark", "evaluate-test"], required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/data/manifest.json")
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts/baselines/cycle_7videos/cycle/selected.pt")
    parser.add_argument("--encoder", type=Path, default=ROOT / "artifacts/pretrained/encoder.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/expanded/cycle")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    {"train": training_phase, "benchmark": benchmark_phase, "evaluate-test": test_phase}[args.phase](args)


if __name__ == "__main__":
    main()
