"""Collect the local experiment's aggregate evidence without frames or scene text."""
from pathlib import Path
import json
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cashwatch.data import file_hash, write_json
from cashwatch.evaluation import binary_metrics


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    manifest = read("artifacts/data/manifest.json")
    comparison = read("artifacts/expanded/cycle/comparison.json")
    benchmark = read("artifacts/expanded/cycle/benchmark.json")
    baseline_root = ROOT / "artifacts/baselines/cycle_7videos"
    integrity = read("artifacts/baselines/cycle_7videos/integrity.json")
    unchanged = all(file_hash(baseline_root / name) == expected for name, expected in integrity["files"].items())
    if not unchanged:
        raise ValueError("Preserved baseline changed")
    runs, sampling_hashes, source_hashes = [], set(), set()
    for head in ("linear", "temporal"):
        for seed in (42, 43, 44):
            directory = ROOT / "artifacts/expanded/cycle/runs" / f"{head}_seed{seed}"
            report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
            sampling_hashes.add(file_hash(directory / "sampling.json"))
            source_hashes.add(file_hash(directory / "source_hashes.json"))
            runs.append({"head": head, "seed": seed,
                **{key: report[key] for key in ["trainable_parameters", "frozen_parameters", "training_steps",
                    "selected_epoch", "selected_checkpoint_steps", "initial_train_loss", "selected_train_loss",
                    "weight_changes", "encoder_sha256_before", "encoder_sha256_after", "model_state_sha256",
                    "training_total_sec", "peak_vram_allocated_mb"]},
                "postprocess": report["config"]["postprocess"],
                "dev": {key: report["dev"][key] for key in ["video", "events"]}})
    training = [r["label"] for r in manifest["records"] if r["split"] == "train"]
    development = [r["label"] for r in manifest["records"] if r["split"] == "dev"]
    majority = int(sum(training) > len(training)/2)
    bundle = ROOT / "artifacts/handoff_expanded"
    test_log = (ROOT / "artifacts/expanded/unit_tests.txt").read_text(encoding="utf-8")
    test_match = re.search(r"Ran (\d+) tests", test_log)
    if not test_match or not test_log.rstrip().endswith("OK"):
        raise ValueError("Final verification suite did not pass")
    zip_path = ROOT / "artifacts/handoff_expanded.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(bundle.rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
                if file.suffix.lower() in {".mov", ".mp4", ".xlsx", ".npz"} or file.name in {"train.py", "data.py"}:
                    raise ValueError("Training data/code must not enter the runtime package")
                archive.write(file, file.relative_to(bundle))
    result = {"dataset": {"videos": len(manifest["records"]),
        "positive": sum(r["label"] for r in manifest["records"]),
        **{split: sum(r["split"] == split for r in manifest["records"]) for split in ["train", "dev", "test"]},
        "missing_files": len(manifest["missing_ids"]), "group_review": manifest["group_review"],
        "manifest_sha256": file_hash(ROOT / "artifacts/data/manifest.json")},
        "baseline_archive_integrity_passed": unchanged,
        "unit_tests": {"passed": int(test_match.group(1)), "failed": 0,
                       "log": "artifacts/expanded/unit_tests.txt"},
        "final_guards": read("artifacts/expanded/final_guards.json"),
        "baseline_archive": "artifacts/baselines/cycle_7videos",
        "source_and_sampling_identical_across_six_runs": len(source_hashes) == len(sampling_hashes) == 1,
        "training_runs": runs, "development_comparison": comparison,
        "constant_majority_development_baseline": binary_metrics(development, [majority]*len(development)),
        "test_quality_evaluated": False, "test_metrics": None,
        "test_inference_for_timing_only": benchmark["test_video_used_for_timing_only"],
        "test_timing_predictions_logged": False, "test_timing_used_for_selection": False,
        "longest_video_benchmark": benchmark,
        "runtime_offline_check": read("artifacts/expanded/offline_runtime/checks.json"),
        "os_network_block_check": read("artifacts/integration_checks/os_offline.json"),
        "http_check": read("artifacts/integration_checks/http_status.json"),
        "experimental_checkpoint": "artifacts/expanded/cycle/selected.pt",
        "default_project_checkpoint": "artifacts/cycle/selected.pt",
        "default_baseline_unchanged": file_hash(ROOT / "artifacts/cycle/selected.pt") == file_hash(baseline_root / "cycle/selected.pt"),
        "bundle": {"zip": str(zip_path.relative_to(ROOT)), "sha256": file_hash(zip_path)},
        "limitations": ["Only two dev videos; their metrics are after selection and threshold tuning",
            "Improved dev event F1 at tolerance 5 seconds does not establish accurate localization",
            "Temporal localization error 4.5-5 seconds is worse than frozen baseline's 4 seconds",
            "Point annotations cannot distinguish false events from unannotated events without human review",
            "Related-record provenance pending; test-quality evaluation deliberately deferred",
            "Runtime verification covers schema and PTS correspondence, not semantic result correctness",
            "No event taxonomy or verified regulatory catalogue mapping",
            "Real backend HTTP timing and OS-level network denial not executed"],
        "post_training_changes": "Input/training behavior unchanged; added stricter guards for future label revisions, test evaluation and checkpoint locking. Every training run contains the exact source used."}
    write_json(ROOT / "artifacts/expanded/summary.json", result)
    print(json.dumps({"saved": "artifacts/expanded/summary.json", "training_runs": len(runs),
                      "archive_integrity": unchanged, "bundle_sha256": result["bundle"]["sha256"]}))


if __name__ == "__main__":
    main()
