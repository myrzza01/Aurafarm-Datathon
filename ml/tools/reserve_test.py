"""Reserve unseen groups before any fitting; old development videos stay fixed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cashwatch.data import file_hash, write_json


def choose_test_groups(records, historical_records, ambiguous_ids=(), per_class=2, seed=42):
    if any(r.get("split") == "test" for r in records):
        raise ValueError("A test partition already exists; it must not be reselected")
    used_ids = {r["id"] for r in historical_records}
    used_hashes = {r["sha256"] for r in historical_records}
    used_groups = {r["group"] for r in historical_records}
    ambiguous_ids = set(ambiguous_ids)
    groups = {}
    for record in records:
        groups.setdefault(record["group"], []).append(record)
    eligible = {0: [], 1: []}
    excluded = {}
    for group, members in groups.items():
        reasons = []
        if group in used_groups or any(r["id"] in used_ids or r["sha256"] in used_hashes for r in members):
            reasons.append("recorded_prior_manifest_group_id_or_bytes")
        if any(r["split"] != "train" for r in members):
            reasons.append("existing_nontraining_assignment")
        if any(r["id"] in ambiguous_ids for r in members):
            reasons.append("unresolved_possible_numeric_overlap")
        labels = {r["label"] for r in members}
        if len(labels) != 1:
            reasons.append("mixed_label_source_group_not_selected_by_balanced_sampler")
        if reasons:
            excluded[group] = reasons
        else:
            eligible[next(iter(labels))].append(group)
    rng = random.Random(seed)
    selected = []
    for label in (0, 1):
        candidates = sorted(eligible[label])
        if len(candidates) < per_class:
            raise ValueError(f"Insufficient unseen nonambiguous groups for label {label}: {len(candidates)} < {per_class}")
        rng.shuffle(candidates)
        selected.extend(candidates[:per_class])
    return sorted(selected), {"eligible_groups_by_class": {str(k): len(v) for k, v in eligible.items()},
                              "excluded_groups": excluded}


def reserve(manifest_path, baseline_manifest, historical_root, per_class=2, seed=42):
    manifest_path, baseline_manifest = Path(manifest_path).resolve(), Path(baseline_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["seed"] != seed:
        raise ValueError("Reservation seed must equal the locked manifest seed")
    provenance = {baseline_manifest}
    for path in Path(historical_root).resolve().rglob("manifest.json"):
        # Preparation alone does not mean use. Training manifests are copied
        # beside model config/report; the explicit baseline covers legacy runs.
        if path != manifest_path and ((path.parent / "report.json").exists() or (path.parent / "config.json").exists()):
            provenance.add(path)
    historical = []
    for path in sorted(provenance):
        old = json.loads(path.read_text(encoding="utf-8"))
        historical.extend(old["records"])
    selected, details = choose_test_groups(manifest["records"], historical,
                                           manifest.get("automatic_group_audit", {}).get("possible_related_ids", []),
                                           per_class, seed)
    old_assignments = {r["id"]: r["split"] for r in manifest["records"]}
    baseline = json.loads(baseline_manifest.read_text(encoding="utf-8"))
    current = {r["id"]: r for r in manifest["records"]}
    for previous in baseline["records"]:
        updated = current.get(previous["id"])
        if updated is None or updated["sha256"] != previous["sha256"]:
            raise ValueError("Baseline videos must remain present with unchanged bytes")
        previous_split = "dev" if previous["split"] == "val" else previous["split"]
        if updated["split"] != previous_split:
            raise ValueError("Baseline split membership changed")
    reservation = {"seed": seed, "per_class_groups": per_class, "selected_groups": selected,
                   "selected_ids": sorted(r["id"] for r in manifest["records"] if r["group"] in selected),
                   "historical_manifests": [{"path": str(p), "sha256": file_hash(p)} for p in sorted(provenance)],
                   "reserved_before_expanded_training": True,
                   "status": "provenance_confirmed" if manifest.get("manual_group_review_complete") else "provenance_pending",
                   "labels_sha256": manifest.get("labels_sha256"),
                   "independence_status": "unseen relative to recorded experiments; source-session independence awaits owner review",
                   "source_owner_review_complete": bool(manifest.get("manual_group_review_complete")), **details}
    registry_path = manifest_path.parent / "split_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    write_json(manifest_path.parent / "manifest_before_test_reservation.json", manifest)
    for record in manifest["records"]:
        if record["group"] in selected:
            record["split"] = "test"
    for record in registry["records"]:
        if record["group"] in selected:
            record["split"] = "test"
    if any(r["split"] != old_assignments[r["id"]] for r in manifest["records"] if r["group"] not in selected):
        raise AssertionError("Reservation changed unrelated assignments")
    manifest["test_reservation"] = registry["test_reservation"] = reservation
    write_json(manifest_path, manifest)
    write_json(registry_path, registry)
    write_json(manifest_path.parent / "test_reservation.json", reservation)
    counts = {split: {"videos": sum(r["split"] == split for r in manifest["records"]),
                      "positive": sum(r["split"] == split and r["label"] == 1 for r in manifest["records"]),
                      "negative": sum(r["split"] == split and r["label"] == 0 for r in manifest["records"])}
              for split in ("train", "dev", "test")}
    print(json.dumps({"split": counts, "source_owner_review_complete": reservation["source_owner_review_complete"]}))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="artifacts/data/manifest.json")
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--historical-root", default="artifacts")
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("--per-class must be positive")
    reserve(args.manifest, args.baseline_manifest, args.historical_root, args.per_class, args.seed)
