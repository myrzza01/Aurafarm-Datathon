"""Train binary localization and a human-labelled visual action classifier.

Only train/dev pixels are read before model selection. The final, frozen model is
evaluated once on the reserved diagnostic partition. Related recordings have not
been source-owner reviewed, so that partition is NOT called independent quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .data import file_hash, write_json
from .evaluation import calibrate, metrics_from_scores, detect_intervals
from .typed_evaluation import typed_event_metrics
from .model import VideoDetector, MODEL_NAME, select_device
from .train import set_seed, tensor_hash, load_caches, encode_caches, supervised_tensors, feature_timelines, balanced_loss
from .video import clip_indices


def native_records(records):
    output = []
    for original in records:
        record = original.copy()
        for key in ("path", "cache"):
            value = record.get(key + "_windows", record[key]) if os.name == "nt" else record[key]
            if os.name != "nt" and len(value) > 2 and value[1] == ":":
                value = "/mnt/" + value[0].lower() + "/" + value[3:].replace("\\", "/")
            record[key] = value
        output.append(record)
    return output


def action_tensors(records, encoded, config, offsets=(-1.0, -0.5, 0.0, 0.5, 1.0)):
    classes = config["event_taxonomy"]["type_ids"]
    clips, labels, sources = [], [], []
    # Each duplicate group has one weight regardless of how many source aliases.
    counts = Counter(e["type_id"] for r in records for e in r["events"] if e.get("type_verified"))
    weights = []
    for record in records:
        features, times = encoded[record["id"]]
        for event in record["events"]:
            if not event.get("type_verified"):
                continue
            for offset in offsets:
                indices = clip_indices(times, event["timecode_sec"] + offset, config["clip_frames"], config["fps"])
                clips.append(features[indices])
                labels.append(classes.index(event["type_id"]))
                sources.append({"video_id": record["id"], "timecode_sec": event["timecode_sec"], "type_id": event["type_id"]})
                weights.append(1.0 / (counts[event["type_id"]] * len(offsets)))
    if not clips:
        return None, None, None, []
    return torch.stack(clips), torch.tensor(labels), torch.tensor(weights, dtype=torch.double), sources


def action_metrics(model, records, encoded, config, device):
    clips, labels, _, sources = action_tensors(records, encoded, config, offsets=(0.0,))
    classes = config["event_taxonomy"]["type_ids"]
    if clips is None:
        return {"events": 0, "accuracy": None, "macro_f1_supported": None, "per_class": {}, "details": []}
    model.eval()
    with torch.inference_mode():
        logits = torch.cat([model.classify_actions(clips[i:i+32].to(device)).cpu() for i in range(0, len(clips), 32)])
    predicted = logits.argmax(1)
    per_class = {}
    for index, name in enumerate(classes):
        tp = int(((labels == index) & (predicted == index)).sum())
        fp = int(((labels != index) & (predicted == index)).sum())
        fn = int(((labels == index) & (predicted != index)).sum())
        support = int((labels == index).sum())
        per_class[name] = {"support": support, "correct": tp, "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
    return {"events": len(labels), "accuracy": float((predicted == labels).float().mean()),
        "macro_f1_supported": float(np.mean([v["f1"] for v in per_class.values() if v["support"]])),
        "cross_entropy": float(nn.functional.cross_entropy(logits, labels)), "per_class": per_class,
        "details": [{**source, "predicted_type_id": classes[int(predicted[i])],
            "confidence": float(logits[i].softmax(0).max())} for i, source in enumerate(sources)],
        "scope": "Classification at annotated event centers; excludes detector misses and false alerts."}


def verified_caches(records):
    for record in records:
        if file_hash(record["path"]) != record["sha256"]:
            raise ValueError("Video bytes changed after import")
        if record.get("cache_sha256") and file_hash(record["cache"]) != record["cache_sha256"]:
            raise ValueError("Cached pixels changed after import")
    return load_caches(records)


def end_to_end_actions(model, records, encoded, timelines, config, device):
    predictions = {}
    model.eval()
    with torch.inference_mode():
        for record in records:
            features, times = encoded[record["id"]]
            events = detect_intervals(*timelines[record["id"]], **config["postprocess"])
            for event in events:
                indices = clip_indices(times, event["timecode_sec"], config["clip_frames"], config["fps"])
                scores = model.classify_actions(features[indices].unsqueeze(0).to(device)).softmax(-1)[0]
                event["type_id"] = config["event_taxonomy"]["type_ids"][int(scores.argmax())]
            predictions[record["id"]] = events
    return typed_event_metrics(records, predictions, config["event_tolerance_sec"])


def train_reviewed(manifest_path, output, encoder_weights, epochs=60, device_name="auto", head_type="temporal", seed=42, refit_all=False):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    manifest_path, output = Path(manifest_path), Path(output)
    if (output / "best.pt").exists():
        raise FileExistsError("Choose a new experiment directory; existing results are preserved")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = native_records(manifest["records"])
    partitions = {split: [r for r in records if r["split"] == split] for split in ("train", "dev", "test")}
    for first, second in (("train", "dev"), ("train", "test"), ("dev", "test")):
        for key in ("id", "group", "sha256"):
            if {r[key] for r in partitions[first]} & {r[key] for r in partitions[second]}:
                raise ValueError(f"Partition leakage: {key}")
    for split in ("train", "dev"):
        if {r["label"] for r in partitions[split]} != {0, 1}:
            raise ValueError("Train and dev require both binary classes")
    taxonomy = manifest["event_taxonomy"]
    if taxonomy["status"] != "human_action_notes_confirmed":
        raise ValueError("Action annotations require explicit human confirmation")
    trained_types = {e["type_id"] for r in partitions["train"] for e in r["events"] if e.get("type_verified")}
    if trained_types != set(taxonomy["type_ids"]):
        raise ValueError("Every deployed action class must have a train example")
    if refit_all:
        partitions = {"train": records, "dev": [], "test": []}
    config = {key: manifest[key] for key in ("fps", "image_size", "clip_frames", "positive_radius_sec")}
    config.update(seed=seed, head_type=head_type, model_name=MODEL_NAME, event_taxonomy=taxonomy,
        stage="frozen_pretrained_encoder_joint_binary_action_heads", epochs=epochs,
        training_scope="all_available_refit" if refit_all else "train_partition_only",
        event_tolerance_sec=5.0, action_confidence_calibrated=False,
        postprocess={"threshold": 0.5, "merge_gap_sec": 1.0, "min_hits": 1})
    set_seed(seed)
    device = select_device(device_name)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", manifest)
    write_json(output / "environment.json", {"device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "timm", "numpy", "av")}})
    write_json(output / "source_hashes.json", {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")})
    source = torch.load(encoder_weights, map_location="cpu", weights_only=True)
    model = VideoDetector(head_type=head_type, action_classes=len(taxonomy["type_ids"])).to(device)
    model.encoder.load_state_dict(source["encoder"])
    model.configure_finetuning(False)
    encoder_hash = tensor_hash(model.encoder.state_dict())
    initial_head = {k: p.detach().cpu().clone() for k, p in model.named_parameters() if p.requires_grad}
    selection_records = partitions["train"] + partitions["dev"]
    started = time.monotonic()
    # Source pixels, not filenames, spreadsheet notes or identifiers, are features.
    encoded = encode_caches(model, selection_records, verified_caches(selection_records), device)
    binary_x, binary_y, binary_w = supervised_tensors(partitions["train"], encoded, config)
    if not refit_all:
        dev_x, dev_y, _ = supervised_tensors(partitions["dev"], encoded, config)
    action_x, action_y, action_w, _ = action_tensors(partitions["train"], encoded, config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0003, weight_decay=0.03)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    generator = torch.Generator().manual_seed(seed)
    best_loss, steps = float("inf"), 0
    print(json.dumps({"stage": "joint_training", "device": str(device), "partitions": {k: len(v) for k, v in partitions.items()},
        "action_classes": len(taxonomy["type_ids"])}), flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        bi = torch.multinomial(binary_w, 160, replacement=True, generator=generator)
        ai = torch.multinomial(action_w, 160, replacement=True, generator=generator)
        running = 0.0
        for start in range(0, 160, 16):
            b, a = bi[start:start+16], ai[start:start+16]
            optimizer.zero_grad(set_to_none=True)
            binary_loss = nn.functional.binary_cross_entropy_with_logits(model.classify(binary_x[b].to(device)), binary_y[b].to(device))
            action_loss = nn.functional.cross_entropy(model.classify_actions(action_x[a].to(device)), action_y[a].to(device), label_smoothing=0.05)
            loss = binary_loss + 0.7 * action_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            steps += 1
            running += float(loss.detach()) / 10
        scheduler.step()
        dev_binary = None if refit_all else balanced_loss(model, dev_x, dev_y, device)
        dev_action = {"accuracy": None} if refit_all else action_metrics(model, partitions["dev"], encoded, config, device)
        # Predeclared objective: equal importance to binary discrimination and
        # semantic CE (scaled by log(class count)); no test-informed selection.
        selection_loss = -epoch if refit_all else dev_binary + 0.7 * dev_action["cross_entropy"] / np.log(len(taxonomy["type_ids"]))
        log = {"epoch": epoch, "loss": running, "dev_binary_loss": dev_binary,
            "dev_action_accuracy": dev_action["accuracy"], "selection_loss": selection_loss,
            "elapsed_sec": round(time.monotonic() - started, 2)}
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(log) + "\n")
        if selection_loss < best_loss:
            best_loss = selection_loss
            checkpoint = {"format_version": 3, "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "config": config.copy(), "epoch": epoch, "training_steps": steps,
                "manifest_sha256": file_hash(manifest_path), "encoder_source_sha256": file_hash(encoder_weights),
                "selection_loss": float(selection_loss), "calibrated_on": "fixed_defaults" if refit_all else "development_only"}
            torch.save(checkpoint, output / "best.pt")
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(log), flush=True)
    saved = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"])
    if tensor_hash(model.encoder.state_dict()) != encoder_hash:
        raise RuntimeError("Frozen encoder unexpectedly changed")
    changes = {name: float((p.detach().cpu() - initial_head[name]).abs().sum()) for name, p in model.named_parameters() if name in initial_head}
    if not sum(value for name, value in changes.items() if name.startswith("action_head")) > 0:
        raise RuntimeError("Action weights were not trained")
    timelines = feature_timelines(model, selection_records, encoded, config, device)
    if refit_all:
        postprocess = {"threshold": 0.5, "merge_gap_sec": 1.0, "min_hits": 1,
            "confirmation": "adjacent_2", "confirmation_max_step_sec": 0.75}
        trials = []
    else:
        postprocess, _, trials = calibrate(partitions["dev"], timelines, 5.0)
    config["postprocess"] = postprocess
    saved.update(config=config, weight_changes=changes, encoder_hash=encoder_hash)
    torch.save(saved, output / "best.pt")
    frozen_hash = file_hash(output / "best.pt")
    write_json(output / "calibration.json", {"split": None if refit_all else "dev", "selected": postprocess, "trials": trials})
    report = {"checkpoint_sha256": frozen_hash, "selected_epoch": saved["epoch"], "training_steps": steps,
        "config": config, "encoder_frozen_verified": True, "head_weight_changes_l1": changes,
        "selection": f"Fixed {epochs}-epoch all-data demonstration refit, fixed threshold. All metrics are resubstitution, not validation."
            if refit_all else "Development binary loss plus normalized action CE; postprocessing development event F1. Test unused for selection.",
        "independent_quality_established": False,
        "limitations": ["Only 40 unique videos; most action classes have one example.",
            "Source sessions and related takes are unreviewed. Held-out results are diagnostic, not independent accuracy.",
            "Some binary labels/timestamps retain original provenance; user confirmed H/J action labels.",
            "Softmax confidence is uncalibrated. Class-specific explanation is a prediction requiring visual review.",
            "Generalization to a new camera angle has not been tested."], "partitions": {}}
    # Freeze checkpoint and ALL choices before reading reserved test pixels.
    if partitions["test"]:
        test_encoded = encode_caches(model, partitions["test"], verified_caches(partitions["test"]), device)
        encoded.update(test_encoded)
        timelines.update(feature_timelines(model, partitions["test"], test_encoded, config, device))
    for split, subset in partitions.items():
        if subset:
            report["partitions"][split] = {"videos": len(subset),
                "detection": metrics_from_scores(subset, timelines, postprocess, 5.0),
                "classification_at_truth": action_metrics(model, subset, encoded, config, device),
                "typed_events_end_to_end": end_to_end_actions(model, subset, encoded, timelines, config, device)}
    if file_hash(output / "best.pt") != frozen_hash:
        raise RuntimeError("Checkpoint changed during held-out evaluation")
    report["elapsed_sec"] = time.monotonic() - started
    write_json(output / "report.json", report)
    write_json(output / "config.json", config)
    print(json.dumps({"training_complete": True, "selected_epoch": saved["epoch"], "elapsed_sec": report["elapsed_sec"],
        "metrics": {s: {"video": r["detection"]["video"], "action_accuracy_at_truth": r["classification_at_truth"]["accuracy"]} for s, r in report["partitions"].items()}}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-weights", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--refit-all", action="store_true", help="Train on all videos; metrics then describe training fit only")
    args = parser.parse_args()
    train_reviewed(args.manifest, args.output, args.encoder_weights, args.epochs, args.device, refit_all=args.refit_all)
