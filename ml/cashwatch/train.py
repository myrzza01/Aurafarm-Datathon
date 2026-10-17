"""Reproducible head-first training and development-only model selection."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

from .data import file_hash, write_json
from .evaluation import binary_metrics, calibrate, metrics_from_scores
from .inference import VideoAnalyzer
from .model import MODEL_NAME, VideoDetector, select_device
from .video import clip_indices


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def set_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_weights(output):
    """Only this explicit command may download pretrained weights."""
    output = Path(output)
    if output.exists():
        print(json.dumps({"encoder_weights_already_present": True, "sha256": file_hash(output)}))
        return
    model = VideoDetector(pretrained=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": MODEL_NAME, "encoder": model.encoder.state_dict()}, output)
    write_json(output.with_suffix(".json"), {"model_name": MODEL_NAME, "sha256": file_hash(output),
        "source": "https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k", "author_license": "Apache-2.0"})
    print(json.dumps({"saved_local_encoder": str(output), "sha256": file_hash(output)}))


def labeled_centers(record, times, radius=1.0):
    if record["label"] == 0:
        return np.linspace(times[0], times[-1], min(24, len(times))).tolist()
    return sorted({float(np.clip(t+offset, times[0], times[-1]))
                   for t in record["event_times"] for offset in np.linspace(-radius, radius, 5)})


def load_caches(records):
    result = {}
    for record in records:
        with np.load(record["cache"], allow_pickle=False) as saved:
            result[record["id"]] = (saved["images"], saved["times"])
    return result


def encode_caches(model, records, caches, device):
    encoded = {}
    model.eval()
    with torch.inference_mode():
        for record in records:
            images, times = caches[record["id"]]
            pieces = []
            for start in range(0, len(images), 32):
                frames = torch.from_numpy(images[start:start+32].copy()).permute(0, 3, 1, 2).to(device).float()/255
                pieces.append(model.encode(frames).cpu())
            # Clone outside inference_mode below; cached visual features are constants.
            encoded[record["id"]] = (torch.cat(pieces), times)
    return {key: (features.clone(), times) for key, (features, times) in encoded.items()}


def supervised_tensors(records, encoded, config):
    clips, targets, weights = [], [], []
    for record in records:
        features, times = encoded[record["id"]]
        centers = labeled_centers(record, times, config["positive_radius_sec"])
        same_class = sum(r["label"] == record["label"] for r in records)
        for center in centers:
            clips.append(features[clip_indices(times, center, config["clip_frames"], config["fps"])])
            targets.append(float(record["label"]))
            weights.append(1/(len(centers)*same_class))
    return torch.stack(clips), torch.tensor(targets), torch.tensor(weights, dtype=torch.double)


def feature_timelines(model, records, encoded, config, device):
    output = {}
    model.eval()
    with torch.inference_mode():
        for record in records:
            features, times = encoded[record["id"]]
            scores = []
            for start in range(0, len(times), 64):
                indices = np.stack([clip_indices(times, center, config["clip_frames"], config["fps"])
                                    for center in times[start:start+64]])
                scores.extend(model.classify(features[indices].to(device)).sigmoid().cpu().tolist())
            output[record["id"]] = (times, np.asarray(scores))
    return output


def balanced_loss(model, clips, targets, device):
    model.eval()
    with torch.inference_mode():
        outputs = torch.cat([model.classify(clips[i:i+32].to(device)).cpu() for i in range(0, len(clips), 32)])
        losses = nn.functional.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
    return float(torch.stack([losses[targets == value].mean() for value in [0, 1]]).mean())


def train(manifest_path, output, epochs=20, batch_size=8, device_name="auto", samples_per_epoch=80,
          head_type="temporal", encoder_weights="artifacts/pretrained/encoder.pt", seed=None):
    manifest_path, output = Path(manifest_path), Path(output)
    if (output / "best.pt").exists():
        raise FileExistsError("Preserve existing experiments: choose a new --output")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_records = [r for r in manifest["records"] if r["split"] == "train"]
    dev_records = [r for r in manifest["records"] if r["split"] in {"dev", "val"}]
    for subset in [train_records, dev_records]:
        if {r["label"] for r in subset} != {0, 1}:
            raise ValueError("Both train and development sets must contain both classes")
    subsets = [train_records, dev_records, [r for r in manifest["records"] if r["split"] == "test"]]
    for i, first in enumerate(subsets):
        for second in subsets[i+1:]:
            for key in ["id", "group", "sha256"]:
                if {r[key] for r in first} & {r[key] for r in second}:
                    raise ValueError(f"Data leakage in {key}")
    if epochs < 1 or batch_size < 1 or samples_per_epoch < 2:
        raise ValueError("Invalid training parameters")
    config = {key: manifest[key] for key in ["fps", "image_size", "clip_frames", "positive_radius_sec", "seed"]}
    config["split_seed"] = manifest["seed"]
    config["seed"] = manifest["seed"] if seed is None else int(seed)
    config.update({"head_type": head_type, "model_name": MODEL_NAME, "stage": "frozen_encoder",
                   "loss": "BCEWithLogitsLoss", "epochs": epochs, "batch_size": batch_size,
                   "samples_per_epoch": samples_per_epoch, "learning_rate": 0.0003,
                   "weight_decay": 0.01, "event_tolerance_sec": 5.0,
                   "event_taxonomy": {"status": "unlabelled", "type_ids": []},
                   "postprocess": {"threshold": 0.5, "merge_gap_sec": 1.0, "min_hits": 1}})
    set_seed(config["seed"])
    device = select_device(device_name)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", manifest)
    write_json(output / "config.json", config)
    versions = {package: importlib.metadata.version(package) for package in ["torch", "torchvision", "timm", "numpy", "av", "Pillow", "openpyxl"]}
    write_json(output / "environment.json", {"python": sys.version, "versions": versions, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic_algorithms": True, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]})
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (output / "requirements-lock.txt").write_text(frozen, encoding="utf-8")
    source_dir = Path(__file__).resolve().parent
    write_json(output / "source_hashes.json", {file.name: file_hash(file) for file in sorted(source_dir.glob("*.py"))})
    (output / "source" / "cashwatch").mkdir(parents=True, exist_ok=True)
    for file in source_dir.glob("*.py"):
        shutil.copyfile(file, output / "source" / "cashwatch" / file.name)
    shutil.copyfile(source_dir.parent / "main.py", output / "source" / "main.py")
    if not Path(encoder_weights).is_file():
        raise FileNotFoundError("Local encoder weights missing. Run main.py init-weights once before training")
    source = torch.load(encoder_weights, map_location="cpu", weights_only=True)
    if source["model_name"] != MODEL_NAME:
        raise ValueError("Unexpected encoder architecture")
    model = VideoDetector(pretrained=False, head_type=head_type).to(device)
    model.encoder.load_state_dict(source["encoder"])
    model.configure_finetuning(unfreeze_encoder=False)
    encoder_before = tensor_hash(model.encoder.state_dict())
    head_initial = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
    trainable = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    write_json(output / "layers.json", {"trainable": trainable, "trainable_count": sum(trainable.values()),
        "frozen": {name: p.numel() for name, p in model.named_parameters() if not p.requires_grad},
        "description": "All encoder parameters and BN statistics frozen; only normalization and classifier are trained."})
    records = train_records + dev_records  # Test is never encoded or read here.
    started = time.monotonic()
    encoded = encode_caches(model, records, load_caches(records), device)
    encode_seconds = time.monotonic()-started
    # Equality of source pixels and indices is independently auditable across heads/seeds.
    sampling = {}
    for record in records:
        _, times = encoded[record["id"]]
        supervised = np.stack([clip_indices(times, center, config["clip_frames"], config["fps"])
            for center in labeled_centers(record, times, config["positive_radius_sec"])])
        timeline = np.stack([clip_indices(times, center, config["clip_frames"], config["fps"])
            for center in times])
        sampling[record["id"]] = {"video_sha256": record["sha256"],
            "cache_sha256": file_hash(record["cache"]), "split": record["split"],
            "times_sha256": hashlib.sha256(times.tobytes()).hexdigest(),
            "supervised_indices_sha256": hashlib.sha256(supervised.tobytes()).hexdigest(),
            "timeline_indices_sha256": hashlib.sha256(timeline.tobytes()).hexdigest()}
    write_json(output / "sampling.json", sampling)
    train_x, train_y, weights = supervised_tensors(train_records, encoded, config)
    dev_x, dev_y, _ = supervised_tensors(dev_records, encoded, config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config["learning_rate"], weight_decay=config["weight_decay"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.BCEWithLogitsLoss()
    initial_loss = balanced_loss(model, train_x, train_y, device)
    best_loss, steps = float("inf"), 0
    generator = torch.Generator().manual_seed(config["seed"])
    print(json.dumps({"head_type": head_type, "device": str(device), "train_videos": len(train_records),
        "dev_videos": len(dev_records), "trainable_parameters": sum(trainable.values()),
        "initial_train_loss": initial_loss}), flush=True)
    for epoch in range(1, epochs+1):
        sampled = torch.multinomial(weights, samples_per_epoch, replacement=True, generator=generator)
        model.train()
        running_loss = 0.0
        for start in range(0, len(sampled), batch_size):
            index = sampled[start:start+batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model.classify(train_x[index].to(device))
            loss = criterion(logits, train_y[index].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            steps += 1
            running_loss += loss.item()*len(index)
        schedule.step()
        dev_loss = balanced_loss(model, dev_x, dev_y, device)
        head_l1 = sum(float((p.detach().cpu()-head_initial[name]).abs().sum())
                      for name, p in model.named_parameters() if name in head_initial)
        checkpoint = {"format_version": 2, "model": {name: value.cpu() for name, value in model.state_dict().items()},
            "config": config.copy(), "training_steps": steps, "epoch": epoch,
            "weight_changes": {"encoder_l1": 0.0, "head_l1": head_l1},
            "encoder_hash": encoder_before, "manifest_sha256": file_hash(manifest_path),
            "encoder_source_sha256": file_hash(encoder_weights), "dev_loss": dev_loss}
        torch.save(checkpoint, output / "last.pt")
        if dev_loss < best_loss:
            best_loss = dev_loss
            torch.save(checkpoint, output / "best.pt")
        log = {"epoch": epoch, "train_loss": running_loss/samples_per_epoch, "dev_loss": dev_loss,
               "head_l1": head_l1, "elapsed_sec": round(time.monotonic()-started, 3)}
        with (output / "history.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(log)+"\n")
        print(json.dumps(log), flush=True)
    saved = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"])
    if tensor_hash(model.encoder.state_dict()) != encoder_before or saved["weight_changes"]["head_l1"] <= 0:
        raise RuntimeError("Frozen encoder / trained head verification failed")
    timelines = feature_timelines(model, records, encoded, config, device)
    postprocess, dev_metrics, trials = calibrate(dev_records, timelines, config["event_tolerance_sec"])
    config["postprocess"] = postprocess
    saved["config"] = config
    saved["calibrated_on"] = "development_only"
    torch.save(saved, output / "best.pt")
    write_json(output / "config.json", config)
    write_json(output / "calibration.json", {"objective": "event F1, then video F1, then matched MAE",
        "split": "dev", "candidate_count": len(trials), "selected": postprocess, "trials": trials})
    state_hash = tensor_hash(saved["model"])
    report = {"head_type": head_type, "trainable_parameters": sum(trainable.values()),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "training_steps": steps, "selected_epoch": saved["epoch"], "selected_checkpoint_steps": saved["training_steps"],
        "initial_train_loss": initial_loss, "selected_train_loss": balanced_loss(model, train_x, train_y, device),
        "selected_dev_loss": best_loss, "weight_changes": saved["weight_changes"],
        "encoder_sha256_before": encoder_before, "encoder_sha256_after": tensor_hash(model.encoder.state_dict()),
        "model_state_sha256": state_hash, "encoder_feature_extraction_sec": encode_seconds,
        "training_total_sec": time.monotonic()-started,
        "peak_vram_allocated_mb": torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else 0,
        "peak_vram_reserved_mb": torch.cuda.max_memory_reserved(device)/2**20 if device.type == "cuda" else 0,
        "train": metrics_from_scores(train_records, timelines, postprocess, config["event_tolerance_sec"]),
        "dev": dev_metrics, "independent_test": None,
        "config": config, "selection": "Epoch and event parameters selected on dev; reported dev metrics are post-selection, not independent test quality.",
        "group_review_complete": manifest.get("manual_group_review_complete", False)}
    write_json(output / "report.json", report)
    return report


def compare_reports(temporal_dir, baseline_dir, output):
    temporal_dir, baseline_dir = Path(temporal_dir), Path(baseline_dir)
    temporal = json.loads((temporal_dir / "report.json").read_text(encoding="utf-8"))
    baseline = json.loads((baseline_dir / "report.json").read_text(encoding="utf-8"))
    first = json.loads((temporal_dir / "manifest.json").read_text(encoding="utf-8"))
    second = json.loads((baseline_dir / "manifest.json").read_text(encoding="utf-8"))
    if first != second:
        raise ValueError("Comparison requires identical data and splits")
    labels = [r["label"] for r in first["records"] if r["split"] == "train"]
    majority = int(sum(labels) > len(labels)/2)
    dev_labels = [r["label"] for r in first["records"] if r["split"] in {"dev", "val"}]
    result = {"temporal": temporal["dev"], "linear_mean_pool_baseline": baseline["dev"],
        "constant_majority_baseline": binary_metrics(dev_labels, [majority]*len(dev_labels)),
        "constant_majority_label": majority, "same_split": True,
        "note": "Both learned models receive the same development tuning budget. Only 2 dev videos; no independent test."}
    def rank(report):
        events = report["dev"]["events"]
        return (events["f1"], report["dev"]["video"]["f1"],
                -(events["mae_sec"] if events["mae_sec"] is not None else float("inf")),
                -report["trainable_parameters"])
    selected_dir = baseline_dir if rank(baseline) >= rank(temporal) else temporal_dir
    result["selected_model"] = "linear" if selected_dir == baseline_dir else "temporal"
    destination = Path(output).parent
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected_dir / "best.pt", destination / "selected.pt")
    shutil.copyfile(selected_dir / "config.json", destination / "selected_config.json")
    write_json(destination / "selection.json", {"model": result["selected_model"],
        "source_checkpoint": str(selected_dir / "best.pt"),
        "reason": "Development event F1, video F1, localization MAE, then smaller head.",
        "independent_test_used": False, "quality_not_established": True})
    write_json(output, result)
    return result


def evaluate_checkpoint(checkpoint, manifest_path, split, output, device="auto"):
    """Evaluation only: never changes checkpoint, threshold or event configuration."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records = [r for r in manifest["records"] if r["split"] == split]
    if not records:
        raise ValueError(f"No videos in split {split}; an independent test is not synthesized")
    if split == "test":
        if manifest.get("group_review", {}).get("status") != "confirmed":
            raise ValueError("Test source-group review is pending; independent quality evaluation is not authorized by the data protocol")
        if (manifest.get("label_revisions_pending_review")
                or manifest.get("test_reservation", {}).get("qualification_status") == "invalidated_by_label_revision"):
            raise ValueError("Test qualification was invalidated by annotation revisions; a reviewed protocol is required")
        for other in [r for r in manifest["records"] if r["split"] != "test"]:
            if any(any(record[key] == other[key] for key in ["id", "sha256", "group"]) for record in records):
                raise ValueError("Test overlaps another partition")
        if Path(output).exists():
            raise FileExistsError("Preserve the existing test evaluation")
    for record in records:
        if file_hash(record["path"]) != record["sha256"]:
            raise ValueError("Video bytes changed after manifest preparation")
    analyzer = VideoAnalyzer(checkpoint, device)
    timelines, timings = {}, []
    for record in records:
        start = time.monotonic()
        times, scores, _ = analyzer.score_video(record["path"])
        timelines[record["id"]] = (times, scores)
        timings.append({"video_id": record["id"], "elapsed_sec": time.monotonic()-start, **analyzer.last_timings})
    metrics = metrics_from_scores(records, timelines, analyzer.postprocess, analyzer.config.get("event_tolerance_sec", 5))
    result = {"split": split, "checkpoint_sha256": file_hash(checkpoint), "metrics": metrics, "timings": timings,
        "parameters_changed": False, "note": "Dev was used for selection; only a separately reserved test is independent."}
    write_json(output, result)
    return result
