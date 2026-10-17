"""Inference uses only pixels and saved weights, never labels or filename features."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .model import VideoDetector, select_device
from .evaluation import detect_intervals
from .video import clip_indices, decode_sampled, evidence_frames


def default_checkpoint_path():
    root = Path(__file__).resolve().parents[1]
    packaged = root / "weights" / "detector.pt"
    return packaged if packaged.is_file() else root / "artifacts" / "cycle" / "selected.pt"


def score_frames(model, images, times, device, frames=8, fps=2.0, deadline=None):
    features = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(images), 32):
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("Inference time budget exceeded")
            batch = torch.from_numpy(images[start:start+32].copy()).permute(0, 3, 1, 2).to(device).float() / 255
            features.append(model.encode(batch))
        features = torch.cat(features)
        scores = []
        for start in range(0, len(times), 64):
            indices = np.stack([clip_indices(times, center, frames, fps) for center in times[start:start+64]])
            windows = features[torch.as_tensor(indices, dtype=torch.long, device=device)]
            scores.extend(model.classify(windows).sigmoid().cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


def event_peaks(times, scores, threshold=0.5, merge_gap=1.0):
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    indices = np.flatnonzero(scores >= threshold)
    groups = []
    for index in indices:
        if not groups or times[index] - times[groups[-1][-1]] > merge_gap:
            groups.append([int(index)])
        else:
            groups[-1].append(int(index))
    return [max(group, key=lambda i: scores[i]) for group in groups]


class VideoAnalyzer:
    """Load once per backend worker; analyze many local videos."""
    def __init__(self, checkpoint=None, device="auto", threshold=None):
        self.device = select_device(device)
        checkpoint = default_checkpoint_path() if checkpoint is None else checkpoint
        saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if saved.get("format_version") not in {1, 2} or saved.get("training_steps", 0) < 1:
            raise ValueError("Expected a trained cashwatch checkpoint")
        self.config = saved["config"]
        self.postprocess = self.config.get("postprocess", {"threshold": self.config.get("threshold", 0.5),
            "merge_gap_sec": 1.0, "min_hits": 1}).copy()
        self.threshold = float(threshold if threshold is not None else self.postprocess["threshold"])
        self.postprocess["threshold"] = self.threshold
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        # pretrained=False is intentional: inference must never download weights.
        self.model = VideoDetector(pretrained=False, head_type=self.config.get("head_type", "temporal")).to(self.device)
        self.model.load_state_dict(saved["model"])
        self.model.eval()

    def score_video(self, video_path, deadline=None):
        started = time.monotonic()
        images, times, duration = decode_sampled(video_path, fps=self.config["fps"],
            size=self.config["image_size"], deadline=deadline)
        decoded = time.monotonic()
        scores = score_frames(self.model, images, times, self.device,
            self.config["clip_frames"], self.config["fps"], deadline)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.last_timings = {"decode_sec": decoded-started, "model_sec": time.monotonic()-decoded}
        return times, scores, duration

    def analyze(self, video_path, timeout_sec=240.0):
        started = time.monotonic()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        deadline = started + timeout_sec
        times, scores, duration = self.score_video(video_path, deadline)
        events = detect_intervals(times, scores, **self.postprocess)
        targets = [point for event in events for point in (event["start_sec"], event["timecode_sec"], event["end_sec"])]
        evidence_started = time.monotonic()
        frames = evidence_frames(video_path, targets, deadline)
        violations = []
        for event in events:
            peak = frames[event["timecode_sec"]]
            event_targets = sorted(set([event["start_sec"], event["timecode_sec"], event["end_sec"]]))
            violations.append({
                "type": "Признаки возможного нарушения (тип не классифицирован)",
                "timecode_sec": peak["timestamp_sec"],
                "description": "Обученная бинарная модель выделила этот видеофрагмент для проверки человеком. "
                    "Конкретное действие и нормативное основание этой моделью не определяются.",
                "evidence_frames": [frames[target] for target in event_targets]})
        if time.monotonic() > deadline:
            raise TimeoutError("Inference time budget exceeded")
        result = {"verdict": "violation" if violations else "no_violation", "violations": violations}
        diagnostics = {"elapsed_sec": round(time.monotonic() - started, 3),
            **self.last_timings, "evidence_sec": time.monotonic()-evidence_started,
            "peak_vram_allocated_mb": torch.cuda.max_memory_allocated(self.device)/2**20 if self.device.type == "cuda" else 0,
            "peak_vram_reserved_mb": torch.cuda.max_memory_reserved(self.device)/2**20 if self.device.type == "cuda" else 0,
            "video_duration_sec": duration, "max_score": float(scores.max()), "threshold": self.threshold,
            "events": len(violations), "sampled_frames": len(times),
            "intervals": events, "postprocess": self.postprocess,
            "full_http_request_sec": None,
            "limitation": "Binary experimental detector. No event taxonomy or legal rule mapping."}
        return result, diagnostics


def analyze_video(video_path, checkpoint=None, device="auto"):
    """Convenience wrapper. For repeated requests instantiate VideoAnalyzer once."""
    return VideoAnalyzer(checkpoint, device).analyze(video_path)[0]
