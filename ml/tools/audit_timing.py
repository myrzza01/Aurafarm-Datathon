"""Numeric-only audit of frozen baseline, PTS coverage and dev postprocessing.

No frames, pixel arrays, image encodings or scene descriptions are written to the
reports. Test videos are never scored. Semantic annotation review remains human.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cashwatch.evaluation import CONFIRMATION_POLICIES, detect_intervals, metrics_from_scores
from cashwatch.inference import VideoAnalyzer
from cashwatch.video import clip_indices, iter_frames


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def pts_only(path, fps):
    """Mirror the current sampler's PTS decisions without converting any pixels."""
    source, sampled, next_sample = [], [], 0.0
    for timestamp, _ in iter_frames(path):
        source.append(timestamp)
        if timestamp + 1e-8 < next_sample:
            continue
        sampled.append(timestamp)
        next_sample = (int(timestamp*fps + 1e-6)+1)/fps
    if not source:
        raise ValueError("No decoded frames")
    return np.asarray(source), np.asarray(sampled)


def clip_summary(times, center, frames=8, fps=2):
    selected = times[clip_indices(times, center, frames, fps)]
    return {"center_pts_sec": float(center), "selected_frame_pts_sec": selected.tolist(),
            "selected_frame_intervals_sec": np.diff(selected).tolist(),
            "unique_frame_count": len(set(selected.tolist())),
            "selected_support_start_sec": float(selected[0]),
            "selected_support_end_sec": float(selected[-1]),
            "selected_support_span_sec": float(selected[-1]-selected[0]),
            "mean_frame_pts_minus_center_sec": float(selected.mean()-center)}


def compact_metrics(records, timelines, config):
    metrics = metrics_from_scores(records, timelines, config)
    return {"configuration": config, "video": metrics["video"], "events": metrics["events"],
            "per_video": [{"id": item["video_id"], "localization": item["localization"],
                           "detected_events": len(item["events"])} for item in metrics["videos"]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/baselines/cycle_7videos/data/manifest.json")
    parser.add_argument("--checkpoint", default="artifacts/baselines/cycle_7videos/cycle/selected.pt")
    parser.add_argument("--coverage-manifest", help="Optional expanded manifest for PTS-only coverage; test scores are never computed")
    parser.add_argument("--output", default="artifacts/expanded")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    records = manifest["records"]
    dev = [record for record in records if record["split"] in {"dev", "val"}]
    if len(dev) != 2:
        raise ValueError("Frozen baseline audit expects its original two development records")
    checkpoint_hash = digest(args.checkpoint)
    analyzer = VideoAnalyzer(args.checkpoint)
    frames, fps = analyzer.config["clip_frames"], analyzer.config["fps"]
    coverage, source_pts = [], {}
    coverage_records = (json.loads(Path(args.coverage_manifest).read_text(encoding="utf-8"))["records"]
                        if args.coverage_manifest else records)
    for record in coverage_records:
        decoded, sampled = pts_only(record["path"], fps)
        source_pts[record["id"]] = decoded
        support_indices = np.unique(np.concatenate([clip_indices(sampled, center, frames, fps)
                                                   for center in sampled]))
        coverage.append({"video_id": record["id"], "decoded_frame_count": len(decoded),
            "sampled_frame_count": len(sampled), "decoded_first_pts_sec": float(decoded[0]),
            "decoded_last_pts_sec": float(decoded[-1]), "sampled_first_pts_sec": float(sampled[0]),
            "sampled_last_pts_sec": float(sampled[-1]),
            "tail_not_sampled_sec": float(decoded[-1]-sampled[-1]),
            "sampling_gap_min_sec": float(np.diff(sampled).min()),
            "sampling_gap_max_sec": float(np.diff(sampled).max()),
            "all_sampled_frames_used_by_windows": len(support_indices) == len(sampled),
            "first_window": clip_summary(sampled, sampled[0], frames, fps),
            "last_window": clip_summary(sampled, sampled[-1], frames, fps)})
    timelines, detailed = {}, []
    for record in dev:
        times, scores, _ = analyzer.score_video(record["path"])
        timelines[record["id"]] = (times, scores)
        events = detect_intervals(times, scores, **analyzer.postprocess)
        event_details = []
        for event in events:
            peak = int(np.argmin(np.abs(times-event["timecode_sec"])))
            closest_source = float(source_pts[record["id"]][
                np.argmin(np.abs(source_pts[record["id"]]-event["timecode_sec"]))])
            support = clip_summary(times, times[peak], frames, fps)
            event_details.append({**event, **support,
                "nearest_source_frame_pts_sec": closest_source,
                "source_evidence_rounding_error_sec": abs(round(closest_source, 4)-event["timecode_sec"]),
                "above_threshold_hits_in_event": int(np.sum((times >= event["start_sec"])
                    & (times <= event["end_sec"]) & (scores >= analyzer.threshold))),
                "annotation_in_peak_window_support": [
                    support["selected_support_start_sec"] <= truth <= support["selected_support_end_sec"]
                    for truth in record["event_times"]]})
        truths = []
        for truth in record["event_times"]:
            nearest = int(np.argmin(np.abs(times-truth)))
            truths.append({"annotation_sec": truth, "nearest_window_center_sec": float(times[nearest]),
                "nearest_window_score": float(scores[nearest]),
                "nearby_scores": [{"center_sec": float(t), "score": float(s)}
                                   for t, s in zip(times, scores) if abs(t-truth) <= 6],
                "annotation_semantic_correctness": "not adjudicated; point-only annotation has no event boundaries"})
        detailed.append({"video_id": record["id"], "label": record["label"], "annotations": truths,
                         "baseline_events": event_details,
                         "score_timeline": [{"center_sec": float(t), "score": float(s)}
                                            for t, s in zip(times, scores)]})
    baseline_metrics = compact_metrics(dev, timelines, analyzer.postprocess)
    tolerance_sensitivity = [{"tolerance_sec": tolerance,
        "events": metrics_from_scores(dev, timelines, analyzer.postprocess, tolerance)["events"]}
        for tolerance in [1, 2, 3, 5]]
    timing = {"checkpoint_sha256": checkpoint_hash, "read_only_baseline": True,
        "test_videos_scored": 0, "scheme": {
            "clip_frames": frames, "sampling_fps": fps,
            "nominal_window_duration_convention_sec": frames/fps,
            "wanted_first_to_last_span_sec": (frames-1)/fps,
            "wanted_offsets_sec": ((np.arange(frames)-(frames-1)/2)/fps).tolist(),
            "nominal_window_step_sec": 1/fps,
            "sampling_rule": "First decoded frame at or after each 0.5-second grid boundary; use actual PTS relative to first decoded frame.",
            "clip_rule": "Closest sampled PTS to each wanted timestamp; ties choose earlier; boundaries repeat first/last sampled frame.",
            "ideal_actual_offsets_sec": [-2, -1.5, -1, -.5, 0, .5, 1, 1.5],
            "ideal_mean_frame_offset_sec": -.25,
            "event_time_rule": "Center PTS of maximum-scoring retained window within merged event; ties choose earlier window. Returned API time is nearest original decoded evidence frame PTS, rounded to 4 decimals.",
            "event_interval_rule": "First/last above-threshold retained window centers; not annotated onset/end.",
            "ending_rule": "Decoder traverses all frames but final source frame is not forcibly added to model samples.",
            "coverage_limitation": "2-fps sampling can miss sub-0.5-second actions anywhere; final sub-0.5-second tail may not enter model."},
        "coverage": coverage, "dev_audit": detailed, "baseline_metrics": baseline_metrics,
        "point_matching_tolerance_sensitivity_no_parameter_selection": tolerance_sensitivity,
        "conclusions": {
            "timestamp_mapping": "Returned peak time equals an actual source PTS; evidence rounding is separately measured.",
            "sampling_alignment": "Even-length clips on an exact 2-fps grid have a deterministic -0.25-second mean alignment offset, insufficient to explain a 4-second error.",
            "localization": "Binary clip training optimizes class scores, not event onset/center. An argmax may fall away from a point label; numeric evidence alone cannot establish which timestamp is semantically correct.",
            "false_event": "An unmatched threshold crossing is extra relative to provided point labels. Classification error versus incomplete annotation requires local human adjudication.",
            "human_review_required": "Annotator must verify all dev events and onset/end locally; no frames or scene descriptions exported."}}
    candidates = []
    # Fixed threshold isolates postprocessing from changing the classifier decision.
    for gap in [.6, 1., 2.]:
        for hits in [1, 2, 3]:
            candidates.append(compact_metrics(dev, timelines, {
                "threshold": analyzer.threshold, "merge_gap_sec": gap, "min_hits": hits}))
        for policy in CONFIRMATION_POLICIES[1:]:
            candidates.append(compact_metrics(dev, timelines, {"threshold": analyzer.threshold,
                "merge_gap_sec": gap, "min_hits": 1, "confirmation": policy,
                "confirmation_max_step_sec": .75}))
    unchanged = all(row["video"] == baseline_metrics["video"] and row["events"] == baseline_metrics["events"]
                    for row in candidates)
    hit_counts = [event["above_threshold_hits_in_event"] for record in detailed
                  for event in record["baseline_events"]]
    postprocess = {"checkpoint_sha256": checkpoint_hash, "split": "original_development_only",
        "dev_videos": len(dev), "test_videos_scored": 0, "threshold_fixed": analyzer.threshold,
        "baseline": baseline_metrics, "comparisons": candidates,
        "all_18_variants_same_metrics_as_baseline": unchanged,
        "baseline_event_hit_counts": hit_counts,
        "measured_conclusion": "All tested confirmation/merge policies preserve the baseline video and point-event metrics. See per-event support counts; stricter confirmation did not solve these errors."
            if unchanged else "Inspect the saved comparisons; policies change event counts or localization.",
        "policies": {"legacy_min_hits": "Number of hits after gap merging; isolated distant spikes can confirm one another.",
            "adjacent_2": "At least two adjacent above-threshold windows before merging, with <=0.75 sec between centers.",
            "adjacent_3": "At least three adjacent above-threshold windows before merging, with <=0.75 sec per step.",
            "two_of_three": "Two hits among three successive timeline positions before merging; allows one below-threshold window.",
            "overlap": "At 2fps, adjacent interior windows usually share seven frames; hit count is not independent evidence count.",
            "short_event_tradeoff": "A true event causing one hit is deleted by every multi-hit policy; two-hit events are deleted by adjacent_3. Point labels cannot quantify duration-specific recall.",
            "gap_tradeoff": "Increasing merge gap can reduce duplicate detections but merge distinct nearby events. With two_of_three and merge_gap=0.6, confirmed hits separated by one miss remain two events."},
        "predeclared_training_grid": {"thresholds": [.3, .4, .5, .6, .7, .8, .9],
            "merge_gaps_sec": [.6, 1., 2.], "confirmation": list(CONFIRMATION_POLICIES),
            "candidate_count": 84, "identical_for_both_heads": True},
        "limitations": ["Only two development videos; post-selection metrics are optimistic.",
                        "No content correctness is inferred from API schema tests.",
                        "No new model or postprocessing parameters selected using independent test."]}
    if digest(args.checkpoint) != checkpoint_hash:
        raise RuntimeError("Baseline changed during read-only audit")
    output = Path(args.output)
    write(output / "timing_audit.json", timing)
    write(output / "postprocess_audit.json", postprocess)
    print(json.dumps({"written": [str(output / "timing_audit.json"), str(output / "postprocess_audit.json")],
        "videos_coverage_audited": len(coverage), "dev_videos_scored": len(dev),
        "max_unsampled_tail_sec": max(item["tail_not_sampled_sec"] for item in coverage),
        "baseline_video": baseline_metrics["video"], "baseline_events": baseline_metrics["events"]}))


if __name__ == "__main__":
    main()
