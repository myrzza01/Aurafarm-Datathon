"""Internal development metrics. Point annotations do not support temporal IoU."""
from __future__ import annotations

import itertools
import math
import numpy as np


def binary_metrics(labels, predictions):
    tp = sum(a == 1 and b == 1 for a, b in zip(labels, predictions))
    tn = sum(a == 0 and b == 0 for a, b in zip(labels, predictions))
    fp = sum(a == 0 and b == 1 for a, b in zip(labels, predictions))
    fn = sum(a == 1 and b == 0 for a, b in zip(labels, predictions))
    precision = tp / (tp+fp) if tp+fp else 0.0
    recall = tp / (tp+fn) if tp+fn else 0.0
    specificity = tn / (tn+fp) if tn+fp else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
            "balanced_accuracy": (recall+specificity)/2}


CONFIRMATION_POLICIES = ("single", "adjacent_2", "adjacent_3", "two_of_three")


def _confirmed_indices(times, indices, confirmation, confirmation_max_step_sec):
    """Confirm locally before merging events; distant spikes cannot confirm each other.

    Positions are indices in the complete score timeline, including below-threshold
    windows. The fixed step cap tolerates PTS jitter at 2 fps, but prevents a decode
    gap from masquerading as contiguous evidence. This is not independent evidence:
    neighbouring windows overlap and usually share seven of eight frames.
    """
    if confirmation in {"legacy", "single"}:
        return indices
    accepted = set()
    if confirmation == "two_of_three":
        for first, second in zip(indices, indices[1:]):
            step_count = int(second-first)
            if step_count <= 2 and times[second]-times[first] <= step_count*confirmation_max_step_sec + 1e-8:
                accepted.update((int(first), int(second)))
    else:
        required = 2 if confirmation == "adjacent_2" else 3
        for start in range(len(indices)-required+1):
            candidate = indices[start:start+required]
            if (np.all(np.diff(candidate) == 1)
                    and np.all(np.diff(times[candidate]) <= confirmation_max_step_sec + 1e-8)):
                accepted.update(int(index) for index in candidate)
    return np.asarray(sorted(accepted), dtype=np.int64)


def detect_intervals(times, scores, threshold=0.5, merge_gap_sec=1.0, min_hits=1,
                     confirmation="legacy", confirmation_max_step_sec=0.75):
    """Return peak-window center PTS, not a semantic event onset or interval boundary.

    Without ``confirmation`` the historical min_hits-after-merge rule is unchanged.
    New policies confirm score hits locally *before* merging, after which min_hits
    still has its original meaning (total retained hits in a merged group).
    """
    if not 0 < threshold < 1 or merge_gap_sec < 0 or min_hits < 1:
        raise ValueError("Invalid event postprocessing configuration")
    if confirmation not in {"legacy", *CONFIRMATION_POLICIES} or confirmation_max_step_sec <= 0:
        raise ValueError("Invalid event confirmation configuration")
    times, scores = np.asarray(times, dtype=np.float64), np.asarray(scores, dtype=np.float64)
    if (times.ndim != 1 or scores.ndim != 1 or len(times) != len(scores)
            or not np.all(np.isfinite(times)) or not np.all(np.isfinite(scores))
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Invalid score timeline")
    indices = _confirmed_indices(times, np.flatnonzero(scores >= threshold),
                                 confirmation, confirmation_max_step_sec)
    groups = []
    for index in indices:
        if not groups or times[index] - times[groups[-1][-1]] > merge_gap_sec + 1e-8:
            groups.append([int(index)])
        else:
            groups[-1].append(int(index))
    events = []
    for group in groups:
        if len(group) < min_hits:
            continue
        peak = max(group, key=lambda i: scores[i])
        events.append({"start_sec": float(times[group[0]]), "end_sec": float(times[group[-1]]),
                       "timecode_sec": float(times[peak]), "score": float(scores[peak]),
                       "type_id": None})  # No category is inferred from binary labels.
    return events


def match_events(truth, predicted, tolerance_sec=5.0):
    """Maximum-cardinality, minimum-error monotone matching of points in 1D."""
    truth, predicted = sorted(truth), sorted(predicted)
    n, m = len(truth), len(predicted)
    # state = matched count, total absolute error, index pairs
    dp = [[(0, 0.0, ()) for _ in range(m+1)] for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            candidates = [dp[i-1][j], dp[i][j-1]]
            error = abs(truth[i-1] - predicted[j-1])
            if error <= tolerance_sec:
                count, cost, pairs = dp[i-1][j-1]
                candidates.append((count+1, cost+error, pairs+((i-1, j-1),)))
            dp[i][j] = max(candidates, key=lambda state: (state[0], -state[1]))
    pairs = dp[n][m][2]
    return {"matched": len(pairs), "missed": n-len(pairs), "extra": m-len(pairs),
            "errors_sec": [abs(truth[i]-predicted[j]) for i, j in pairs],
            "pairs": [{"truth_sec": truth[i], "predicted_sec": predicted[j],
                       "error_sec": abs(truth[i]-predicted[j])} for i, j in pairs]}


def metrics_from_scores(records, timelines, postprocess, tolerance_sec=5.0):
    labels, predictions, details, errors = [], [], [], []
    matched = missed = extra = 0
    for record in records:
        times, scores = timelines[record["id"]]
        events = detect_intervals(times, scores, **postprocess)
        match = match_events(record["event_times"], [event["timecode_sec"] for event in events], tolerance_sec)
        matched += match["matched"]
        missed += match["missed"]
        extra += match["extra"]
        errors.extend(match["errors_sec"])
        labels.append(record["label"])
        predictions.append(int(bool(events)))
        details.append({"video_id": record["id"], "label": record["label"], "prediction": int(bool(events)),
                        "max_score": float(scores.max()), "events": events, "localization": match})
    event_metrics = {"matched": matched, "missed": missed, "extra": extra,
        "precision": matched/(matched+extra) if matched+extra else 0.0,
        "recall": matched/(matched+missed) if matched+missed else 0.0,
        "f1": 2*matched/(2*matched+missed+extra) if 2*matched+missed+extra else 0.0,
        "errors_sec": errors, "mae_sec": float(np.mean(errors)) if errors else None,
        "median_error_sec": float(np.median(errors)) if errors else None,
        "tolerance_sec": tolerance_sec,
        "note": "Internal point matching, not jury metrics. Unmatched detections may reflect incomplete annotations."}
    return {"video": binary_metrics(labels, predictions), "events": event_metrics, "videos": details}


def calibration_configs():
    """Predeclared grid: same 84 settings for every model, development only."""
    for threshold, gap, policy in itertools.product(
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [0.6, 1.0, 2.0], CONFIRMATION_POLICIES):
        yield {"threshold": threshold, "merge_gap_sec": gap, "min_hits": 1,
               "confirmation": policy, "confirmation_max_step_sec": 0.75}


def calibrate(records, timelines, tolerance_sec=5.0):
    if not records or any(r["split"] not in {"dev", "val"} for r in records):
        raise ValueError("Calibration is allowed only on explicitly marked development videos")
    candidates, best = [], None
    for config in calibration_configs():
        threshold, gap = config["threshold"], config["merge_gap_sec"]
        metrics = metrics_from_scores(records, timelines, config, tolerance_sec)
        e, v = metrics["events"], metrics["video"]
        key = (e["f1"], v["f1"], -(e["mae_sec"] if e["mae_sec"] is not None else math.inf),
               -e["extra"], -abs(threshold-0.5), -gap,
               -CONFIRMATION_POLICIES.index(config["confirmation"]))
        candidates.append({**config, "event_f1": e["f1"], "video_f1": v["f1"],
                           "tp": v["tp"], "fp": v["fp"], "fn": v["fn"], "tn": v["tn"],
                           "matched": e["matched"], "matched_mae_sec": e["mae_sec"],
                           "missed": e["missed"], "extra": e["extra"]})
        if best is None or key > best[0]:
            best = (key, config, metrics)
    return best[1], best[2], candidates
