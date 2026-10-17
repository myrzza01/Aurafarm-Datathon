"""End-to-end event metrics include localization misses and wrong action types.

No scores are calibrated here. Monotone temporal matching maximizes event
matches, then minimizes total absolute timing error; action agreement breaks
remaining ties between equally good monotone solutions.
Unknown human truth categories are reported explicitly rather than fabricated.
"""
from __future__ import annotations

from collections import defaultdict
import math


UNCLASSIFIED = "__unclassified__"


def _event(value, index, *, truth):
    if not isinstance(value, dict):
        raise ValueError("Each event must be an object")
    try:
        timestamp = float(value["timecode_sec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Each event requires a numeric timecode_sec") from exc
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("Event timecodes must be finite and nonnegative")
    type_id = value.get("type_id")
    if type_id is not None and (not isinstance(type_id, str) or not type_id.strip()):
        raise ValueError("type_id must be a nonempty string or null")
    if type_id == UNCLASSIFIED:
        raise ValueError("Reserved unknown-type marker cannot be an input class")
    known = bool(type_id and value.get("type_verified", False)) if truth else bool(type_id)
    return {"timecode_sec": timestamp, "type_id": type_id, "known": known, "index": index}


def _matches(truth, predicted, tolerance):
    ordered_truth = sorted(truth, key=lambda event: (event["timecode_sec"], event["type_id"] or "", event["index"]))
    ordered_prediction = sorted(predicted, key=lambda event: (event["timecode_sec"], event["type_id"] or "", event["index"]))
    # State: temporal cardinality, total timing error, correct types, index pairs.
    previous = [(0, 0.0, 0, ())] * (len(ordered_prediction) + 1)
    for actual in ordered_truth:
        current = [(0, 0.0, 0, ())]
        for column, proposed in enumerate(ordered_prediction, 1):
            candidates = [previous[column], current[column - 1]]
            error = abs(actual["timecode_sec"] - proposed["timecode_sec"])
            if error <= tolerance:
                count, cost, correct, pairs = previous[column - 1]
                agreement = actual["known"] and actual["type_id"] == proposed["type_id"]
                candidates.append((count + 1, cost + error, correct + int(agreement),
                    pairs + ((actual["index"], proposed["index"]),)))
            current.append(max(candidates, key=lambda state: (state[0], -state[1], state[2])))
        previous = current
    return previous[-1][3]


def _rates(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def typed_event_metrics(records, predictions, tolerance_sec=5.0):
    """Evaluate actual detections with their predicted type, per video.

    ``predictions`` maps each record ID to a list of objects containing
    ``timecode_sec`` and ``type_id`` (null is an unclassified alert). A missing
    prediction entry is an incomplete analysis and raises instead of silently
    becoming ``no_violation``. Truth types require ``type_verified=True``.

    A wrong type contributes one typed FP and one typed FN. A prediction matched
    to truth without a verified type is excluded from typed precision/recall;
    localization metrics still include it. Unmatched alerts remain typed FPs,
    including alerts on negative videos and alerts with no predicted type.
    """
    if not isinstance(predictions, dict):
        raise ValueError("Predictions must map video IDs to event lists")
    if not isinstance(tolerance_sec, (int, float)) or not math.isfinite(tolerance_sec) or tolerance_sec < 0:
        raise ValueError("tolerance_sec must be finite and nonnegative")
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    counts = {key: 0 for key in ("truth", "known_truth", "untyped_truth", "predicted", "matched", "missed", "extra",
        "correct", "wrong", "known_missed", "untyped_matched")}
    details, errors, seen_ids = [], [], set()
    for record in records:
        identifier = record["id"]
        if identifier in seen_ids:
            raise ValueError("Duplicate video IDs would double-weight the evaluation")
        seen_ids.add(identifier)
        if identifier not in predictions:
            raise ValueError(f"Missing prediction for video {identifier}")
        if not isinstance(predictions[identifier], list):
            raise ValueError("Each completed video prediction must be an event list")
        raw_truth = record.get("events")
        if raw_truth is None:
            raw_truth = [{"timecode_sec": timestamp, "type_id": None} for timestamp in record.get("event_times", [])]
        if not isinstance(raw_truth, list):
            raise ValueError("Human truth events must be a list")
        truth = [_event(event, index, truth=True) for index, event in enumerate(raw_truth)]
        predicted = [_event(event, index, truth=False) for index, event in enumerate(predictions[identifier])]
        pairs = _matches(truth, predicted, tolerance_sec)
        paired_truth, paired_prediction = {a for a, _ in pairs}, {b for _, b in pairs}
        for actual in truth:
            if actual["known"]:
                per_class[actual["type_id"]]["support"] += 1
        matches, missed, extra = [], [], []
        correct = wrong = known_missed = untyped_matched = 0
        for first, second in pairs:
            actual, proposed = truth[first], predicted[second]
            error = abs(actual["timecode_sec"] - proposed["timecode_sec"])
            errors.append(error)
            if not actual["known"]:
                category = "untyped_truth"
                untyped_matched += 1
            elif actual["type_id"] == proposed["type_id"]:
                category = "type_correct"
                correct += 1
                per_class[actual["type_id"]]["tp"] += 1
            else:
                category = "wrong_type"
                wrong += 1
                per_class[actual["type_id"]]["fn"] += 1
                per_class[proposed["type_id"] or UNCLASSIFIED]["fp"] += 1
            matches.append({"truth_event_index": first, "predicted_event_index": second,
                "truth_sec": actual["timecode_sec"], "predicted_sec": proposed["timecode_sec"], "error_sec": error,
                "truth_type_id": actual["type_id"] if actual["known"] else None,
                "predicted_type_id": proposed["type_id"], "category": category})
        for index, actual in enumerate(truth):
            if index not in paired_truth:
                missed.append({"truth_event_index": index, "timecode_sec": actual["timecode_sec"],
                    "type_id": actual["type_id"] if actual["known"] else None})
                if actual["known"]:
                    known_missed += 1
                    per_class[actual["type_id"]]["fn"] += 1
        for index, proposed in enumerate(predicted):
            if index not in paired_prediction:
                extra.append({"predicted_event_index": index, "timecode_sec": proposed["timecode_sec"], "type_id": proposed["type_id"]})
                per_class[proposed["type_id"] or UNCLASSIFIED]["fp"] += 1
        local = {"truth": len(truth), "known_truth": sum(event["known"] for event in truth),
            "untyped_truth": sum(not event["known"] for event in truth), "predicted": len(predicted),
            "matched": len(pairs), "missed": len(missed), "extra": len(extra),
            "correct": correct, "wrong": wrong, "known_missed": known_missed, "untyped_matched": untyped_matched}
        for key, value in local.items():
            counts[key] += value
        details.append({"video_id": identifier, "counts": local, "matches": matches,
            "missed_truth": missed, "extra_predictions": extra})
    finalized_classes = {key: {**_rates(value["tp"], value["fp"], value["fn"]), "support": value["support"]}
        for key, value in sorted(per_class.items())}
    supported = [value["f1"] for value in finalized_classes.values() if value["support"]]
    typed = _rates(counts["correct"], counts["extra"] + counts["wrong"], counts["known_missed"] + counts["wrong"])
    typed.update({"type_correct_matched": counts["correct"], "wrong_type_matched": counts["wrong"],
        "missed_typed_truth": counts["known_missed"], "unmatched_alerts": counts["extra"],
        "classification_accuracy_on_temporally_matched_typed_events": counts["correct"] / (counts["correct"] + counts["wrong"]) if counts["correct"] + counts["wrong"] else None,
        "macro_f1_supported": sum(supported) / len(supported) if supported else None})
    return {"tolerance_sec": float(tolerance_sec), "video_count": len(seen_ids),
        "ground_truth_events": counts["truth"], "typed_ground_truth_events": counts["known_truth"],
        "untyped_ground_truth_events": counts["untyped_truth"], "predicted_events": counts["predicted"],
        "predictions_excluded_for_untyped_matched_truth": counts["untyped_matched"],
        "detection": {**_rates(counts["matched"], counts["extra"], counts["missed"]),
            "matched": counts["matched"], "missed": counts["missed"], "extra": counts["extra"],
            "errors_sec": errors, "matched_mae_sec": sum(errors) / len(errors) if errors else None},
        "typed": typed, "per_class": finalized_classes, "videos": details,
        "matching": "One-to-one monotone temporal matching: maximum cardinality, minimum total absolute timing error, then type agreement between equally good monotone solutions.",
        "scope": "End-to-end detected typed events; includes detector misses, wrong types, and unmatched alerts. Human-unverified truth types are explicitly excluded from typed scoring."}
