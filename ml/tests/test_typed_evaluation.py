import math
import unittest

from cashwatch.typed_evaluation import UNCLASSIFIED, typed_event_metrics


def event(time, kind, verified=True):
    return {"timecode_sec": time, "type_id": kind, "type_verified": verified}


def video(identifier="video", truth=()):
    return {"id": identifier, "events": list(truth)}


class TypedEvaluationTests(unittest.TestCase):
    def test_wrong_type_counts_as_typed_fp_and_fn_and_false_alert_stays_fp(self):
        actual = [video(truth=[event(10, "a"), event(20, "b")])]
        result = typed_event_metrics(actual, {"video": [event(10, "a"), event(20, "c"), event(40, "a")]})
        self.assertEqual((result["detection"]["matched"], result["detection"]["missed"], result["detection"]["extra"]), (2, 0, 1))
        self.assertEqual((result["typed"]["tp"], result["typed"]["fp"], result["typed"]["fn"]), (1, 2, 1))
        self.assertAlmostEqual(result["typed"]["f1"], 0.4)
        self.assertEqual(result["per_class"]["b"]["fn"], 1)
        self.assertEqual(result["per_class"]["c"]["fp"], 1)

    def test_detection_misses_reduce_typed_recall_even_if_classification_at_truth_would_be_perfect(self):
        result = typed_event_metrics([video(truth=[event(10, "a"), event(20, "b")])], {"video": [event(10, "a")]})
        self.assertEqual(result["typed"]["missed_typed_truth"], 1)
        self.assertEqual(result["typed"]["recall"], 0.5)
        self.assertEqual(result["typed"]["classification_accuracy_on_temporally_matched_typed_events"], 1.0)
        self.assertLess(result["typed"]["f1"], 1.0)

    def test_duplicate_alerts_cannot_match_one_truth_twice(self):
        result = typed_event_metrics([video(truth=[event(10, "a")])], {"video": [event(10, "a"), event(11, "a")]})
        self.assertEqual(result["typed"]["tp"], 1)
        self.assertEqual(result["typed"]["fp"], 1)
        self.assertEqual(result["detection"]["extra"], 1)

    def test_temporal_matching_does_not_reassign_to_farther_correct_class_to_hide_wrong_types(self):
        result = typed_event_metrics([video(truth=[event(10, "a"), event(20, "b")])],
            {"video": [event(10, "b"), event(20, "a")]}, tolerance_sec=11)
        self.assertEqual(result["detection"]["matched_mae_sec"], 0)
        self.assertEqual(result["typed"]["tp"], 0)
        self.assertEqual(result["typed"]["wrong_type_matched"], 2)

    def test_simultaneous_distinct_actions_match_types_independently_of_input_order(self):
        result = typed_event_metrics([video(truth=[event(10, "b"), event(10, "a")])],
            {"video": [event(10, "a"), event(10, "b")]}, tolerance_sec=0)
        self.assertEqual(result["typed"]["tp"], 2)
        self.assertEqual(result["typed"]["f1"], 1.0)

    def test_negative_video_false_alerts_are_scored(self):
        result = typed_event_metrics([video("negative")], {"negative": [event(5, "a")]})
        self.assertEqual(result["typed"]["fp"], 1)
        self.assertEqual(result["ground_truth_events"], 0)
        self.assertEqual(result["per_class"]["a"]["support"], 0)
        self.assertIsNone(result["typed"]["macro_f1_supported"])

    def test_unclassified_detected_alert_is_wrong_type_for_known_truth(self):
        result = typed_event_metrics([video(truth=[event(10, "a")])], {"video": [event(10, None)]})
        self.assertEqual(result["typed"]["wrong_type_matched"], 1)
        self.assertEqual(result["per_class"][UNCLASSIFIED]["fp"], 1)

    def test_unverified_human_type_is_reported_without_invented_ground_truth_class(self):
        actual = [video(truth=[event(10, "a"), event(20, "unverified-note", verified=False)])]
        result = typed_event_metrics(actual, {"video": [event(10, "a"), event(20, "b")]})
        self.assertEqual(result["untyped_ground_truth_events"], 1)
        self.assertEqual(result["predictions_excluded_for_untyped_matched_truth"], 1)
        self.assertEqual(result["typed"]["f1"], 1.0)
        self.assertNotIn("unverified-note", result["per_class"])
        self.assertNotIn("b", result["per_class"])

    def test_video_timelines_are_never_matched_across_video_boundaries(self):
        result = typed_event_metrics([video("a", [event(10, "a")]), video("b")],
            {"a": [], "b": [event(10, "a")]})
        self.assertEqual(result["typed"]["tp"], 0)
        self.assertEqual((result["typed"]["fp"], result["typed"]["fn"]), (1, 1))

    def test_missing_or_failed_analysis_is_not_silently_treated_as_no_violation(self):
        with self.assertRaisesRegex(ValueError, "Missing prediction"):
            typed_event_metrics([video("not-analyzed")], {})
        with self.assertRaisesRegex(ValueError, "event list"):
            typed_event_metrics([video()], {"video": {"status": "failed"}})
        with self.assertRaisesRegex(ValueError, "double-weight"):
            typed_event_metrics([video(), video()], {"video": []})

    def test_invalid_timing_values_fail_explicitly(self):
        for tolerance in (-1, math.nan, math.inf):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                typed_event_metrics([video()], {"video": []}, tolerance)
        for timestamp in (-1, math.nan, math.inf, "not-a-time"):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                typed_event_metrics([video()], {"video": [event(timestamp, "a")]})

    def test_empty_predictions_for_completed_negative_video_remain_valid(self):
        result = typed_event_metrics([video()], {"video": []})
        self.assertEqual(result["video_count"], 1)
        self.assertEqual(result["typed"]["fp"], 0)
        self.assertEqual(result["detection"]["missed"], 0)


if __name__ == "__main__":
    unittest.main()
