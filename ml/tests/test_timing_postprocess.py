import unittest

import numpy as np

from cashwatch.evaluation import calibration_configs, calibrate, detect_intervals
from cashwatch.video import clip_indices


class ConfirmationTests(unittest.TestCase):
    def test_legacy_hits_can_merge_isolated_peaks_but_local_confirmation_cannot(self):
        times = np.arange(9) / 2
        scores = np.array([0, .8, 0, 0, 0, .9, 0, 0, 0])
        legacy = detect_intervals(times, scores, merge_gap_sec=2, min_hits=2)
        self.assertEqual(len(legacy), 1)
        for policy in ["adjacent_2", "adjacent_3", "two_of_three"]:
            self.assertEqual(detect_intervals(times, scores, merge_gap_sec=2,
                                             confirmation=policy), [])

    def test_short_real_event_tradeoff_is_explicit(self):
        times = np.arange(7) / 2
        singleton = np.array([0, 0, 0, .9, 0, 0, 0])
        self.assertEqual(len(detect_intervals(times, singleton, confirmation="single")), 1)
        self.assertEqual(detect_intervals(times, singleton, confirmation="adjacent_2"), [])
        pair = np.array([0, 0, .8, .9, 0, 0, 0])
        self.assertEqual(len(detect_intervals(times, pair, confirmation="adjacent_2")), 1)
        self.assertEqual(detect_intervals(times, pair, confirmation="adjacent_3"), [])

    def test_confirmation_uses_below_threshold_positions_and_precedes_merging(self):
        times = np.arange(9) / 2
        scores = np.array([0, .8, 0, .9, 0, 0, 0, 0, 0])
        self.assertEqual(detect_intervals(times, scores, merge_gap_sec=2,
                                         confirmation="adjacent_2"), [])
        events = detect_intervals(times, scores, merge_gap_sec=1,
                                  confirmation="two_of_three")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["timecode_sec"], 1.5)
        self.assertEqual(len(detect_intervals(times, scores, merge_gap_sec=.6,
                                              confirmation="two_of_three")), 2)

    def test_confirmation_does_not_bridge_decode_gap(self):
        times = np.array([0., .5, 10., 10.5])
        scores = np.array([0., .8, .9, 0.])
        self.assertEqual(detect_intervals(times, scores, merge_gap_sec=20,
                                         confirmation="adjacent_2"), [])

    def test_invalid_timeline_is_an_error(self):
        for times in [[0, 0], [1, 0], [0, np.nan]]:
            with self.assertRaises(ValueError):
                detect_intervals(times, [.8, .8])
        with self.assertRaises(ValueError):
            detect_intervals([0, 1], [.8, .8], confirmation="invented")

    def test_fixed_grid_is_shared_and_calibration_rejects_test(self):
        configs = list(calibration_configs())
        self.assertEqual(len(configs), 84)
        self.assertEqual(len({str(config) for config in configs}), 84)
        with self.assertRaises(ValueError):
            calibrate([{"id": "held-out", "split": "test"}], {})


class SamplingTests(unittest.TestCase):
    def test_even_clip_ties_choose_earlier_sample_and_duplicate_at_boundaries(self):
        times = np.arange(21) / 2
        actual = times[clip_indices(times, center=5, frames=8, fps=2)]
        np.testing.assert_array_equal(actual, np.arange(3, 7, .5))
        self.assertEqual(float(actual.mean()), 4.75)
        first = times[clip_indices(times, center=0)]
        last = times[clip_indices(times, center=10)]
        np.testing.assert_array_equal(first, [0, 0, 0, 0, 0, .5, 1, 1.5])
        np.testing.assert_array_equal(last, [8, 8.5, 9, 9.5, 10, 10, 10, 10])


if __name__ == "__main__":
    unittest.main()
