import base64
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
import torch
from PIL import Image

from cashwatch.data import canonical_name, grouped_split, parse_label, parse_timestamps, preserve_split, resolve_groups
from cashwatch.evaluation import match_events, detect_intervals, calibrate
from cashwatch.inference import event_peaks, score_frames, VideoAnalyzer
from cashwatch.model import VideoDetector
from cashwatch.train import labeled_centers
from cashwatch.video import clip_indices, decode_sampled, evidence_frames


class DataTests(unittest.TestCase):
    def test_new_data_preserves_dev_and_group_alias(self):
        previous = {"seed": 42, "records": [
            {"id": "a", "group": "ga", "sha256": "1", "split": "train"},
            {"id": "b", "group": "gb", "sha256": "2", "split": "dev"}]}
        records = [dict(r) for r in previous["records"]] + [
            {"id": "c", "group": "gc", "sha256": "3"},
            {"id": "b-copy", "group": "gb", "sha256": "2"}]
        preserve_split(records, previous, 42)
        self.assertEqual([r["split"] for r in records], ["train", "dev", "train", "dev"])
        with self.assertRaises(ValueError):
            preserve_split(records, previous, 43)
        records[0]["group"] = "gb"
        with self.assertRaises(ValueError):
            preserve_split(records, previous, 42)

    def test_event_matching_is_one_to_one(self):
        result = match_events([10, 12], [11], tolerance_sec=5)
        self.assertEqual((result["matched"], result["missed"], result["extra"]), (1, 1, 0))
        result = match_events([10, 20], [11, 19, 40], tolerance_sec=2)
        self.assertEqual(result["errors_sec"], [1, 1])
        self.assertEqual(result["extra"], 1)
        self.assertEqual(match_events([10], [0], 5)["matched"], 0)

    def test_duplicate_and_related_group_union(self):
        records = [
            {"id": "a", "sha256": "hash1", "label": 0, "event_times": [], "duration_sec": 10},
            {"id": "a-copy", "sha256": "hash1", "label": 0, "event_times": [], "duration_sec": 10},
            {"id": "retake", "sha256": "hash3", "label": 1, "event_times": [5], "duration_sec": 12}]
        signatures = {r["id"]: np.zeros((2, 2, 3), dtype=np.uint8) for r in records}
        resolve_groups(records, None, {"a": "session-1", "retake": "session-1"}, signatures)
        self.assertEqual(len({r["group"] for r in records}), 1)
        records[1]["label"] = 1
        with self.assertRaises(ValueError):
            resolve_groups(records, None, {}, signatures)

    def test_event_filter_and_no_test_calibration(self):
        times = np.arange(0, 4, 0.5)
        scores = np.array([0, 0.6, 0.8, 0, 0, 0, 0.9, 0])
        events = detect_intervals(times, scores, threshold=0.5, merge_gap_sec=0.6, min_hits=2)
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0]["start_sec"], events[0]["end_sec"]), (0.5, 1.0))
        with self.assertRaises(ValueError):
            calibrate([{"split": "test"}], {})
    def test_filename_and_timestamp_formats(self):
        self.assertEqual(canonical_name("Х-4.mov"), canonical_name("X-4"))
        self.assertEqual(parse_timestamps("22 секунда; 01:12.5"), [22, 72.5])
        self.assertEqual(parse_label("без нарушения"), 0)
        with self.assertRaises(ValueError):
            parse_timestamps("10-15 секунд")
        with self.assertRaises(ValueError):
            parse_label("maybe")

    def test_split_keeps_groups_and_both_classes(self):
        records = [{"id": str(i), "group": str(i//2), "label": (i//2) % 2} for i in range(12)]
        grouped_split(records)
        train = [r for r in records if r["split"] == "train"]
        val = [r for r in records if r["split"] == "val"]
        self.assertFalse({r["group"] for r in train} & {r["group"] for r in val})
        self.assertEqual({r["label"] for r in train}, {0, 1})
        self.assertEqual({r["label"] for r in val}, {0, 1})

    def test_positive_video_unknown_sections_not_negative(self):
        times = np.arange(0, 60, 0.5)
        centers = labeled_centers({"label": 1, "event_times": [22]}, times)
        self.assertTrue(all(21 <= value <= 23 for value in centers))
        self.assertNotIn(0, centers)

    def test_clip_edges_and_separate_events(self):
        times = np.arange(0, 10, 0.5)
        self.assertTrue(np.all(clip_indices(times, 0) >= 0))
        self.assertTrue(np.all(clip_indices(times, 9.5) < len(times)))
        scores = np.zeros(len(times))
        scores[2:5] = [0.6, 0.9, 0.7]
        scores[14:16] = [0.8, 0.95]
        self.assertEqual(event_peaks(times, scores), [3, 15])
        self.assertEqual(event_peaks(times, np.zeros(len(times))), [])


class VideoTests(unittest.TestCase):
    def test_early_h264_evidence_finishes_in_subprocess(self):
        # Run native decoder teardown out of process so a stuck FFmpeg worker
        # fails with a timeout instead of hanging the entire test suite.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "early-evidence.mov"
            with av.open(str(path), mode="w") as container:
                stream = container.add_stream("libx264", rate=30)
                stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
                stream.options = {"preset": "fast", "bf": "3"}
                for index in range(180):
                    pixels = np.full((64, 96, 3), index % 255, dtype=np.uint8)
                    pixels[:, index % 80:index % 80 + 16] = 255
                    for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            command = (
                "import sys; from cashwatch.video import evidence_frames; "
                "result=evidence_frames(sys.argv[1], [0.25, 0.75]); "
                "assert all(abs(result[t]['timestamp_sec']-t)<0.04 for t in result); "
                "assert all(result[t]['image_base64'] for t in result); print('ok')"
            )
            result = subprocess.run([sys.executable, "-c", command, str(path)],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                timeout=30, check=True)
            self.assertEqual(result.stdout.strip(), "ok")

    def test_decode_pts_and_jpeg_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.mov"
            with av.open(str(path), mode="w") as container:
                stream = container.add_stream("mpeg4", rate=10)
                stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
                for i in range(20):
                    pixels = np.full((48, 64, 3), i*10, dtype=np.uint8)
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            images, times, duration = decode_sampled(path, size=64)
            self.assertEqual(len(images), 4)
            np.testing.assert_allclose(times, [0, 0.5, 1, 1.5], atol=0.01)
            self.assertAlmostEqual(duration, 1.9, places=2)
            evidence = evidence_frames(path, [0.73])[0.73]
            self.assertAlmostEqual(evidence["timestamp_sec"], 0.7, places=2)
            with Image.open(io.BytesIO(base64.b64decode(evidence["image_base64"]))) as image:
                self.assertEqual(image.format, "JPEG")
            with self.assertRaises(FileNotFoundError):
                decode_sampled(Path(folder) / "missing.mov")


class ModelTests(unittest.TestCase):
    def test_point_head_training_preserves_all_legacy_weights_and_outputs(self):
        torch.set_num_threads(2)
        torch.manual_seed(23)
        legacy = VideoDetector(pretrained=False, action_classes=2).eval()
        torch.manual_seed(23)
        model = VideoDetector(pretrained=False, action_classes=2, point_localizer=True).eval()
        original = {key: value.clone() for key, value in legacy.state_dict().items()}
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in original.items()))
        self.assertFalse(any(key.startswith("localizer.") for key in original))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.localizer.parameters():
            parameter.requires_grad_(True)
        features = torch.rand(2, 8, model.encoder.conv_head.out_channels)
        before = model.localize(features).detach().clone()
        optimizer = torch.optim.SGD(model.localizer.parameters(), lr=0.1)
        optimizer.zero_grad()
        model.localize(features).sum().backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.localize(features)))
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in original.items()))
        self.assertTrue(torch.equal(legacy.classify(features), model.classify(features)))
        self.assertTrue(torch.equal(legacy.classify_actions(features), model.classify_actions(features)))
        with self.assertRaises(ValueError):
            legacy.localize(features)

    def test_score_frames_localizer_uses_pixels_and_preserves_legacy_return_values(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        model = VideoDetector(pretrained=False, action_classes=2, point_localizer=True).eval()
        times = np.arange(4) / 2
        dark = np.zeros((4, 64, 64, 3), dtype=np.uint8)
        bright = np.random.default_rng(7).integers(0, 256, dark.shape, dtype=np.uint8)
        binary_only = score_frames(model, dark, times, "cpu", frames=4)
        binary_actions = score_frames(model, dark, times, "cpu", frames=4, return_actions=True)
        all_dark = score_frames(model, dark, times, "cpu", frames=4, return_actions=True, return_localization=True)
        all_bright = score_frames(model, bright, times, "cpu", frames=4, return_localization=True)
        np.testing.assert_array_equal(binary_only, all_dark[0])
        np.testing.assert_array_equal(binary_actions[0], all_dark[0])
        np.testing.assert_array_equal(binary_actions[1], all_dark[1])
        self.assertEqual(all_dark[2].shape, (4,))
        self.assertIsNone(all_bright[1])
        self.assertGreater(float(np.max(np.abs(all_dark[2] - all_bright[2]))), 1e-7)

    def test_format_four_localization_loads_and_gates_event_peaks_without_changing_legacy(self):
        torch.set_num_threads(2)
        postprocess = {"threshold": 0.5, "merge_gap_sec": 1.0, "min_hits": 1,
                       "confirmation": "adjacent_2", "confirmation_max_step_sec": 0.75}
        localization = {"binary_gate_threshold": 0.5, "postprocess": postprocess}
        types = ["cash_counting_omitted", "cash_not_fully_delivered"]
        config = {"fps": 2.0, "image_size": 64, "clip_frames": 8, "head_type": "temporal",
                  "event_taxonomy": {"type_ids": types}, "postprocess": postprocess,
                  "event_localization": localization}
        binary = np.array([.9, .9, .1, .1, .9, .9, .1, .1])
        local = np.array([.1, .1, .99, .99, .6, .95, .99, .99])
        times = np.arange(8) / 2
        actions = np.array([[.99, .01]] * 8)
        actions[5] = [.01, .99]
        with tempfile.TemporaryDirectory() as folder:
            for version in (1, 2, 3, 4):
                with self.subTest(format_version=version):
                    source = VideoDetector(pretrained=False, action_classes=2 if version >= 3 else 0,
                                           point_localizer=version == 4)
                    checkpoint = Path(folder) / f"v{version}.pt"
                    torch.save({"format_version": version, "training_steps": 1,
                                "config": config, "model": source.state_dict()}, checkpoint)
                    analyzer = VideoAnalyzer(checkpoint, device="cpu")
                    self.assertEqual(hasattr(analyzer.model, "localizer"), version == 4)
                    for key, value in source.state_dict().items():
                        self.assertTrue(torch.equal(value, analyzer.model.state_dict()[key]))
                    self.assertEqual(analyzer.type_ids, types if version >= 3 else [])
                    analyzer.last_action_scores = actions
                    analyzer.last_localization_scores = local if version == 4 else None
                    analyzer.last_timings = {"decode_sec": 0., "model_sec": 0.}
                    def evidence(_path, targets, _deadline):
                        return {target: {"timestamp_sec": float(target), "image_base64": "test"} for target in targets}
                    with patch.object(analyzer, "score_video", return_value=(times, binary, 3.5)), \
                            patch("cashwatch.inference.evidence_frames", side_effect=evidence):
                        result, diagnostics = analyzer.analyze(Path(folder) / "any-name.mov")
                    expected = [2.5] if version == 4 else [0., 2.]
                    self.assertEqual([event["timecode_sec"] for event in result["violations"]], expected)
                    self.assertEqual(diagnostics["postprocess"], postprocess)
                    if version == 4:
                        self.assertEqual(result["violations"][0]["type_id"], types[1])
                        self.assertEqual(diagnostics["event_localization"], localization)
                        self.assertEqual(diagnostics["max_score"], .9)
                        overridden = VideoAnalyzer(checkpoint, device="cpu", threshold=.95)
                        self.assertEqual(overridden.event_localization["binary_gate_threshold"], .95)
                        self.assertEqual(overridden.event_localization["postprocess"]["threshold"], .5)
                        self.assertEqual(overridden.config["event_localization"]["binary_gate_threshold"], .5)
                        overridden.last_action_scores = actions
                        overridden.last_localization_scores = local
                        overridden.last_timings = {"decode_sec": 0., "model_sec": 0.}
                        with patch.object(overridden, "score_video", return_value=(times, binary, 3.5)), \
                                patch("cashwatch.inference.evidence_frames", side_effect=evidence):
                            overridden_result, _ = overridden.analyze(Path(folder) / "any-name.mov")
                        self.assertEqual(overridden_result, {"verdict": "no_violation", "violations": []})
                    else:
                        self.assertNotIn("event_localization", diagnostics)

    def test_backward_updates_encoder_and_temporal_head(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        model = VideoDetector(pretrained=False)
        model.configure_finetuning(unfreeze_encoder=True)
        model.train()
        inputs = torch.rand(2, 4, 3, 64, 64)
        output = model(inputs)
        torch.nn.functional.binary_cross_entropy_with_logits(output, torch.tensor([0., 1.])).backward()
        encoder_grad = sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None)
        head_grad = sum(float(p.grad.abs().sum()) for p in model.temporal.parameters() if p.grad is not None)
        self.assertGreater(encoder_grad, 0)
        self.assertGreater(head_grad, 0)
        self.assertEqual(tuple(output.shape), (2,))

    def test_frozen_stage_has_no_encoder_gradients(self):
        model = VideoDetector(pretrained=False)
        model.configure_finetuning()
        model.train()
        features = torch.rand(2, 8, model.encoder.conv_head.out_channels)
        model.classify(features).sum().backward()
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.encoder.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.temporal.parameters()))


if __name__ == "__main__":
    unittest.main()
