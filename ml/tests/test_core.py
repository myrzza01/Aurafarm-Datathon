import base64
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

from cashwatch.data import canonical_name, grouped_split, parse_label, parse_timestamps, preserve_split, resolve_groups
from cashwatch.evaluation import match_events, detect_intervals, calibrate
from cashwatch.inference import event_peaks
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
