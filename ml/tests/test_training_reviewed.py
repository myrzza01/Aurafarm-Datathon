"""Meaningful guards for joint training; no external weights or video are needed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from cashwatch.data import file_hash, write_json
from cashwatch import train_reviewed as training


class TinyDetector(nn.Module):
    """Exercise real optimization/checkpointing without an image-model download."""
    def __init__(self, head_type="temporal", action_classes=2, **kwargs):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.normalization = nn.LayerNorm(4)
        self.temporal = nn.Linear(4, 1)
        self.action_head = nn.Linear(4, action_classes)

    def configure_finetuning(self, unfreeze_encoder=False):
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(unfreeze_encoder)

    def classify(self, embeddings):
        return self.temporal(self.normalization(embeddings).mean(1)).squeeze(-1)

    def classify_actions(self, embeddings):
        return self.action_head(self.normalization(embeddings).mean(1))


def annotated(identifier, split, label, type_id=None):
    event = {"timecode_sec": 2.0, "type_id": type_id, "type_verified": True,
             "observed_description": "Human target text, never a visual feature."}
    return {"id": identifier, "split": split, "label": label, "group": identifier,
        "sha256": identifier, "path": identifier + ".mov", "cache": identifier + ".npz",
        "events": [event] if label else [], "event_times": [2.0] if label else []}


def manifest_fixture(root):
    records = [annotated("train-a", "train", 1, "a"), annotated("train-b", "train", 1, "b"),
        annotated("train-negative", "train", 0), annotated("dev-a", "dev", 1, "a"),
        annotated("dev-negative", "dev", 0), annotated("test-b", "test", 1, "b"),
        annotated("test-negative", "test", 0)]
    for record in records:
        video, cache = root / record["path"], root / record["cache"]
        video.write_bytes(record["id"].encode())
        np.savez_compressed(cache, images=np.zeros((12, 224, 224, 3), dtype=np.uint8),
            times=np.arange(12) / 2, fps=2.0, size=224, duration=5.5)
        record.update(path=str(video), cache=str(cache), sha256=file_hash(video), cache_sha256=file_hash(cache))
    manifest = {"fps": 2.0, "image_size": 224, "clip_frames": 8, "positive_radius_sec": 1.0,
        "records": records, "event_taxonomy": {"status": "human_action_notes_confirmed", "type_ids": ["a", "b"]},
        "independent_test": False, "group_review": {"status": "pending"}}
    path = root / "manifest.json"
    write_json(path, manifest)
    return manifest, path


class ActionTargetTests(unittest.TestCase):
    def setUp(self):
        self.config = {"fps": 2.0, "clip_frames": 8, "event_taxonomy": {"type_ids": ["a", "b"]}}
        self.features = torch.arange(48, dtype=torch.float32).reshape(12, 4)
        self.times = np.arange(12) / 2

    def test_human_text_filename_and_alias_metadata_do_not_enter_visual_features(self):
        first = annotated("video-a", "train", 1, "a")
        second = copy.deepcopy(first)
        second["id"] = "completely-different-name"
        second["path"] = "renamed.mov"
        second["events"][0]["observed_description"] = "Different private annotation text"
        second["alias_annotations"] = [{"id": str(index)} for index in range(30)]
        x1, y1, w1, _ = training.action_tensors([first], {first["id"]: (self.features, self.times)}, self.config)
        x2, y2, w2, _ = training.action_tensors([second], {second["id"]: (self.features, self.times)}, self.config)
        torch.testing.assert_close(x1, x2)
        torch.testing.assert_close(y1, y2)
        torch.testing.assert_close(w1, w2)
        self.assertEqual(len(y2), 5)

    def test_distinct_events_on_one_video_have_their_own_localized_class_targets(self):
        record = annotated("one-sha", "train", 1, "a")
        record["events"].append({"timecode_sec": 4.5, "type_id": "b", "type_verified": True})
        _, labels, weights, sources = training.action_tensors([record], {record["id"]: (self.features, self.times)}, self.config)
        self.assertEqual(labels.tolist(), [0] * 5 + [1] * 5)
        self.assertEqual({source["timecode_sec"] for source in sources}, {2.0, 4.5})
        self.assertAlmostEqual(float(weights[labels == 0].sum()), float(weights[labels == 1].sum()))

    def test_unconfirmed_action_notes_are_not_supervised_targets(self):
        record = annotated("unconfirmed", "train", 1, "a")
        record["events"][0]["type_verified"] = False
        clips, labels, weights, sources = training.action_tensors([record], {record["id"]: (self.features, self.times)}, self.config)
        self.assertEqual((clips, labels, weights, sources), (None, None, None, []))

    def test_classification_at_truth_is_explicitly_conditioned_on_annotation_centers(self):
        class AlwaysA(nn.Module):
            def classify_actions(self, clips):
                return torch.tensor([[8.0, -8.0]]).repeat(len(clips), 1)
        record = annotated("positive", "dev", 1, "a")
        result = training.action_metrics(AlwaysA(), [record], {record["id"]: (self.features, self.times)}, self.config, torch.device("cpu"))
        self.assertEqual(result["accuracy"], 1.0)
        self.assertIn("excludes detector misses", result["scope"])
        self.assertEqual(result["per_class"]["b"]["support"], 0)


class ReviewedTrainingGuardTests(unittest.TestCase):
    def test_cross_split_byte_overlap_rejected_before_pixel_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, path = manifest_fixture(root)
            manifest["records"][-1]["sha256"] = manifest["records"][0]["sha256"]
            write_json(path, manifest)
            with patch.object(training, "load_caches") as loader:
                with self.assertRaisesRegex(ValueError, "Partition leakage: sha256"):
                    training.train_reviewed(path, root / "run", root / "unused.pt", epochs=1, device_name="cpu")
                loader.assert_not_called()

    def test_each_deployed_semantic_class_must_be_present_in_train(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, path = manifest_fixture(root)
            manifest["event_taxonomy"]["type_ids"].append("not-trained")
            write_json(path, manifest)
            with patch.object(training, "load_caches") as loader:
                with self.assertRaisesRegex(ValueError, "Every deployed action class"):
                    training.train_reviewed(path, root / "run", root / "unused.pt", epochs=1, device_name="cpu")
                loader.assert_not_called()

    def test_action_confirmation_not_inferred_from_nonempty_text(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, path = manifest_fixture(root)
            manifest["event_taxonomy"]["status"] = "unverified_notes"
            write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "explicit human confirmation"):
                training.train_reviewed(path, root / "run", root / "unused.pt", epochs=1, device_name="cpu")

    def test_reserved_pixels_are_loaded_only_after_selection_and_checkpoint_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, path = manifest_fixture(root)
            output = root / "run"
            source = root / "encoder.pt"
            torch.save({"model_name": training.MODEL_NAME, "encoder": TinyDetector().encoder.state_dict()}, source)
            progress = {"calibrated": False, "test_read": False, "frozen_hash": None, "loaded_ids": []}
            original_calibrate = training.calibrate

            def load(records):
                progress["loaded_ids"].extend(r["id"] for r in records)
                if any(r["split"] == "test" for r in records):
                    self.assertTrue(progress["calibrated"], "Test pixels read before dev calibration")
                    self.assertTrue((output / "best.pt").is_file())
                    progress["frozen_hash"] = file_hash(output / "best.pt")
                    progress["test_read"] = True
                else:
                    self.assertFalse(progress["test_read"])
                return {}

            def encode(model, records, caches, device):
                return {r["id"]: (torch.arange(48, dtype=torch.float32).reshape(12, 4) / 50 + r["label"] * 0.2,
                    np.arange(12) / 2) for r in records}

            def calibrate(records, timelines, tolerance):
                self.assertTrue(all(r["split"] == "dev" for r in records))
                self.assertFalse(any(identifier.startswith("test-") for identifier in timelines))
                result = original_calibrate(records, timelines, tolerance)
                progress["calibrated"] = True
                return result

            with patch.object(training, "VideoDetector", TinyDetector), patch.object(training, "load_caches", side_effect=load), \
                    patch.object(training, "encode_caches", side_effect=encode), patch.object(training, "calibrate", side_effect=calibrate):
                report = training.train_reviewed(path, output, source, epochs=1, device_name="cpu")
            self.assertTrue(progress["test_read"])
            self.assertEqual(progress["frozen_hash"], file_hash(output / "best.pt"))
            self.assertEqual(len(progress["loaded_ids"]), 7)
            self.assertTrue(report["encoder_frozen_verified"])
            self.assertEqual(set(report["partitions"]), {"train", "dev", "test"})
            self.assertGreater(sum(value for name, value in report["head_weight_changes_l1"].items() if name.startswith("action_head")), 0)
            with self.assertRaises(FileExistsError):
                training.train_reviewed(path, output, source, epochs=1, device_name="cpu")


if __name__ == "__main__":
    unittest.main()
