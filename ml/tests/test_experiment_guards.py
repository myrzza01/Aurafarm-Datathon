import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cashwatch.data import file_hash, write_json
from cashwatch.train import evaluate_checkpoint
from tools.run_expanded_cycle import save_locked_selection, verify_locked_weights


class ExperimentGuardTests(unittest.TestCase):
    def test_direct_evaluation_cannot_bypass_pending_group_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "manifest.json", {"records": [{"split": "test"}],
                                               "group_review": {"status": "pending"}})
            with patch("cashwatch.train.VideoAnalyzer") as analyzer:
                with self.assertRaisesRegex(ValueError, "review is pending"):
                    evaluate_checkpoint("unused.pt", root / "manifest.json", "test", root / "result.json")
                analyzer.assert_not_called()
            self.assertFalse((root / "result.json").exists())

    def test_changed_video_rejected_before_model_or_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "input.mov"
            video.write_bytes(b"original")
            record = {"split": "dev", "sha256": file_hash(video), "path": str(video)}
            write_json(root / "manifest.json", {"records": [record]})
            video.write_bytes(b"replaced")
            with patch("cashwatch.train.VideoAnalyzer") as analyzer:
                with self.assertRaisesRegex(ValueError, "bytes changed"):
                    evaluate_checkpoint("unused.pt", root / "manifest.json", "dev", root / "result.json")
                analyzer.assert_not_called()

    def test_annotation_revision_blocks_test_even_with_confirmed_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "manifest.json", {"records": [{"split": "test"}],
                "group_review": {"status": "confirmed"}, "label_revisions_pending_review": True})
            with patch("cashwatch.train.VideoAnalyzer") as analyzer:
                with self.assertRaisesRegex(ValueError, "invalidated"):
                    evaluate_checkpoint("unused.pt", root / "manifest.json", "test", root / "result.json")
                analyzer.assert_not_called()

    def test_failed_selection_change_does_not_overwrite_locked_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, candidate = root / "original.pt", root / "candidate.pt"
            original.write_bytes(b"original")
            candidate.write_bytes(b"candidate")
            selection = {"selected_model": "linear", "selected_sha256": file_hash(original),
                         "plan_sha256": "fixed", "checkpoints": {}}
            save_locked_selection(root, original, {"head_type": "linear"}, selection)
            before = {p.name: p.read_bytes() for p in root.glob("selected*")}
            changed = {**selection, "selected_model": "temporal", "selected_sha256": file_hash(candidate)}
            with self.assertRaisesRegex(ValueError, "Cannot change"):
                save_locked_selection(root, candidate, {"head_type": "temporal"}, changed)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.glob("selected*")})


if __name__ == "__main__":
    unittest.main()
