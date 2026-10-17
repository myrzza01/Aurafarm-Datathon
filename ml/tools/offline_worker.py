"""Fresh-process runtime verification. Never imports training modules."""
import argparse
import base64
import builtins
import io
import json
import os
from pathlib import Path
import socket
import sys
import time
from unittest.mock import patch


def run():
    parser = argparse.ArgumentParser()
    for name in ["bundle", "video", "alias", "longest", "output"]:
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.bundle).resolve()))
    counters = {"network_attempts": 0, "training_artifact_reads": 0}
    real_open = builtins.open

    def deny_network(*args, **kwargs):
        counters["network_attempts"] += 1
        raise RuntimeError("Network is disabled for this verification process")

    def guarded_open(file, mode="r", *pos, **kw):
        if isinstance(file, (str, os.PathLike)) and ("r" in mode or "+" in mode):
            path = Path(file)
            if path.suffix.lower() in {".xlsx", ".xls", ".csv", ".npz"} or path.name in {"manifest.json", "split_registry.json"}:
                counters["training_artifact_reads"] += 1
                raise RuntimeError("Reading training data is disabled")
        return real_open(file, mode, *pos, **kw)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HOME"] = str(Path(args.output).parent / "empty_hf_cache")
    with patch.object(socket.socket, "connect", deny_network), \
         patch.object(socket.socket, "connect_ex", deny_network), \
         patch.object(socket, "create_connection", deny_network), \
         patch.object(socket, "getaddrinfo", deny_network), \
         patch("builtins.open", guarded_open), patch("io.open", guarded_open):
        import numpy as np
        from PIL import Image
        from cashwatch.inference import VideoAnalyzer, analyze_video
        from cashwatch.video import iter_frames, frame_image
        assert "cashwatch.data" not in sys.modules and "cashwatch.train" not in sys.modules
        started = time.monotonic()
        analyzer = VideoAnalyzer(Path(args.bundle) / "weights" / "detector.pt")
        load_sec = time.monotonic()-started
        times, scores, _ = analyzer.score_video(args.video)
        alias_times, alias_scores, _ = analyzer.score_video(args.alias)
        np.testing.assert_array_equal(times, alias_times)
        np.testing.assert_array_equal(scores, alias_scores)
        result, benchmark = analyzer.analyze(args.longest)
        analyze_video(args.alias)  # Public API accepts just the new video path.
        invalid_video_raises = False
        try:
            analyzer.analyze(str(Path(args.alias).with_name("missing.mov")))
        except FileNotFoundError:
            invalid_video_raises = True
        assert invalid_video_raises
        assert set(result) == {"verdict", "violations"}
        assert result["verdict"] in {"violation", "no_violation"}
        expected = {}
        for event in result["violations"]:
            assert set(event) == {"type", "timecode_sec", "description", "evidence_frames"}
            assert 1 <= len(event["evidence_frames"]) <= 3
            assert event["timecode_sec"] in [frame["timestamp_sec"] for frame in event["evidence_frames"]]
            for frame in event["evidence_frames"]:
                decoded = Image.open(io.BytesIO(base64.b64decode(frame["image_base64"], validate=True)))
                assert decoded.format == "JPEG"
                expected[frame["timestamp_sec"]] = np.asarray(decoded.convert("RGB")).astype(float)
        matches, max_pixel_error = set(), 0.0
        for timestamp, frame in iter_frames(args.longest):
            for target in expected:
                if abs(timestamp-target) <= 0.00006:
                    source = frame_image(frame)
                    source.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                    pixels = np.asarray(source).astype(float)
                    assert pixels.shape == expected[target].shape
                    error = float(np.abs(pixels-expected[target]).mean())
                    assert error < 15, "Evidence does not match the decoded source frame"
                    max_pixel_error = max(max_pixel_error, error)
                    matches.add(target)
        assert len(matches) == len(expected)
        assert not counters["network_attempts"] and not counters["training_artifact_reads"]
        report = {"separate_process": True, "runtime_only_bundle": True,
            "training_modules_imported": False, **counters, "rename_max_score_difference": float(abs(scores-alias_scores).max()),
            "public_path_only_function_passed": True, "invalid_video_raises": invalid_video_raises,
            "api_schema_passed": True, "evidence_frames_verified": len(matches),
            "evidence_max_mean_pixel_error": max_pixel_error, "model_load_sec": load_sec,
            "longest_video_benchmark": {key: value for key, value in benchmark.items() if key != "intervals"},
            "network_guard": "Python socket calls denied before ML imports; not an OS firewall test",
            "full_http_request_sec": None}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"offline_passed": True, **counters, "evidence_frames_verified": len(matches),
        "longest_video_duration_sec": benchmark["video_duration_sec"], "model_sec": benchmark["model_sec"],
        "full_local_analysis_sec": benchmark["elapsed_sec"], "peak_vram_allocated_mb": benchmark["peak_vram_allocated_mb"]}), flush=True)


if __name__ == "__main__":
    run()
