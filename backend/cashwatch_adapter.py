"""Application adapter for locally trained detectors. No cloud inference."""
import importlib.util
import functools
import hashlib
import json
import os
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "ml" / "weights"
DEFAULT_CHECKPOINT = WEIGHTS_DIR / "event_localizer_detector.pt"


def checkpoint_path() -> Path:
    legacy = WEIGHTS_DIR / "detector.pt"
    default = next((path for path in (
        DEFAULT_CHECKPOINT, WEIGHTS_DIR / "hard_negative_detector.pt", WEIGHTS_DIR / "reviewed_detector.pt", legacy
    ) if path.is_file()), legacy)
    return Path(os.environ.get("CASHWATCH_CHECKPOINT", str(default))).expanduser().resolve()


@functools.lru_cache(maxsize=8)
def _checkpoint_hash(path: str, size: int, modified_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyzer_kind() -> str:
    # Attested file content, not its filename, determines advertised capability.
    # Unknown custom weights stay explicit; never import Torch in a health request.
    path = checkpoint_path()
    try:
        stat = path.stat()
        digest = _checkpoint_hash(str(path), stat.st_size, stat.st_mtime_ns)
        for report_name in ("event_localizer_report.json", "hard_negative_report.json", "reviewed_report.json"):
            try:
                report = json.loads((WEIGHTS_DIR / report_name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(report, dict) and digest == report.get("checkpoint_sha256"):
                return "semantic"
        source = json.loads((WEIGHTS_DIR.parent / "SOURCE.json").read_text(encoding="utf-8"))
        if digest in source.get("sha256", {}).values():
            return "binary"
    except (OSError, ValueError):
        pass
    return "custom_checkpoint"


def runtime_available() -> bool:
    return checkpoint_path().is_file() and all(
        importlib.util.find_spec(name) is not None for name in ("torch", "torchvision", "timm", "numpy", "av", "PIL")
    )


def analyze_video(video_path: Path) -> dict:
    # This import and the model weights stay in the isolated job process.
    # timm is called with pretrained=False in the supplied implementation.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from ml.cashwatch.inference import VideoAnalyzer

    analyzer = VideoAnalyzer(checkpoint=checkpoint_path(), device=os.environ.get("CASHWATCH_DEVICE", "auto"))
    result, _diagnostics = analyzer.analyze(video_path, timeout_sec=240)
    for event in result["violations"]:
        if "rule_source" in event:
            continue
        # Legacy binary checkpoints still have no lawful semantic mapping.
        event["rule_source"] = {
            "status": "unmapped",
            "reason": "Бинарный детектор выделил фрагмент для проверки человеком. "
                      "Тип действия и применимое правило пока не определены; "
                      "нормативный источник не назначен.",
        }
    return result
