"""Application adapter for locally trained detectors. No cloud inference."""
import importlib.util
import functools
import hashlib
import json
import os
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "ml" / "weights"
DEFAULT_CHECKPOINT = WEIGHTS_DIR / "reviewed_detector.pt"


def checkpoint_path() -> Path:
    default = DEFAULT_CHECKPOINT if DEFAULT_CHECKPOINT.is_file() else WEIGHTS_DIR / "detector.pt"
    return Path(os.environ.get("CASHWATCH_CHECKPOINT", str(default))).expanduser().resolve()


@functools.lru_cache(maxsize=8)
def _checkpoint_hash(path: str, size: int, modified_ns: int) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def analyzer_kind() -> str:
    # Attested file content, not its filename, determines advertised capability.
    # Unknown custom weights stay explicit; never import Torch in a health request.
    path = checkpoint_path()
    try:
        stat = path.stat()
        digest = _checkpoint_hash(str(path), stat.st_size, stat.st_mtime_ns)
        report_path = WEIGHTS_DIR / "reviewed_report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if digest == report.get("checkpoint_sha256"):
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
