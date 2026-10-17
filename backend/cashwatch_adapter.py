"""Application adapter for the team's local binary detector. No cloud inference."""
import importlib.util
import os
from pathlib import Path

SOURCE_COMMIT = "2051a0e008042a3cdb0669371232dc81cab671f0"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "ml" / "weights" / "detector.pt"


def checkpoint_path() -> Path:
    return Path(os.environ.get("CASHWATCH_CHECKPOINT", str(DEFAULT_CHECKPOINT))).expanduser().resolve()


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
        # This trained model has no semantic class -> legal rule mapping.
        # Keep that limitation visible, while preserving actual timestamps and JPEGs.
        event["rule_source"] = {
            "status": "unmapped",
            "reason": "Бинарный детектор выделил фрагмент для проверки человеком. "
                      "Тип действия и применимое правило пока не определены; "
                      "нормативный источник не назначен.",
        }
    return result
