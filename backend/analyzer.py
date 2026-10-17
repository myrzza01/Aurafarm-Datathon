from pathlib import Path
import importlib
import importlib.util
import os
import re

from backend.schemas import AnalysisResult


class AnalyzerNotConfigured(RuntimeError):
    pass


DEFAULT_ANALYZER = "backend.cashwatch_adapter:analyze_video"


def analyzer_spec() -> str:
    # An explicitly empty variable disables inference; otherwise the shipped model is used.
    return os.environ.get("ANALYZER_CALLABLE", DEFAULT_ANALYZER).strip()


def analyzer_configured() -> bool:
    spec = analyzer_spec()
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", spec):
        return False
    if spec == "backend.analyzer:analyze_video":
        return False
    if spec == DEFAULT_ANALYZER:
        from backend.cashwatch_adapter import runtime_available
        return runtime_available()
    # Do not import model code or load weights in the API process.
    try:
        return importlib.util.find_spec(spec.split(":", 1)[0].split(".", 1)[0]) is not None
    except (ImportError, ValueError):
        return False


def load_analyzer():
    """Load only a trusted, locally installed module selected by the operator."""
    spec = analyzer_spec()
    if not spec or ":" not in spec:
        raise AnalyzerNotConfigured("Укажите ANALYZER_CALLABLE=модуль:функция для локальной модели")
    module_name, function_name = spec.split(":", 1)
    if module_name == __name__ and function_name == "analyze_video":
        raise AnalyzerNotConfigured("ANALYZER_CALLABLE должен указывать на модуль модели")
    try:
        function = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise AnalyzerNotConfigured("Не удалось загрузить локальный обработчик ANALYZER_CALLABLE") from exc
    if not callable(function):
        raise AnalyzerNotConfigured("ANALYZER_CALLABLE должен указывать на функцию")
    return function


def analyze_video(video_path: Path) -> AnalysisResult:
    """Integration point for the team's local model, with no cloud calls.

    Read the local video and return a validated AnalysisResult including
    timestamps, JPEG evidence and verified rule sources for the UI.
    """
    return AnalysisResult.model_validate(load_analyzer()(video_path))
