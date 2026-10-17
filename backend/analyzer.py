from pathlib import Path

from backend.schemas import AnalysisResult


class AnalyzerNotConfigured(RuntimeError):
    pass


def analyze_video(video_path: Path) -> AnalysisResult:
    """Integration point for the team's local model, with no cloud calls.

    Read the local video and return a validated AnalysisResult including
    timestamps, JPEG evidence and verified rule sources for the UI.
    """
    raise AnalyzerNotConfigured("Локальная модель ещё не подключена")
