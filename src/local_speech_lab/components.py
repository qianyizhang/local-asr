from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from local_speech_lab.schemas import ReviewFlag, TranscriptJudgment


class MediaPreparer(ABC):
    """Prepare local media for ASR input."""

    @abstractmethod
    def prepare(self, input_path: Path) -> Path:
        raise NotImplementedError

    def cleanup(self) -> None:
        return None


class AsrBackend(ABC):
    """Recognize text from a prepared audio path."""

    @abstractmethod
    def transcribe(self, audio_path: Path) -> str:
        raise NotImplementedError


class TextPostprocessor(ABC):
    """Transform ASR text into final transcript text."""

    @abstractmethod
    def process(self, text: str) -> str:
        raise NotImplementedError


class ReviewFlagger(ABC):
    """Identify review-worthy transcript content without rewriting it."""

    @abstractmethod
    def flag(self, text: str) -> list[ReviewFlag]:
        raise NotImplementedError


class MetricEvaluator(ABC):
    """Evaluate a transcript against a reference when one exists."""

    @abstractmethod
    def evaluate(self, hypothesis: str, reference: str) -> dict[str, float]:
        raise NotImplementedError


class JudgmentEvaluator(ABC):
    """Judge whether a transcript is semantically useful against a reference."""

    @abstractmethod
    def judge(self, hypothesis: str, reference: str) -> TranscriptJudgment:
        raise NotImplementedError

