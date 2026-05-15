from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

from local_speech_lab.components import (
    AsrBackend,
    JudgmentEvaluator,
    MediaPreparer,
    MetricEvaluator,
    ReviewFlagger,
)
from local_speech_lab.llm_judge import LlamaCppCliSemanticJudge, TransformersSemanticJudge
from local_speech_lab.medical_text import (
    apply_corrections,
    flag_blacklisted_terms,
    load_corrections,
    load_lines,
    strip_sensevoice_tags,
)
from local_speech_lab.metrics import transcript_metrics
from local_speech_lab.schemas import ComponentSpec, ReviewFlag, TranscriptJudgment, TranscriptResult


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


class LocalMediaPreparer(MediaPreparer):
    def __init__(self, sample_rate: int = 16000, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def prepare(self, input_path: Path) -> Path:
        if input_path.suffix.lower() not in VIDEO_SUFFIXES:
            return input_path

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("Video input requires ffmpeg, but ffmpeg was not found.")

        self._temp_dir = tempfile.TemporaryDirectory(prefix="local-asr-")
        audio_path = Path(self._temp_dir.name) / f"{input_path.stem}.wav"
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(input_path),
                "-vn",
                "-ac",
                str(self.channels),
                "-ar",
                str(self.sample_rate),
                str(audio_path),
            ],
            check=True,
        )
        return audio_path

    def cleanup(self) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None


class FunAsrBackend(AsrBackend):
    def __init__(self, model: str, hotwords_path: Path | None = None) -> None:
        self.model = model
        self.hotwords_path = hotwords_path
        self._recognizer: Any | None = None

    @property
    def recognizer(self) -> Any:
        if self._recognizer is None:
            from funasr import AutoModel

            self._recognizer = AutoModel(
                model=self.model,
                trust_remote_code=True,
                disable_update=True,
                disable_pbar=True,
                log_level="ERROR",
            )
        return self._recognizer

    def transcribe(self, audio_path: Path) -> str:
        hotwords = ""
        if self.hotwords_path is not None:
            hotwords = " ".join(load_lines(self.hotwords_path))

        result = self.recognizer.generate(input=str(audio_path), hotword=hotwords)
        if result and isinstance(result, list):
            return str(result[0].get("text", ""))
        return ""


class WhisperCppBackend(AsrBackend):
    def __init__(
        self,
        model_path: Path,
        executable: str = "whisper-cpp",
        language: str = "zh",
        threads: int = 4,
    ) -> None:
        self.model_path = model_path
        self.executable = executable
        self.language = language
        self.threads = threads

    def transcribe(self, audio_path: Path) -> str:
        exe = shutil.which(self.executable)
        if not exe:
            raise RuntimeError(f"whisper-cpp executable '{self.executable}' not found.")
        with tempfile.TemporaryDirectory(prefix="local-asr-whisper-") as tmpdir:
            tmp_audio = Path(tmpdir) / audio_path.name
            shutil.copy2(audio_path, tmp_audio)
            subprocess.run(
                [
                    exe,
                    "--model", str(self.model_path),
                    "--file", str(tmp_audio),
                    "--language", self.language,
                    "--threads", str(self.threads),
                    "--output-txt",
                    "--no-prints",
                ],
                check=True,
            )
            txt_path = Path(tmpdir) / (audio_path.name + ".txt")
            if txt_path.exists():
                return txt_path.read_text(encoding="utf-8").strip()
        return ""


class SenseVoiceTagCleaner:
    def process(self, text: str) -> str:
        return strip_sensevoice_tags(text)


class LiteralCorrectionPostprocessor:
    def __init__(self, corrections_path: Path) -> None:
        self.corrections_path = corrections_path

    def process(self, text: str) -> str:
        return apply_corrections(text, load_corrections(self.corrections_path))


class BlacklistFlagger(ReviewFlagger):
    def __init__(self, blacklist_path: Path) -> None:
        self.blacklist_path = blacklist_path

    def flag(self, text: str) -> list[ReviewFlag]:
        return [
            ReviewFlag(kind="blacklist", value=term)
            for term in flag_blacklisted_terms(text, load_lines(self.blacklist_path))
        ]


class TranscriptMetricEvaluator(MetricEvaluator):
    def evaluate(self, hypothesis: str, reference: str) -> dict[str, float]:
        return transcript_metrics(hypothesis, reference)


def build_media_preparer(spec: ComponentSpec) -> MediaPreparer:
    if spec.name == "local":
        return LocalMediaPreparer(
            sample_rate=int(spec.options.get("sample_rate", 16000)),
            channels=int(spec.options.get("channels", 1)),
        )
    raise ValueError(f"Unknown media component: {spec.name}")


def build_asr_backend(spec: ComponentSpec) -> AsrBackend:
    if spec.name == "funasr":
        model = str(spec.options["model"])
        hotwords = spec.options.get("hotwords_path")
        return FunAsrBackend(model=model, hotwords_path=Path(hotwords) if hotwords else None)
    if spec.name == "whisper_cpp":
        return WhisperCppBackend(
            model_path=Path(str(spec.options["model_path"])),
            executable=str(spec.options.get("executable", "whisper-cpp")),
            language=str(spec.options.get("language", "zh")),
            threads=int(spec.options.get("threads", 4)),
        )
    raise ValueError(f"Unknown ASR component: {spec.name}")


def build_postprocessor(spec: ComponentSpec) -> SenseVoiceTagCleaner | LiteralCorrectionPostprocessor:
    if spec.name == "sensevoice_tags":
        return SenseVoiceTagCleaner()
    if spec.name == "literal_corrections":
        return LiteralCorrectionPostprocessor(Path(spec.options["corrections_path"]))
    raise ValueError(f"Unknown postprocess component: {spec.name}")


def build_review_flagger(spec: ComponentSpec) -> ReviewFlagger:
    if spec.name == "blacklist":
        return BlacklistFlagger(Path(spec.options["blacklist_path"]))
    raise ValueError(f"Unknown review component: {spec.name}")


def build_metric_evaluator(spec: ComponentSpec) -> MetricEvaluator:
    if spec.name == "transcript":
        return TranscriptMetricEvaluator()
    raise ValueError(f"Unknown metric component: {spec.name}")


def build_judgment_evaluator(spec: ComponentSpec) -> JudgmentEvaluator:
    if spec.name == "semantic_llm":
        provider = str(spec.options.get("provider", "qwen"))
        if provider in {"qwen", "gemma"}:
            runtime = str(spec.options.get("runtime", "transformers"))
            model_path = Path(spec.options.get("model_path", f"models/{provider}_judge"))
            if runtime == "llama_cpp":
                return LlamaCppCliSemanticJudge(
                    provider=provider,
                    model_path=model_path,
                    executable=str(spec.options.get("executable", "llama-cli")),
                    max_tokens=int(spec.options.get("max_new_tokens", 256)),
                    temperature=float(spec.options.get("temperature", 0.0)),
                )
            if runtime != "transformers":
                raise ValueError(f"Unknown semantic LLM judge runtime: {runtime}")
            return TransformersSemanticJudge(
                provider=provider,
                model_path=model_path,
                max_new_tokens=int(spec.options.get("max_new_tokens", 256)),
            )
        raise ValueError(f"Unknown semantic LLM judge provider: {provider}")
    raise ValueError(f"Unknown judge component: {spec.name}")


def judgment_metrics(judgment: TranscriptJudgment) -> dict[str, float]:
    if judgment.error is not None:
        return {}
    prefix = f"judge_{judgment.provider}"
    metrics: dict[str, float] = {}
    if judgment.semantic_equivalent is not None:
        metrics[f"{prefix}_semantic_equivalent"] = float(judgment.semantic_equivalent)
    if judgment.useful is not None:
        metrics[f"{prefix}_useful"] = float(judgment.useful)
    if judgment.score is not None:
        metrics[f"{prefix}_score"] = judgment.score
    return metrics


class OfflinePipeline:
    def __init__(
        self,
        media_preparer: MediaPreparer,
        asr_backend: AsrBackend,
        postprocessors: list[SenseVoiceTagCleaner | LiteralCorrectionPostprocessor],
        review_flaggers: list[ReviewFlagger],
        metric_evaluators: list[MetricEvaluator],
        judgment_evaluators: list[JudgmentEvaluator],
    ) -> None:
        self.media_preparer = media_preparer
        self.asr_backend = asr_backend
        self.postprocessors = postprocessors
        self.review_flaggers = review_flaggers
        self.metric_evaluators = metric_evaluators
        self.judgment_evaluators = judgment_evaluators

    def run_sample(
        self,
        sample_id: str,
        input_path: Path,
        reference_text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TranscriptResult:
        started = perf_counter()
        try:
            prepared_path = self.media_preparer.prepare(input_path)
            raw_text = self.asr_backend.transcribe(prepared_path)
            text = raw_text
            for postprocessor in self.postprocessors:
                text = postprocessor.process(text)

            flags: list[ReviewFlag] = []
            for flagger in self.review_flaggers:
                flags.extend(flagger.flag(text))

            metrics: dict[str, float] = {}
            judgments: list[TranscriptJudgment] = []
            if reference_text is not None:
                for evaluator in self.metric_evaluators:
                    metrics.update(evaluator.evaluate(text, reference_text))
                for evaluator in self.judgment_evaluators:
                    judgment = evaluator.judge(text, reference_text)
                    judgments.append(judgment)
                    metrics.update(judgment_metrics(judgment))

            return TranscriptResult(
                sample_id=sample_id,
                input_path=input_path,
                raw_text=raw_text,
                text=text,
                reference_text=reference_text,
                flags=flags,
                judgments=judgments,
                metrics=metrics,
                latency_seconds=perf_counter() - started,
                metadata=metadata or {},
            )
        except Exception as exc:
            return TranscriptResult(
                sample_id=sample_id,
                input_path=input_path,
                reference_text=reference_text,
                latency_seconds=perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
                metadata=metadata or {},
            )
        finally:
            self.media_preparer.cleanup()


def build_offline_pipeline(specs: list[ComponentSpec]) -> OfflinePipeline:
    media_specs = [spec for spec in specs if spec.kind == "media"]
    asr_specs = [spec for spec in specs if spec.kind == "asr"]
    if len(media_specs) != 1:
        raise ValueError("Pipeline config must include exactly one media component.")
    if len(asr_specs) != 1:
        raise ValueError("Pipeline config must include exactly one ASR component.")

    return OfflinePipeline(
        media_preparer=build_media_preparer(media_specs[0]),
        asr_backend=build_asr_backend(asr_specs[0]),
        postprocessors=[build_postprocessor(spec) for spec in specs if spec.kind == "postprocess"],
        review_flaggers=[build_review_flagger(spec) for spec in specs if spec.kind == "review"],
        metric_evaluators=[build_metric_evaluator(spec) for spec in specs if spec.kind == "metric"],
        judgment_evaluators=[build_judgment_evaluator(spec) for spec in specs if spec.kind == "judge"],
    )
