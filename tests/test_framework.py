from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from local_speech_lab.benchmark import summarize_results
from local_speech_lab.framework import WhisperCppBackend
from local_speech_lab.io import load_pipeline_config, load_scenario_manifest
from local_speech_lab.medical_text import apply_corrections, flag_blacklisted_terms, strip_sensevoice_tags
from local_speech_lab.metrics import character_error_rate, word_error_rate
from local_speech_lab.schemas import PipelineConfig, ScenarioManifest, ScenarioSample, TranscriptResult


def test_transcript_metrics() -> None:
    assert character_error_rate("二甲双胍", "二甲双瓜") == 0.25
    assert word_error_rate("hello world", "hello there") == 0.5
    assert character_error_rate("", "") == 0.0


def test_medical_text_processing() -> None:
    cleaned = strip_sensevoice_tags("<|zh|><|NEUTRAL|>患者需要服用二甲双瓜")
    corrected = apply_corrections(cleaned, {"二甲双瓜": "二甲双胍"})
    assert corrected == "患者需要服用二甲双胍"
    assert flag_blacklisted_terms(corrected, ["二甲双胍"]) == ["二甲双胍"]


def test_load_jsonl_manifest() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "scenario.jsonl"
        path.write_text(
            '{"id":"a","input_path":"data/audio/a.wav","reference_text":"你好"}\n',
            encoding="utf-8",
        )
        manifest = load_scenario_manifest(path)

    assert manifest.name == "scenario"
    assert manifest.samples[0].id == "a"
    assert manifest.samples[0].reference_text == "你好"


def test_pipeline_config_extends_base_components() -> None:
    config = load_pipeline_config("configs/pipelines/zh_medical_funasr_qwen_small_judge.yaml")

    assert config.name == "zh_medical_funasr_qwen_small_judge"
    assert [component.kind for component in config.components] == [
        "media",
        "asr",
        "postprocess",
        "postprocess",
        "review",
        "metric",
        "judge",
    ]
    assert config.components[-1].options["model_path"] == "models/qwen3_0_6b"


def test_summarize_results() -> None:
    config = PipelineConfig(name="pipe", components=[])
    scenario = ScenarioManifest(
        name="scenario",
        samples=[ScenarioSample(id="a", input_path=Path("a.wav"))],
    )
    result = TranscriptResult(
        sample_id="a",
        input_path=Path("a.wav"),
        text="你好",
        metrics={"cer": 0.25, "wer": 0.5},
        latency_seconds=2.0,
    )

    summary = summarize_results(
        run_id="run",
        pipeline_config=config,
        scenario=scenario,
        results=[result],
        output_dir=Path("outputs/benchmarks/run"),
    )

    assert summary.succeeded == 1
    assert summary.failed == 0
    assert summary.metrics["mean_cer"] == 0.25
    assert summary.metrics["mean_wer"] == 0.5
    assert summary.metrics["mean_latency_seconds"] == 2.0


def test_summarize_results_with_metric_weights() -> None:
    config = PipelineConfig(name="pipe", components=[])
    scenario = ScenarioManifest(
        name="scenario",
        samples=[ScenarioSample(id="a", input_path=Path("a.wav"))],
        metric_weights={"cer": 2.0, "wer": 1.0},
    )
    result = TranscriptResult(
        sample_id="a",
        input_path=Path("a.wav"),
        text="你好",
        metrics={"cer": 0.25, "wer": 0.5},
        latency_seconds=1.0,
    )
    summary = summarize_results(
        run_id="run",
        pipeline_config=config,
        scenario=scenario,
        results=[result],
        output_dir=Path("outputs/benchmarks/run"),
    )
    # weighted_mean = (2.0*0.25 + 1.0*0.5) / 3.0 = 1.0/3.0
    assert "weighted_mean" in summary.metrics
    assert abs(summary.metrics["weighted_mean"] - (2.0 * 0.25 + 1.0 * 0.5) / 3.0) < 1e-9
    assert summary.metrics["mean_cer"] == 0.25
    assert summary.metrics["mean_wer"] == 0.5


def test_whisper_cpp_backend_missing_executable() -> None:
    backend = WhisperCppBackend(
        model_path=Path("models/ggml-large-v3-turbo.bin"),
        executable="__nonexistent_whisper_cpp__",
    )
    with pytest.raises(RuntimeError, match="not found"):
        backend.transcribe(Path("data/audio/test.wav"))


def test_whisper_pipeline_config_loads() -> None:
    config = load_pipeline_config("configs/pipelines/zh_medical_whisper.yaml")
    assert config.name == "zh_medical_whisper"
    asr = next(c for c in config.components if c.kind == "asr")
    assert asr.name == "whisper_cpp"
    assert asr.options["model_path"] == "models/ggml-large-v3-turbo.bin"
    assert "sensevoice_tags" not in [c.name for c in config.components]
