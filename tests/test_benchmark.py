from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from local_speech_lab.benchmark import summarize_results
from local_speech_lab.io import load_scenario_manifest
from local_speech_lab.schemas import (
    BenchmarkSummary,
    PipelineConfig,
    ScenarioManifest,
    ScenarioSample,
    TranscriptResult,
)


def _summary(results: list[TranscriptResult], weights: dict[str, float]) -> BenchmarkSummary:
    return summarize_results(
        run_id="run",
        pipeline_config=PipelineConfig(name="pipe", components=[]),
        scenario=ScenarioManifest(
            name="scenario",
            samples=[
                ScenarioSample(
                    id=result.sample_id,
                    input_path=result.input_path,
                    reference_text=result.reference_text,
                )
                for result in results
            ],
            metric_weights=weights,
        ),
        results=results,
        output_dir=Path("outputs/benchmarks/run"),
    )


@pytest.mark.parametrize("case", ["empty", "failed", "missing", "no_reference"])
def test_weighted_mean_without_observations_is_absent(case: str) -> None:
    results = (
        []
        if case == "empty"
        else [
            TranscriptResult(
                sample_id="a",
                input_path=Path("a.wav"),
                text="hello",
                reference_text=None if case == "no_reference" else "hello",
                metrics={"cer": 0.0, "wer": 0.0} if case == "failed" else {},
                latency_seconds=2.0,
                error="ASR failed" if case == "failed" else None,
            )
        ]
    )

    summary = _summary(results, {"cer": 2.0, "wer": 1.0})

    assert "weighted_mean" not in summary.metrics
    assert "mean_cer" not in summary.metrics
    assert "mean_wer" not in summary.metrics
    assert summary.total_samples == len(results)
    assert summary.failed == int(case == "failed")
    assert summary.succeeded == int(case in {"missing", "no_reference"})
    if summary.succeeded:
        assert summary.metrics["mean_latency_seconds"] == 2.0


@pytest.mark.parametrize("missing_key", ["wer", "unknown_metric", "mean_cer"])
def test_missing_positive_weight_metric_does_not_become_zero(missing_key: str) -> None:
    result = TranscriptResult(sample_id="a", input_path=Path("a.wav"), metrics={"cer": 0.6})

    summary = _summary([result], {"cer": 1.0, missing_key: 1.0})

    assert summary.metrics == {"mean_cer": 0.6}
    assert result.metrics == {"cer": 0.6}


def test_weighted_mean_keeps_observed_sample_means_and_failure_counts() -> None:
    results = [
        TranscriptResult(sample_id="a", input_path=Path("a.wav"), metrics={"cer": 0.2}),
        TranscriptResult(sample_id="b", input_path=Path("b.wav"), metrics={"cer": 0.6, "wer": 0.8}),
        TranscriptResult(sample_id="c", input_path=Path("c.wav")),
        TranscriptResult(
            sample_id="d",
            input_path=Path("d.wav"),
            metrics={"cer": 100.0, "wer": 100.0},
            error="ASR failed",
        ),
    ]
    original = [result.model_dump() for result in results]

    summary = _summary(results, {"cer": 2.0, "wer": 1.0})

    assert summary.metrics["mean_cer"] == pytest.approx(0.4)
    assert summary.metrics["mean_wer"] == 0.8
    assert summary.metrics["weighted_mean"] == pytest.approx(1.6 / 3)
    assert (summary.total_samples, summary.succeeded, summary.failed) == (4, 3, 1)
    assert [result.model_dump() for result in results] == original


@pytest.mark.parametrize("weights", [{}, {"cer": 0.0}, {"unknown_metric": 0.0}])
def test_no_positive_weights_disables_weighted_mean(weights: dict[str, float]) -> None:
    result = TranscriptResult(sample_id="a", input_path=Path("a.wav"), metrics={"cer": 0.6})

    assert _summary([result], weights).metrics == {"mean_cer": 0.6}


def test_zero_weights_ignore_missing_and_nonfinite_metrics() -> None:
    result = TranscriptResult(
        sample_id="a", input_path=Path("a.wav"), metrics={"cer": 0.6, "wer": math.nan}
    )

    summary = _summary([result], {"cer": 1.0, "wer": 0.0, "unknown_metric": 0.0})

    assert summary.metrics["weighted_mean"] == 0.6
    assert math.isnan(summary.metrics["mean_wer"])
    assert math.isnan(result.metrics["wer"])


@pytest.mark.parametrize("weight", [-1.0, math.nan, math.inf, -math.inf])
def test_manifest_rejects_invalid_weights(weight: float) -> None:
    with pytest.raises(ValidationError):
        ScenarioManifest(name="scenario", samples=[], metric_weights={"cer": weight})


@pytest.mark.parametrize("weight", ["-1", ".nan", ".inf", "-.inf"])
def test_yaml_manifest_rejects_invalid_weights(tmp_path: Path, weight: str) -> None:
    path = tmp_path / "scenario.yaml"
    path.write_text(
        f"name: scenario\nsamples: []\nmetric_weights:\n  cer: {weight}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_scenario_manifest(path)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("metric", ["cer", "latency_seconds"])
def test_nonfinite_selected_mean_omits_only_weighted_mean(value: float, metric: str) -> None:
    result = TranscriptResult(
        sample_id="a",
        input_path=Path("a.wav"),
        metrics={"cer": value} if metric == "cer" else {},
        latency_seconds=value if metric == "latency_seconds" else None,
    )

    summary = _summary([result], {metric: 1.0})

    assert "weighted_mean" not in summary.metrics
    assert not math.isfinite(summary.metrics[f"mean_{metric}"])
    assert summary.succeeded == 1
    assert summary.failed == 0
    raw = result.metrics["cer"] if metric == "cer" else result.latency_seconds
    assert raw is not None and not math.isfinite(raw)


@pytest.mark.parametrize(
    ("values", "weights", "expected"),
    [
        ({"cer": 0.25, "wer": 0.5}, {"cer": 2.0, "wer": 1.0}, 1.0 / 3),
        ({"cer": 0.0, "wer": 0.0}, {"cer": 2.0, "wer": 1.0}, 0.0),
        (
            {"signed_score": -2.0, "other_score": 1.0},
            {"signed_score": 2.0, "other_score": 1.0},
            -1.0,
        ),
        ({"cer": 0.25, "wer": 0.5}, {"cer": 1e308, "wer": 1e308}, 0.375),
        ({"score": 1e308}, {"score": 2.0}, 1e308),
    ],
)
def test_valid_weighted_means(
    values: dict[str, float], weights: dict[str, float], expected: float
) -> None:
    result = TranscriptResult(sample_id="a", input_path=Path("a.wav"), metrics=values)

    summary = _summary([result], weights)

    assert math.isfinite(summary.metrics["weighted_mean"])
    assert summary.metrics["weighted_mean"] == pytest.approx(expected)


def test_overflowed_mean_cannot_produce_a_weighted_score() -> None:
    results = [
        TranscriptResult(sample_id=str(index), input_path=Path("a.wav"), metrics={"score": 1e308})
        for index in range(2)
    ]

    summary = _summary(results, {"score": 1.0})

    assert math.isinf(summary.metrics["mean_score"])
    assert "weighted_mean" not in summary.metrics
