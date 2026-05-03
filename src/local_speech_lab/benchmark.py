from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from local_speech_lab.framework import build_offline_pipeline
from local_speech_lab.io import write_jsonl, write_model_json, write_model_yaml
from local_speech_lab.schemas import BenchmarkSummary, PipelineConfig, ScenarioManifest, TranscriptResult


def make_run_id(pipeline_name: str, scenario_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{pipeline_name}-{scenario_name}".replace(" ", "_")


def summarize_results(
    run_id: str,
    pipeline_config: PipelineConfig,
    scenario: ScenarioManifest,
    results: list[TranscriptResult],
    output_dir: Path,
) -> BenchmarkSummary:
    metric_totals: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    for result in results:
        if result.error:
            continue
        for metric_name, value in result.metrics.items():
            metric_totals[metric_name] = metric_totals.get(metric_name, 0.0) + value
            metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1

    metrics = {
        f"mean_{metric_name}": metric_totals[metric_name] / metric_counts[metric_name]
        for metric_name in sorted(metric_totals)
        if metric_counts[metric_name]
    }
    latencies = [
        result.latency_seconds for result in results if result.latency_seconds is not None and not result.error
    ]
    if latencies:
        metrics["mean_latency_seconds"] = sum(latencies) / len(latencies)

    failed = sum(1 for result in results if result.error)
    return BenchmarkSummary(
        run_id=run_id,
        pipeline_name=pipeline_config.name,
        scenario_name=scenario.name,
        total_samples=len(results),
        succeeded=len(results) - failed,
        failed=failed,
        metrics=metrics,
        output_dir=output_dir,
    )


def run_benchmark(
    pipeline_config: PipelineConfig,
    scenario: ScenarioManifest,
    run_id: str | None = None,
) -> BenchmarkSummary:
    resolved_run_id = run_id or make_run_id(pipeline_config.name, scenario.name)
    output_dir = pipeline_config.output_dir / resolved_run_id
    pipeline = build_offline_pipeline(pipeline_config.components)
    results = [
        pipeline.run_sample(
            sample_id=sample.id,
            input_path=sample.input_path,
            reference_text=sample.reference_text,
        )
        for sample in scenario.samples
    ]
    summary = summarize_results(
        run_id=resolved_run_id,
        pipeline_config=pipeline_config,
        scenario=scenario,
        results=results,
        output_dir=output_dir,
    )

    write_model_yaml(output_dir / "config_snapshot.yaml", pipeline_config)
    write_jsonl(output_dir / "samples.jsonl", results)
    write_model_json(output_dir / "summary.json", summary)
    return summary
