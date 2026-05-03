from __future__ import annotations

from pathlib import Path

import typer

from local_speech_lab.benchmark import run_benchmark
from local_speech_lab.io import load_pipeline_config, load_scenario_manifest

app = typer.Typer(help="Benchmark a configured ASR pipeline over a scenario manifest.")


@app.command()
def run(
    config: Path = typer.Option(
        Path("configs/pipelines/zh_medical_funasr.yaml"),
        "--config",
        "-c",
        help="Pipeline YAML config.",
    ),
    manifest: Path = typer.Option(
        Path("configs/scenarios/zh_medical_smoke.jsonl"),
        "--manifest",
        "-m",
        help="Scenario manifest YAML or JSONL.",
    ),
    run_id: str | None = typer.Option(None, help="Optional stable run id."),
) -> None:
    pipeline_config = load_pipeline_config(config)
    scenario = load_scenario_manifest(manifest)
    summary = run_benchmark(pipeline_config=pipeline_config, scenario=scenario, run_id=run_id)
    typer.echo(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    app()

