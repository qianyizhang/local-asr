from __future__ import annotations

from pathlib import Path

import typer

from local_speech_lab.benchmark import run_benchmark
from local_speech_lab.io import load_pipeline_config
from local_speech_lab.schemas import ScenarioManifest, ScenarioSample

app = typer.Typer(help="Run one configured ASR pipeline on one local audio or video file.")


@app.command()
def run(
    input_path: Path = typer.Argument(..., help="Local audio/video file to transcribe."),
    config: Path = typer.Option(
        Path("configs/pipelines/zh_medical_funasr.yaml"),
        "--config",
        "-c",
        help="Pipeline YAML config.",
    ),
    sample_id: str = typer.Option("manual", help="Sample id stored in run artifacts."),
    reference_text: str | None = typer.Option(None, help="Optional transcript reference for metrics."),
    run_id: str | None = typer.Option(None, help="Optional stable run id."),
) -> None:
    pipeline_config = load_pipeline_config(config)
    scenario = ScenarioManifest(
        name=sample_id,
        description="Single-sample pipeline run.",
        samples=[
            ScenarioSample(
                id=sample_id,
                input_path=input_path,
                reference_text=reference_text,
            )
        ],
    )
    summary = run_benchmark(pipeline_config=pipeline_config, scenario=scenario, run_id=run_id)
    typer.echo(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    app()

