from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from local_speech_lab.schemas import PipelineConfig, ScenarioManifest, ScenarioSample

ModelT = TypeVar("ModelT", bound=BaseModel)


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return loaded


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    return PipelineConfig.model_validate(read_yaml(path))


def load_scenario_manifest(path: str | Path) -> ScenarioManifest:
    manifest_path = Path(path)
    if manifest_path.suffix == ".jsonl":
        samples: list[ScenarioSample] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = line.strip()
                if not value:
                    continue
                try:
                    samples.append(ScenarioSample.model_validate_json(value))
                except ValueError as exc:
                    raise ValueError(f"Invalid sample at {manifest_path}:{line_number}") from exc
        return ScenarioManifest(name=manifest_path.stem, samples=samples)
    return ScenarioManifest.model_validate(read_yaml(manifest_path))


def write_model_json(path: str | Path, model: BaseModel) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_model_yaml(path: str | Path, model: BaseModel) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(model.model_dump(mode="json"), handle, allow_unicode=True, sort_keys=False)


def write_jsonl(path: str | Path, models: list[BaseModel]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(model.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
        for model in models
    ]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
