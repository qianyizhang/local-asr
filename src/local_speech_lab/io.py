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
    return PipelineConfig.model_validate(_read_pipeline_config(path))


def _read_pipeline_config(path: str | Path, seen: set[Path] | None = None) -> dict[str, Any]:
    config_path = Path(path)
    resolved_path = config_path.resolve()
    visited = seen or set()
    if resolved_path in visited:
        raise ValueError(f"Pipeline config extends cycle includes {config_path}")
    visited.add(resolved_path)

    data = read_yaml(config_path)
    base_path_value = data.pop("extends", None)
    extra_components = data.pop("extra_components", [])
    if extra_components and not isinstance(extra_components, list):
        raise ValueError(f"Expected extra_components list in {config_path}")

    if base_path_value is None:
        if extra_components:
            components = data.get("components", [])
            if not isinstance(components, list):
                raise ValueError(f"Expected components list in {config_path}")
            data["components"] = [*components, *extra_components]
        return data

    base_path = Path(str(base_path_value))
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path

    base_data = _read_pipeline_config(base_path, visited)
    child_components = data.pop("components", None)
    if child_components is not None and not isinstance(child_components, list):
        raise ValueError(f"Expected components list in {config_path}")

    merged = {**base_data, **data}
    if child_components is not None:
        merged["components"] = child_components
    if extra_components:
        components = merged.get("components", [])
        if not isinstance(components, list):
            raise ValueError(f"Expected inherited components list for {config_path}")
        merged["components"] = [*components, *extra_components]
    return merged


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
