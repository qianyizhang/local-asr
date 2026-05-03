from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ComponentKind = Literal["media", "asr", "postprocess", "review", "metric", "judge", "writer"]


class ComponentSpec(BaseModel):
    """Configuration for one swappable pipeline component."""

    kind: ComponentKind
    name: str
    options: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """A complete offline ASR pipeline definition."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    components: list[ComponentSpec]
    output_dir: Path = Path("outputs/benchmarks")


class ScenarioSample(BaseModel):
    """One benchmark or transcription input."""

    model_config = ConfigDict(extra="forbid")

    id: str
    input_path: Path
    reference_text: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScenarioManifest(BaseModel):
    """A collection of inputs evaluated together."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    samples: list[ScenarioSample]


class ReviewFlag(BaseModel):
    """A term or issue that should be reviewed by a human."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    value: str


class TranscriptJudgment(BaseModel):
    """LLM or human-style judgment for semantic transcript quality."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    semantic_equivalent: bool | None = None
    useful: bool | None = None
    score: float | None = None
    reason: str = ""
    raw_response: str = ""
    error: str | None = None


class TranscriptResult(BaseModel):
    """Pipeline output for a single sample."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    input_path: Path
    raw_text: str = ""
    text: str = ""
    reference_text: str | None = None
    flags: list[ReviewFlag] = Field(default_factory=list)
    judgments: list[TranscriptJudgment] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    latency_seconds: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkSummary(BaseModel):
    """Aggregate benchmark result for a pipeline run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    pipeline_name: str
    scenario_name: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_samples: int
    succeeded: int
    failed: int
    metrics: dict[str, float] = Field(default_factory=dict)
    output_dir: Path
