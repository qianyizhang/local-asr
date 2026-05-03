from __future__ import annotations

import os
from pathlib import Path

import typer
from huggingface_hub import snapshot_download

app = typer.Typer(help="Download local LLM judge models for semantic transcript evaluation.")

MODEL_PRESETS = {
    "qwen3.6": (
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        Path("models/qwen3_6_35b_a3b_gguf"),
        ["Qwen3.6-35B-A3B-UD-IQ4_NL.gguf"],
    ),
    "qwen3-small": ("Qwen/Qwen3-0.6B", Path("models/qwen3_0_6b"), None),
    "gemma-small": ("google/gemma-3-270m-it", Path("models/gemma3_270m_it"), None),
}


@app.command()
def run(
    preset: str = typer.Option(
        "qwen3-small",
        help="One of: qwen3.6, qwen3-small, gemma-small.",
    ),
    repo_id: str | None = typer.Option(None, help="Override Hugging Face model id."),
    output_dir: Path | None = typer.Option(None, help="Override local model directory."),
) -> None:
    if preset not in MODEL_PRESETS and repo_id is None:
        raise typer.BadParameter(f"Unknown preset: {preset}")

    preset_repo_id, preset_output_dir, preset_patterns = MODEL_PRESETS.get(preset, (None, None, None))
    selected_repo_id = repo_id or preset_repo_id
    selected_output_dir = output_dir or preset_output_dir
    if selected_repo_id is None or selected_output_dir is None:
        raise typer.BadParameter("Provide both --repo-id and --output-dir for custom presets.")

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))

    patterns = preset_patterns or [
        "*.json",
        "*.safetensors",
        "*.txt",
        "*.model",
        "*.py",
        "*.jinja",
        "LICENSE",
        "README.md",
        "merges.txt",
        "vocab.json",
    ]
    path = snapshot_download(
        repo_id=selected_repo_id,
        local_dir=selected_output_dir,
        allow_patterns=patterns,
    )
    typer.echo(path)


if __name__ == "__main__":
    app()
