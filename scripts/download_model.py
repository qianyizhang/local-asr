from __future__ import annotations

import os
from pathlib import Path

import typer
import yaml

app = typer.Typer(help="Download models from Hugging Face mirror or ModelScope.")


def load_catalog() -> dict:
    with Path("configs/model_catalog.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@app.command()
def list_models() -> None:
    catalog = load_catalog()
    for group, models in catalog.items():
        if group == "defaults":
            continue
        typer.echo(f"[{group}]")
        for name, spec in models.items():
            typer.echo(f"  {name}: {spec['provider']}:{spec['model_id']}")


@app.command()
def fetch(name: str) -> None:
    catalog = load_catalog()
    defaults = catalog.get("defaults", {})
    model_dir = Path(defaults.get("model_dir", "models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", defaults.get("huggingface_endpoint", "https://hf-mirror.com"))
    os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
    os.environ.setdefault("MODELSCOPE_CACHE", str(Path(".cache/modelscope").resolve()))
    os.environ.setdefault(
        "MODELSCOPE_CREDENTIALS_PATH",
        str(Path(".cache/modelscope/credentials").resolve()),
    )

    selected: dict | None = None
    for group, models in catalog.items():
        if group == "defaults":
            continue
        if name in models:
            selected = models[name]
            break

    if selected is None:
        raise typer.BadParameter(f"Unknown model: {name}. Run list-models first.")

    provider = selected["provider"]
    model_id = selected["model_id"]

    if provider == "huggingface":
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=model_id, local_dir=model_dir / name)
        typer.echo(path)
        return

    if provider == "modelscope":
        from modelscope import snapshot_download

        path = snapshot_download(model_id, cache_dir=str(model_dir / name))
        typer.echo(path)
        return

    typer.echo(
        f"{name} is listed as provider={provider}. "
        "Follow the upstream project instructions and place artifacts under models/."
    )


if __name__ == "__main__":
    app()
