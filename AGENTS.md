# AGENTS.md

Guidance for coding agents working in this repository.

## Project Scope

This repo is a local speech-recognition lab for Chinese and medical-domain workflows. The main pipeline is:

```text
audio/video input -> optional ffmpeg extraction -> FunASR -> SenseVoice tag cleanup
  -> medical corrections -> blacklist flagging -> transcript output
```

Keep changes aligned with that experimental, local-first scope. Prefer small, inspectable scripts and resource files over service-style infrastructure unless the user explicitly asks for it.

## Repository Layout

- `src/local_speech_lab/`: importable Python package code.
- `scripts/`: command-line workflows for bootstrap, environment checks, model downloads, transcription, and post-processing.
- `configs/`: YAML/JSONL configuration for models, pipelines, scenarios, and wake-word experiments.
- `resources/`: editable medical vocabulary, correction, and blacklist files.
- `docs/glossary.md`: plain-language definitions for speech, ASR, and medical-domain terms.
- `docs/design/`: design notes for complex tasks.
- `data/`, `models/`, `outputs/`, `.cache/`: local runtime artifacts. Do not treat these as source unless the user specifically asks.

## Setup And Commands

Use Python 3.10 through 3.12.

```bash
./scripts/bootstrap.sh
make check
```

`uv` is a global tool, but it should manage this repo's local `.venv` from `pyproject.toml`. Prefer the `Makefile` for common workflows; it exports the repo-local uv cache and `PYTHONPATH=src`.

For direct one-off script runs after bootstrap, use `uv run --no-sync` instead of calling `.venv/bin/python` directly:

```bash
export UV_CACHE_DIR=.cache/uv
export PYTHONPATH=src
uv run --no-sync --no-dev scripts/postprocess_transcript.py "患者需要服用二甲双瓜"
uv run --no-sync --no-dev scripts/asr_funasr_file.py data/audio/sample.wav
uv run --no-sync --no-dev scripts/benchmark_pipeline.py --config configs/pipelines/zh_medical_funasr.yaml --manifest configs/scenarios/zh_medical_smoke.jsonl
uv run --no-sync --no-dev scripts/download_model.py list-models
```

Makefile shortcuts:

```bash
make check
make test
make postprocess
make benchmark
```

For linting after dev dependencies are installed:

```bash
make lint
```

Pytest is the main test suite:

```bash
make test
```

For lightweight verification of text-processing changes, use:

```bash
make postprocess
```

Expected behavior: the output should include `二甲双胍` and `糖化血红蛋白 HbA1c`.

## Local Environment Rules

- Do not download large models unless the task requires it. Model downloads may use network and fill `models/` or `.cache/`.
- Do not run `scripts/install_zsh_defaults.sh` unless the user asks. It edits the user's `~/.zshrc`.
- Prefer repo-local cache paths already used by the scripts: `.cache/huggingface`, `.cache/modelscope`, and `.cache/uv`.
- Prefer `uv sync` for environment updates and `uv run --no-sync` for routine one-off script runs; avoid ad hoc `pip install` or direct `.venv/bin/*` calls unless diagnosing environment issues.
- The default Hugging Face endpoint is `https://hf-mirror.com`; ModelScope is also used for FunASR models.
- Do not download Hugging Face model artifacts with raw `curl` or direct file URLs. Use Hugging Face tooling such as `huggingface_hub.snapshot_download()` with the repo-local cache/proxy setup.
- Video transcription requires `ffmpeg`; do not silently replace this path with another media pipeline.

## Coding Conventions

- Prefer simplicity and clarity over exhaustive coverage. A short, obvious implementation is better than a comprehensive abstraction until the repo proves it needs more.
- Use type hints and `from __future__ import annotations` for Python files, matching the existing code.
- Type hinting is required for new and changed Python code.
- Use Pydantic `BaseModel` for data that crosses process, CLI, API, config, or serialization boundaries.
- Use dataclasses for simple internal structures.
- Avoid plain `dict` unless the value is truly a mapping. If a dict-shaped object is required, prefer `TypedDict` for fixed keys.
- Keep command-line scripts Typer-based when adding user-facing commands.
- Keep resource loading explicit and UTF-8 encoded.
- Preserve Chinese medical terms exactly in resource files; avoid automatic formatting that could reorder or alter vocabulary.
- Keep generated transcript and model artifacts out of source changes unless the user explicitly asks to update examples.
- Use structured parsers for YAML/TSV where practical; avoid brittle ad hoc parsing for config changes.

## Documentation Expectations

- Keep `docs/glossary.md` updated with technical terms that a non-expert reader may not know.
- For complex work, draft `docs/design/{taskname}.md` before or alongside implementation.
- Design notes should cover the problem, goal, non-goals, proposed solution, tradeoffs, implementation details, and future notes.
- Keep docs concise and practical. Prefer clear examples over broad background.

## ASR And Post-Processing Notes

- `scripts/asr_funasr_file.py` defaults to `models/funasr_sensevoice_small/iic/SenseVoiceSmall/example/zh.mp3`.
- `resources/hotwords.zh-medical.txt` biases recognition.
- `resources/corrections.zh-medical.tsv` applies literal post-ASR replacements in file order.
- `resources/blacklist.zh-medical.txt` flags terms for review; it should not delete or rewrite transcript text.
- `strip_sensevoice_tags()` removes SenseVoice marker tokens before correction and blacklist checks.
- Framework components live under `src/local_speech_lab/`; prefer adding swappable media, ASR, post-processing, review, or metric components there before adding one-off scripts.
- Scenario manifests should use stable sample ids and optional `reference_text`; benchmark outputs belong under `outputs/benchmarks/`.

## Verification Guidance

Choose the narrowest verification that covers the change:

- Medical text logic: run the postprocess smoke command above.
- Unit/framework changes: run `make test`.
- CLI or import changes: run `make check` plus the affected script with `uv run --no-sync --no-dev`.
- ASR pipeline changes: use a small local audio file when available; avoid model downloads unless already present or requested.
- Framework changes: run config/manifest loading plus `scripts/benchmark_pipeline.py` when the local sample model exists.
- Formatting/lint-only changes: run `make lint`.

If verification requires missing models, network access, microphone access, or external audio files, say that explicitly in the final response.

## Git And File Hygiene

- Check `git status --short` before and after edits.
- Do not revert unrelated user changes.
- Do not commit, push, or create branches unless the user asks.
- Avoid editing runtime directories such as `models/`, `data/`, `outputs/`, and `.cache/`.
- Keep AGENTS.md updates concise and current with the actual repo behavior.
