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
source .venv/bin/activate
python scripts/check_env.py
```

When the virtualenv is not already activated, use `.venv/bin/python`; bare `python` may not exist on the shell PATH. Set `PYTHONPATH=src` unless the package is already installed editable in the active environment:

```bash
PYTHONPATH=src .venv/bin/python scripts/postprocess_transcript.py "患者需要服用二甲双瓜"
PYTHONPATH=src .venv/bin/python scripts/asr_funasr_file.py data/audio/sample.wav
PYTHONPATH=src .venv/bin/python scripts/benchmark_pipeline.py --config configs/pipelines/zh_medical_funasr.yaml --manifest configs/scenarios/zh_medical_smoke.jsonl
PYTHONPATH=src .venv/bin/python scripts/download_model.py list-models
```

For linting after dev dependencies are installed:

```bash
.venv/bin/ruff check .
```

There is currently no dedicated test suite. For lightweight verification of text-processing changes, use:

```bash
PYTHONPATH=src .venv/bin/python scripts/postprocess_transcript.py "患者需要服用二甲双瓜并复查糖化血红蛋白a1c"
```

Expected behavior: the output should include `二甲双胍` and `糖化血红蛋白 HbA1c`.

## Local Environment Rules

- Do not download large models unless the task requires it. Model downloads may use network and fill `models/` or `.cache/`.
- Do not run `scripts/install_zsh_defaults.sh` unless the user asks. It edits the user's `~/.zshrc`.
- Prefer repo-local cache paths already used by the scripts: `.cache/huggingface`, `.cache/modelscope`, and `.cache/uv`.
- The default Hugging Face endpoint is `https://hf-mirror.com`; ModelScope is also used for FunASR models.
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
- CLI or import changes: run `.venv/bin/python scripts/check_env.py` plus the affected script with `PYTHONPATH=src`.
- ASR pipeline changes: use a small local audio file when available; avoid model downloads unless already present or requested.
- Framework changes: run config/manifest loading plus `scripts/benchmark_pipeline.py` when the local sample model exists.
- Formatting/lint-only changes: run `.venv/bin/ruff check .` if Ruff is installed.

If verification requires missing models, network access, microphone access, or external audio files, say that explicitly in the final response.

## Git And File Hygiene

- Check `git status --short` before and after edits.
- Do not revert unrelated user changes.
- Do not commit, push, or create branches unless the user asks.
- Avoid editing runtime directories such as `models/`, `data/`, `outputs/`, and `.cache/`.
- Keep AGENTS.md updates concise and current with the actual repo behavior.
