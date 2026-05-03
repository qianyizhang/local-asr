# Local Speech Lab - Makefile
# Python 3.10-3.12 required; uv manages the local .venv.

export UV_CACHE_DIR ?= .cache/uv
export PYTHONPATH := src

PIPELINE ?= configs/pipelines/zh_medical_funasr.yaml
SMOKE_MANIFEST ?= configs/scenarios/zh_medical_smoke.jsonl
AISHELL_MANIFEST ?= configs/scenarios/aishell_smoke.jsonl
POSTPROCESS_SAMPLE ?= 患者需要服用二甲双瓜并复查糖化血红蛋白a1c

.PHONY: all bootstrap setup venv check test lint fmt
.PHONY: download-model download-aishell download-judge download-judge-small
.PHONY: postprocess asr asr-file pipeline benchmark benchmark-aishell
.PHONY: clean clean-cache clean-all clean-outputs help

# Default target
all: check

# Setup and bootstrap
# -------------------

bootstrap: ## Create venv and install dependencies (uses uv if available)
	./scripts/bootstrap.sh

setup: bootstrap ## Alias for bootstrap

venv: ## Sync uv-managed venv
	uv sync --extra ui --extra vad --group dev

# Verification and quality
# ------------------------

check: ## Run environment check
	uv run --no-sync --no-dev scripts/check_env.py

test: ## Run pytest suite
	uv run --no-sync --group dev pytest

lint: ## Run ruff linter
	uv run --no-sync --group dev ruff check .

fmt: ## Auto-format with ruff (if supported)
	uv run --no-sync --group dev ruff format . 2>/dev/null || uv run --no-sync --group dev ruff check --fix .

# Model downloads
# ---------------

download-model: ## Download ASR models (interactive)
	uv run --no-sync --no-dev scripts/download_model.py list-models
	@echo "Run 'make download-model-<name>' to download a specific model"

download-model-%: ## Download specific model (e.g., download-model-sensevoice_small)
	uv run --no-sync --no-dev scripts/download_model.py $*

download-aishell: ## Download AISHELL smoke dataset
	uv run --no-sync --no-dev scripts/download_aishell_smoke.py

download-judge: ## Download LLM judge model (default: qwen3.6)
	uv run --no-sync --no-dev scripts/download_judge_model.py --preset qwen3.6

download-judge-small: ## Download small judge model (qwen3-small)
	uv run --no-sync --no-dev scripts/download_judge_model.py --preset qwen3-small

# Core workflows
# --------------

postprocess: ## Run medical post-processor smoke test
	uv run --no-sync --no-dev scripts/postprocess_transcript.py "$(POSTPROCESS_SAMPLE)"

asr: ## Run FunASR on default sample (models/funasr_sensevoice_small/example/zh.mp3)
	uv run --no-sync --no-dev scripts/asr_funasr_file.py

asr-file: ## Run FunASR on specific file (set FILE=path/to/audio.wav)
	@test -n "$(FILE)" || (echo "Usage: make asr-file FILE=path/to/audio.wav" && exit 1)
	uv run --no-sync --no-dev scripts/asr_funasr_file.py $(FILE)

pipeline: ## Run full pipeline on a file (set FILE=path/to/audio.wav)
	@test -n "$(FILE)" || (echo "Usage: make pipeline FILE=path/to/audio.wav" && exit 1)
	uv run --no-sync --no-dev scripts/run_pipeline.py $(FILE) --config $(PIPELINE)

benchmark: ## Run benchmark with smoke scenario
	uv run --no-sync --no-dev scripts/benchmark_pipeline.py --config $(PIPELINE) --manifest $(SMOKE_MANIFEST)

benchmark-aishell: download-aishell ## Run AISHELL smoke benchmark
	uv run --no-sync --no-dev scripts/benchmark_pipeline.py --config $(PIPELINE) --manifest $(AISHELL_MANIFEST)

# Development helpers
# -------------------

clean: ## Remove Python cache and output artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true

clean-cache: ## Remove model and download caches (destructive)
	rm -rf .cache/huggingface .cache/modelscope .cache/uv

clean-all: clean clean-cache ## Remove all generated artifacts including venv
	rm -rf .venv outputs/data outputs/benchmarks

clean-outputs: ## Remove output files only
	rm -rf outputs/*

# Utilities
# ---------

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
