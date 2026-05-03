# Local Speech Lab — Makefile
# Python 3.10–3.12 required; uses .venv for isolation

.PHONY: all bootstrap setup check test lint clean clean-cache help
.PHONY: download-model download-aishell download-judge-model
.PHONY: postprocess asr benchmark pipeline run-demo

# Default target
all: check

# Setup and bootstrap
# -------------------

bootstrap: ## Create venv and install dependencies (uses uv if available)
	./scripts/bootstrap.sh

setup: bootstrap ## Alias for bootstrap

venv: ## Ensure venv exists (create if missing)
	@test -d .venv || $(MAKE) bootstrap

# Verification and quality
# ------------------------

check: venv ## Run environment check
	.venv/bin/python scripts/check_env.py

test: venv ## Run pytest suite
	.venv/bin/pytest

lint: venv ## Run ruff linter
	.venv/bin/ruff check .

fmt: venv ## Auto-format with ruff (if supported)
	.venv/bin/ruff format . 2>/dev/null || .venv/bin/ruff check --fix .

# Model downloads
# ---------------

download-model: venv ## Download ASR models (interactive)
	PYTHONPATH=src .venv/bin/python scripts/download_model.py list-models
	@echo "Run 'make download-model MODEL=<name>' to download a specific model"

download-model-%: venv ## Download specific model (e.g., download-model-sensevoice_small)
	PYTHONPATH=src .venv/bin/python scripts/download_model.py $*

download-aishell: venv ## Download AISHELL smoke dataset
	PYTHONPATH=src .venv/bin/python scripts/download_aishell_smoke.py

download-judge: venv ## Download LLM judge model (default: qwen3.6)
	PYTHONPATH=src .venv/bin/python scripts/download_judge_model.py --preset qwen3.6

download-judge-small: venv ## Download small judge model (qwen3-small)
	PYTHONPATH=src .venv/bin/python scripts/download_judge_model.py --preset qwen3-small

# Core workflows
# --------------

postprocess: venv ## Run medical post-processor smoke test
	PYTHONPATH=src .venv/bin/python scripts/postprocess_transcript.py "患者需要服用二甲双瓜并复查糖化血红蛋白a1c"

asr: venv ## Run FunASR on default sample (models/funasr_sensevoice_small/example/zh.mp3)
	PYTHONPATH=src .venv/bin/python scripts/asr_funasr_file.py

asr-file: venv ## Run FunASR on specific file (set FILE=path/to/audio.wav)
	@test -n "$(FILE)" || (echo "Usage: make asr-file FILE=path/to/audio.wav" && exit 1)
	PYTHONPATH=src .venv/bin/python scripts/asr_funasr_file.py $(FILE)

pipeline: venv ## Run full pipeline on a file (set FILE=path/to/audio.wav)
	@test -n "$(FILE)" || (echo "Usage: make pipeline FILE=path/to/audio.wav" && exit 1)
	PYTHONPATH=src .venv/bin/python scripts/run_pipeline.py $(FILE) \
	  --config configs/pipelines/zh_medical_funasr.yaml

benchmark: venv ## Run benchmark with smoke scenario
	PYTHONPATH=src .venv/bin/python scripts/benchmark_pipeline.py \
	  --config configs/pipelines/zh_medical_funasr.yaml \
	  --manifest configs/scenarios/zh_medical_smoke.jsonl

benchmark-aishell: venv download-aishell ## Run AISHELL smoke benchmark
	PYTHONPATH=src .venv/bin/python scripts/benchmark_pipeline.py \
	  --config configs/pipelines/zh_medical_funasr.yaml \
	  --manifest configs/scenarios/aishell_smoke.jsonl

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
