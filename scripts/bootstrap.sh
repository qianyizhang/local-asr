#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$PWD/.cache/modelscope}"
export MODELSCOPE_CREDENTIALS_PATH="${MODELSCOPE_CREDENTIALS_PATH:-$PWD/.cache/modelscope/credentials}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"

if command -v uv >/dev/null 2>&1; then
  uv sync --extra ui --extra vad --group dev
else
  python3 -m venv .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -e ".[ui,vad]"
fi

mkdir -p models data/audio data/transcripts outputs .cache/huggingface .cache/modelscope/credentials .cache/uv

echo "Bootstrap complete."
echo "HF_ENDPOINT=$HF_ENDPOINT"
