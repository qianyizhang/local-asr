#!/usr/bin/env bash
set -euo pipefail

ZSHRC="${HOME}/.zshrc"
BLOCK_START="# >>> local-speech-lab defaults >>>"
BLOCK_END="# <<< local-speech-lab defaults <<<"

if [ ! -f "$ZSHRC" ]; then
  touch "$ZSHRC"
fi

if grep -q "$BLOCK_START" "$ZSHRC"; then
  if ! grep -q "MODELSCOPE_CREDENTIALS_PATH" "$ZSHRC"; then
    sed -i '' "/export MODELSCOPE_CACHE=/a\\
export MODELSCOPE_CREDENTIALS_PATH=\\$HOME/.cache/modelscope/credentials
" "$ZSHRC"
    echo "Updated local-speech-lab defaults in $ZSHRC"
    exit 0
  fi
  echo "local-speech-lab defaults already exist in $ZSHRC"
  exit 0
fi

{
  echo ""
  echo "$BLOCK_START"
  echo "export HF_ENDPOINT=https://hf-mirror.com"
  echo "export HF_HOME=\$HOME/.cache/huggingface"
  echo "export MODELSCOPE_CACHE=\$HOME/.cache/modelscope"
  echo "export MODELSCOPE_CREDENTIALS_PATH=\$HOME/.cache/modelscope/credentials"
  echo "export SPEECH_LAB_MODEL_DIR=\$HOME/models/speech"
  echo "$BLOCK_END"
} >> "$ZSHRC"

echo "Added local-speech-lab defaults to $ZSHRC"
