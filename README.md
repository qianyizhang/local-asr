# Local Speech Lab

Local speech recognition and wake-word experiments for Chinese and medical-domain workflows.

This project separates the speech stack into small pieces so each part can be benchmarked or replaced independently:

```text
microphone
  -> VAD / endpointing
  -> wake word / keyword spotting
  -> ASR
  -> hotword biasing
  -> medical correction / blacklist flagging
  -> transcript
```

## Recommended Stack

- Wake word / keyword spotting: `sherpa-onnx`
- VAD: `sherpa-onnx` or `silero-vad`
- Chinese ASR: `FunASR`
- Mac baseline ASR: `whisper.cpp`
- Medical vocabulary: curated hotword and correction files under `resources/`
- Model downloads: Hugging Face mirror via `HF_ENDPOINT=https://hf-mirror.com`; ModelScope as alternate source

## Quick Start

```bash
./scripts/bootstrap.sh
source .venv/bin/activate
python scripts/check_env.py
```

Run the medical post-processor:

```bash
PYTHONPATH=src python scripts/postprocess_transcript.py "患者需要服用二甲双瓜并复查糖化血红蛋白a1c"
```

Run FunASR on an audio file:

```bash
PYTHONPATH=src python scripts/asr_funasr_file.py data/audio/sample.wav
```

Run the bundled local streaming smoke test:

```bash
PYTHONPATH=src python scripts/asr_funasr_file.py
cat outputs/local_stream_transcript.txt
```

By default this uses `models/funasr_sensevoice_small/iic/SenseVoiceSmall/example/zh.mp3`,
emits JSONL progress/transcript events to stdout, and writes the final transcript to
`outputs/local_stream_transcript.txt`.

Then edit:

- `resources/hotwords.zh-medical.txt`
- `resources/corrections.zh-medical.tsv`
- `resources/blacklist.zh-medical.txt`
- `configs/model_catalog.yaml`

## Iterative Experiments

The repo now has a small framework for comparing ASR pipeline changes over scenario manifests:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_pipeline.py \
  --config configs/pipelines/zh_medical_funasr.yaml \
  --manifest configs/scenarios/zh_medical_smoke.jsonl
```

For one local file:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_pipeline.py data/audio/sample.wav \
  --config configs/pipelines/zh_medical_funasr.yaml
```

Pipeline configs choose swappable components for media prep, ASR backend, post-processing,
review flagging, and metrics. Scenario manifests list samples with `id`, `input_path`, optional
`reference_text`, tags, and metadata. When references are present, benchmark summaries include
CER/WER; otherwise they still record transcripts, flags, latency, and errors.

Benchmark artifacts are written under `outputs/benchmarks/{run_id}/`:

- `config_snapshot.yaml`
- `samples.jsonl`
- `summary.json`

Use `docs/design/asr_experiment_framework.md` as the operating guide for iterative component
improvements.

## Hugging Face Mirror

For faster downloads from China, the project uses:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

To make this default in zsh, run:

```bash
./scripts/install_zsh_defaults.sh
source ~/.zshrc
```

The script appends a guarded block to `~/.zshrc`.

## Model Notes

Start with these tracks:

1. `sherpa-onnx` keyword spotting and VAD for wake-up detection.
2. `FunASR` for Mandarin ASR and punctuation.
3. `whisper.cpp` for Mac-native baseline comparison.

For medical-domain accuracy, first try hotwords/contextual biasing and post-processing correction. Full ASR fine-tuning is a later step and is often easier on an NVIDIA GPU machine.
