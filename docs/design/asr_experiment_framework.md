# ASR Experiment Framework

## Problem

ASR improvements can come from many places: base model choice, hotwords, fine-tuning data, decoding
settings, streaming strategy, post-processing, and domain review rules. Those changes are hard to
compare when they are embedded in one-off scripts.

## Goal

Make the repo a local-first framework for iterative ASR pipeline improvements. Each experiment should
change a small component, run against a stable scenario manifest, and leave enough artifacts to compare
the result later.

## Non-goals

- No hosted service or database.
- No automatic model downloads during benchmark runs.
- No autonomous keep/discard loop in v1.
- No forced medical domain; Chinese medical resources are one example profile.

## Proposed Workflow

1. Define or update a scenario manifest with stable sample ids and local input paths.
2. Add `reference_text` when a transcript is known; omit it for exploratory audio-only runs.
3. Copy or edit a pipeline config to swap one component at a time.
4. Run `scripts/benchmark_pipeline.py`.
5. Compare `summary.json` and inspect `samples.jsonl`.
6. Keep the change only if it improves the chosen metric or produces a useful qualitative artifact.

## Framework Shape

- Schemas live in `src/local_speech_lab/schemas.py`.
- Component interfaces live in `src/local_speech_lab/components.py`.
- Built-in offline components live in `src/local_speech_lab/framework.py`.
- Benchmark orchestration and artifact writing live in `src/local_speech_lab/benchmark.py`.

The first built-in pipeline supports local media prep, FunASR transcription, SenseVoice tag cleanup,
literal correction files, blacklist review flags, and CER/WER metrics.

## Future Notes

- Add a streaming pipeline runner that uses the same scenario and result schemas.
- Add adapters for Whisper or sherpa-onnx as independent ASR backends.
- Add scenario-level metric weights once real datasets identify the target tradeoffs.
- Add an agent loop only after benchmarks are stable enough to make automated comparison meaningful.
