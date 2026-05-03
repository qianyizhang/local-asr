# Glossary

Plain-language definitions for speech, ASR, and medical-domain terms used in this repo.

## ASR

Automatic Speech Recognition. Software that turns spoken audio into written text.

## Blacklist

A list of terms that should be flagged for human review when they appear in a transcript. In this repo, blacklist terms are not automatically deleted or rewritten.

## Benchmark

A repeatable run of one pipeline over one scenario manifest. It records sample-level outputs and an aggregate summary so changes can be compared.

## CER

Character Error Rate. The number of character edits needed to turn a transcript into the reference text, divided by the reference length.

## Component

A swappable part of the ASR pipeline, such as media preparation, ASR backend, post-processing, review flagging, or metrics.

## Endpointing

Detecting where speech starts and stops in an audio stream.

## FunASR

An open-source speech recognition toolkit used here for Mandarin and Chinese medical-domain experiments.

## Hotword

A word or phrase given to an ASR model as a hint so the model is more likely to recognize it correctly.

## Hugging Face

A model hosting platform. This repo can download some model artifacts from Hugging Face, using a mirror endpoint when configured.

## ModelScope

A model hosting platform commonly used for Chinese speech models. This repo uses it for FunASR models.

## SenseVoice

A FunASR speech model family used here as the default local ASR baseline.

## Scenario Manifest

A YAML or JSONL file listing benchmark samples, their audio or video paths, optional reference transcripts, tags, and metadata.

## VAD

Voice Activity Detection. Software that detects whether audio currently contains speech.

## Wake Word

A phrase used to activate a speech system, such as a keyword that starts listening or transcription.

## WER

Word Error Rate. The number of word edits needed to turn a transcript into the reference text, divided by the reference word count.
