# Glossary

Plain-language definitions for speech, ASR, and medical-domain terms used in this repo.

## ASR

Automatic Speech Recognition. Software that turns spoken audio into written text.

## Blacklist

A list of terms that should be flagged for human review when they appear in a transcript. In this repo, blacklist terms are not automatically deleted or rewritten.

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

## VAD

Voice Activity Detection. Software that detects whether audio currently contains speech.

## Wake Word

A phrase used to activate a speech system, such as a keyword that starts listening or transcription.
