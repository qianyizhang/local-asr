from __future__ import annotations

import asyncio
import contextlib
import io
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import typer

from local_speech_lab.medical_text import (
    apply_corrections,
    flag_blacklisted_terms,
    load_corrections,
    load_lines,
    strip_sensevoice_tags,
)

app = typer.Typer(help="Run FunASR on a local audio file and apply medical post-processing.")

DEFAULT_LOCAL_MODEL = Path("models/funasr_sensevoice_small/iic/SenseVoiceSmall")
DEFAULT_LOCAL_CLIP = DEFAULT_LOCAL_MODEL / "example/zh.mp3"
DEFAULT_OUTPUT_PATH = Path("outputs/local_stream_transcript.txt")


def _emit_stream(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


async def _extract_audio_if_needed_async(
    media_path: Path,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if media_path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
        return media_path, None

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise typer.BadParameter("Video input requires ffmpeg, but ffmpeg was not found.")

    temp_dir = tempfile.TemporaryDirectory(prefix="local-asr-")
    audio_path = Path(temp_dir.name) / f"{media_path.stem}.wav"
    process = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(audio_path),
    )
    returncode = await process.wait()
    if returncode != 0:
        temp_dir.cleanup()
        raise typer.BadParameter(f"ffmpeg failed to extract audio from {media_path}")
    return audio_path, temp_dir


def _recognize(
    audio_path: Path,
    model: str,
    hotwords_path: Path,
) -> str:
    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output), contextlib.redirect_stderr(captured_output):
        from funasr import AutoModel

        hotwords = " ".join(load_lines(hotwords_path))
        recognizer = AutoModel(
            model=model,
            trust_remote_code=True,
            disable_update=True,
            disable_pbar=True,
            log_level="ERROR",
        )
        result = recognizer.generate(input=str(audio_path), hotword=hotwords)

    if result and isinstance(result, list):
        return str(result[0].get("text", ""))
    return ""


async def _recognize_async(
    audio_path: Path,
    model: str,
    hotwords_path: Path,
) -> str:
    return await asyncio.to_thread(_recognize, audio_path, model, hotwords_path)


async def _transcribe_async(
    audio_path: Path,
    model: str,
    hotwords_path: Path,
    corrections_path: Path,
    blacklist_path: Path,
    output_path: Path,
    stream: bool,
) -> str:
    if stream:
        _emit_stream("input", path=str(audio_path))

    extracted_path, temp_dir = await _extract_audio_if_needed_async(audio_path)
    try:
        if stream and extracted_path != audio_path:
            _emit_stream("audio_extracted", path=str(extracted_path))

        if stream:
            _emit_stream("model_loading", model=model)
        raw_text = await _recognize_async(extracted_path, model, hotwords_path)

        corrections = load_corrections(corrections_path)
        corrected = apply_corrections(strip_sensevoice_tags(raw_text), corrections)
        flagged = flag_blacklisted_terms(corrected, load_lines(blacklist_path))

        if stream:
            _emit_stream("transcript", text=corrected)
            if flagged:
                _emit_stream("flagged_terms", terms=flagged)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            corrected,
            "",
            f"source={audio_path}",
            f"model={model}",
        ]
        if flagged:
            lines.append(f"flagged_terms={','.join(flagged)}")
        output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

        if stream:
            _emit_stream("output", path=str(output_path))
        return corrected
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


@app.command()
def transcribe(
    audio_path: Path = typer.Argument(
        DEFAULT_LOCAL_CLIP,
        help="Local audio/video file to transcribe. Defaults to the bundled SenseVoice sample.",
    ),
    model: str = typer.Option(
        str(DEFAULT_LOCAL_MODEL),
        help="FunASR model name or local model directory. Defaults to the bundled local model.",
    ),
    hotwords_path: Path = Path("resources/hotwords.zh-medical.txt"),
    corrections_path: Path = Path("resources/corrections.zh-medical.tsv"),
    blacklist_path: Path = Path("resources/blacklist.zh-medical.txt"),
    output_path: Path = typer.Option(
        DEFAULT_OUTPUT_PATH,
        "--output",
        "-o",
        help="Transcript output path.",
    ),
    stream: bool = typer.Option(True, help="Emit JSONL streaming progress and transcript events."),
) -> None:
    corrected = asyncio.run(
        _transcribe_async(
            audio_path=audio_path,
            model=model,
            hotwords_path=hotwords_path,
            corrections_path=corrections_path,
            blacklist_path=blacklist_path,
            output_path=output_path,
            stream=stream,
        )
    )
    if not stream:
        typer.echo(corrected)


if __name__ == "__main__":
    app()
