from __future__ import annotations

from pathlib import Path

import typer

from local_speech_lab.medical_text import (
    apply_corrections,
    flag_blacklisted_terms,
    load_corrections,
    load_lines,
    strip_sensevoice_tags,
)

app = typer.Typer(help="Apply medical corrections and flag blacklisted terms.")


@app.command()
def run(
    text: str,
    corrections_path: Path = Path("resources/corrections.zh-medical.tsv"),
    blacklist_path: Path = Path("resources/blacklist.zh-medical.txt"),
) -> None:
    corrections = load_corrections(corrections_path)
    blacklist = load_lines(blacklist_path)
    corrected = apply_corrections(strip_sensevoice_tags(text), corrections)
    flagged = flag_blacklisted_terms(corrected, blacklist)

    typer.echo(corrected)
    if flagged:
        typer.echo(f"FLAGGED_TERMS={','.join(flagged)}")


if __name__ == "__main__":
    app()
