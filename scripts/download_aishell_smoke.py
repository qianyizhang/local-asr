from __future__ import annotations

import tarfile
from pathlib import Path

import typer
from huggingface_hub import hf_hub_download

app = typer.Typer(help="Download a tiny AISHELL-1 subset for local ASR smoke evaluation.")

REPO_ID = "AISHELL/AISHELL-1"
TRANSCRIPT_FILE = "data_aishell/transcript/aishell_transcript_v0.8.txt"
SPEAKER_ARCHIVE = "data_aishell/wav/S0002.tar.gz"
SAMPLE_IDS = [
    "BAC009S0002W0122",
    "BAC009S0002W0123",
    "BAC009S0002W0124",
    "BAC009S0002W0125",
    "BAC009S0002W0126",
]


@app.command()
def run(output_dir: Path = Path("data/datasets/aishell_smoke")) -> None:
    downloads_dir = output_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=TRANSCRIPT_FILE,
            local_dir=downloads_dir,
        )
    )
    archive_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=SPEAKER_ARCHIVE,
            local_dir=downloads_dir,
        )
    )

    wanted_members = {f"train/S0002/{sample_id}.wav" for sample_id in SAMPLE_IDS}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.name in wanted_members]
        archive.extractall(output_dir / "wav", members=members, filter="data")

    transcript_by_id: dict[str, str] = {}
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        sample_id, reference = line.split(maxsplit=1)
        if sample_id in SAMPLE_IDS:
            transcript_by_id[sample_id] = reference.replace(" ", "")

    for sample_id in SAMPLE_IDS:
        wav_path = output_dir / "wav" / "train" / "S0002" / f"{sample_id}.wav"
        reference = transcript_by_id.get(sample_id, "")
        typer.echo(f"{sample_id}\t{wav_path}\t{reference}")


if __name__ == "__main__":
    app()

