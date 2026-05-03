from __future__ import annotations

from pathlib import Path
import re


def load_lines(path: str | Path) -> list[str]:
    items: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            items.append(value)
    return items


def load_corrections(path: str | Path) -> dict[str, str]:
    corrections: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        wrong, correct = value.split("\t", maxsplit=1)
        corrections[wrong] = correct
    return corrections


def apply_corrections(text: str, corrections: dict[str, str]) -> str:
    result = text
    for wrong, correct in corrections.items():
        result = result.replace(wrong, correct)
    return result


def strip_sensevoice_tags(text: str) -> str:
    return re.sub(r"<\|[^|]+?\|>", "", text).strip()


def flag_blacklisted_terms(text: str, blacklist: list[str]) -> list[str]:
    return [term for term in blacklist if term in text]
