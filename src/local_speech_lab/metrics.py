from __future__ import annotations


def _edit_distance(source: list[str], target: list[str]) -> int:
    previous = list(range(len(target) + 1))
    for source_index, source_item in enumerate(source, start=1):
        current = [source_index]
        for target_index, target_item in enumerate(target, start=1):
            substitution = previous[target_index - 1] + int(source_item != target_item)
            insertion = current[target_index - 1] + 1
            deletion = previous[target_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def character_error_rate(hypothesis: str, reference: str) -> float:
    reference_units = list(reference)
    if not reference_units:
        return 0.0 if not hypothesis else 1.0
    return _edit_distance(list(hypothesis), reference_units) / len(reference_units)


def word_error_rate(hypothesis: str, reference: str) -> float:
    reference_units = reference.split()
    if not reference_units:
        return 0.0 if not hypothesis.split() else 1.0
    return _edit_distance(hypothesis.split(), reference_units) / len(reference_units)


def transcript_metrics(hypothesis: str, reference: str) -> dict[str, float]:
    return {
        "cer": character_error_rate(hypothesis, reference),
        "wer": word_error_rate(hypothesis, reference),
    }

