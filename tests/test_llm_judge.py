from __future__ import annotations

from pathlib import Path

import pytest

from local_speech_lab.llm_judge import TransformersSemanticJudge, _parse_optional_bool
from local_speech_lab.metrics import character_error_rate


QWEN_SMALL_MODEL = Path("models/qwen3_0_6b")


def test_parse_boolean_strings() -> None:
    assert _parse_optional_bool("true") is True
    assert _parse_optional_bool("false") is False
    assert _parse_optional_bool(True) is True
    assert _parse_optional_bool(False) is False
    assert _parse_optional_bool(None) is None


@pytest.fixture(scope="module")
def qwen_small_judge() -> TransformersSemanticJudge:
    if not (QWEN_SMALL_MODEL / "model.safetensors").exists():
        pytest.skip("Qwen3-0.6B judge model is not downloaded.")
    return TransformersSemanticJudge(provider="qwen", model_path=QWEN_SMALL_MODEL)


def assert_low_cer_but_wrong(
    qwen_small_judge: TransformersSemanticJudge,
    reference: str,
    hypothesis: str,
) -> None:
    assert character_error_rate(hypothesis, reference) < 0.3
    judgment = qwen_small_judge.judge(hypothesis=hypothesis, reference=reference)

    assert judgment.error is None, judgment.error
    assert judgment.semantic_equivalent is False, judgment.reason
    assert judgment.useful is False, judgment.reason
    assert judgment.score is not None
    assert judgment.score <= 0.2, judgment.reason


def test_negation_flip_is_not_equivalent(qwen_small_judge: TransformersSemanticJudge) -> None:
    assert_low_cer_but_wrong(
        qwen_small_judge=qwen_small_judge,
        reference="没有发现肺部结节",
        hypothesis="发现肺部结节",
    )


def test_dose_frequency_change_is_not_equivalent(
    qwen_small_judge: TransformersSemanticJudge,
) -> None:
    assert_low_cer_but_wrong(
        qwen_small_judge=qwen_small_judge,
        reference="患者每日服用二甲双胍一次",
        hypothesis="患者每日服用二甲双胍三次",
    )


def test_numeric_value_change_is_not_equivalent(qwen_small_judge: TransformersSemanticJudge) -> None:
    assert_low_cer_but_wrong(
        qwen_small_judge=qwen_small_judge,
        reference="血糖为七点八",
        hypothesis="血糖为九点八",
    )


def test_body_part_change_is_not_equivalent(qwen_small_judge: TransformersSemanticJudge) -> None:
    assert_low_cer_but_wrong(
        qwen_small_judge=qwen_small_judge,
        reference="建议复查肝功能",
        hypothesis="建议复查肾功能",
    )


def test_time_of_day_change_is_not_equivalent(qwen_small_judge: TransformersSemanticJudge) -> None:
    assert_low_cer_but_wrong(
        qwen_small_judge=qwen_small_judge,
        reference="明天上午九点复诊",
        hypothesis="明天晚上九点复诊",
    )
