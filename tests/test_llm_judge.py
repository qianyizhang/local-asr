from __future__ import annotations

import unittest
from pathlib import Path

from local_speech_lab.llm_judge import TransformersSemanticJudge, _parse_optional_bool
from local_speech_lab.metrics import character_error_rate


QWEN_SMALL_MODEL = Path("models/qwen3_0_6b")


class JudgeParsingTests(unittest.TestCase):
    def test_parse_boolean_strings(self) -> None:
        self.assertIs(_parse_optional_bool("true"), True)
        self.assertIs(_parse_optional_bool("false"), False)
        self.assertIs(_parse_optional_bool(True), True)
        self.assertIs(_parse_optional_bool(False), False)
        self.assertIsNone(_parse_optional_bool(None))


@unittest.skipUnless(
    (QWEN_SMALL_MODEL / "model.safetensors").exists(),
    "Qwen3-0.6B judge model is not downloaded.",
)
class QwenSmallJudgeSemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.judge = TransformersSemanticJudge(provider="qwen", model_path=QWEN_SMALL_MODEL)

    def assert_low_cer_but_wrong(self, reference: str, hypothesis: str) -> None:
        self.assertLess(character_error_rate(hypothesis, reference), 0.3)
        judgment = self.judge.judge(hypothesis=hypothesis, reference=reference)

        self.assertIsNone(judgment.error, judgment.error)
        self.assertIs(judgment.semantic_equivalent, False, judgment.reason)
        self.assertIs(judgment.useful, False, judgment.reason)
        self.assertIsNotNone(judgment.score)
        self.assertLessEqual(judgment.score or 0.0, 0.2, judgment.reason)

    def test_negation_flip_is_not_equivalent(self) -> None:
        self.assert_low_cer_but_wrong(
            reference="没有发现肺部结节",
            hypothesis="发现肺部结节",
        )

    def test_dose_frequency_change_is_not_equivalent(self) -> None:
        self.assert_low_cer_but_wrong(
            reference="患者每日服用二甲双胍一次",
            hypothesis="患者每日服用二甲双胍三次",
        )

    def test_numeric_value_change_is_not_equivalent(self) -> None:
        self.assert_low_cer_but_wrong(
            reference="血糖为七点八",
            hypothesis="血糖为九点八",
        )

    def test_body_part_change_is_not_equivalent(self) -> None:
        self.assert_low_cer_but_wrong(
            reference="建议复查肝功能",
            hypothesis="建议复查肾功能",
        )

    def test_time_of_day_change_is_not_equivalent(self) -> None:
        self.assert_low_cer_but_wrong(
            reference="明天上午九点复诊",
            hypothesis="明天晚上九点复诊",
        )


if __name__ == "__main__":
    unittest.main()

