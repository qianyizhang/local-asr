from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from local_speech_lab.components import JudgmentEvaluator
from local_speech_lab.schemas import TranscriptJudgment


def build_semantic_judge_prompt(hypothesis: str, reference: str) -> str:
    return f"""You are judging an ASR transcript.

Reference transcript:
{reference}

ASR transcript:
{hypothesis}

Evaluate whether the ASR transcript preserves the reference meaning and would be useful to a human
who needs the spoken content. Ignore harmless punctuation, spacing, casing, and wording differences.
Penalize changed facts, missing entities, wrong numbers, wrong negation, or clinically/business
important omissions.

Return only JSON with these keys:
semantic_equivalent: boolean
useful: boolean
score: number from 0.0 to 1.0
reason: short string
"""


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise
        loaded = json.loads(match.group(0))
    if not isinstance(loaded, dict):
        raise ValueError("Judge response must be a JSON object.")
    return loaded


def _judgment_from_payload(
    provider: str,
    model: str,
    payload: dict[str, Any],
    raw_response: str,
) -> TranscriptJudgment:
    score = payload.get("score")
    return TranscriptJudgment(
        provider=provider,
        model=model,
        semantic_equivalent=bool(payload.get("semantic_equivalent")),
        useful=bool(payload.get("useful")),
        score=float(score) if score is not None else None,
        reason=str(payload.get("reason", "")),
        raw_response=raw_response,
    )


class TransformersSemanticJudge(JudgmentEvaluator):
    def __init__(
        self,
        provider: str,
        model_path: Path = Path("models/qwen3_0_6b"),
        max_new_tokens: int = 256,
    ) -> None:
        self.provider = provider
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is None or self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
            )
        return self._tokenizer, self._model

    def judge(self, hypothesis: str, reference: str) -> TranscriptJudgment:
        try:
            tokenizer, model = self._load()
            prompt = build_semantic_judge_prompt(hypothesis, reference)
            messages = [{"role": "user", "content": prompt + "\n/no_think"}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            model_inputs = tokenizer([text], return_tensors="pt")
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
            raw_response = tokenizer.decode(output_ids, skip_special_tokens=True)
            payload = _extract_json_object(raw_response)
            return _judgment_from_payload(self.provider, str(self.model_path), payload, raw_response)
        except Exception as exc:
            return TranscriptJudgment(
                provider=self.provider,
                model=str(self.model_path),
                error=f"{type(exc).__name__}: {exc}",
            )


class LlamaCppCliSemanticJudge(JudgmentEvaluator):
    def __init__(
        self,
        provider: str,
        model_path: Path,
        executable: str = "llama-cli",
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        self.provider = provider
        self.model_path = model_path
        self.executable = executable
        self.max_tokens = max_tokens
        self.temperature = temperature

    def judge(self, hypothesis: str, reference: str) -> TranscriptJudgment:
        prompt = build_semantic_judge_prompt(hypothesis, reference)
        try:
            process = subprocess.run(
                [
                    self.executable,
                    "-m",
                    str(self.model_path),
                    "-p",
                    prompt,
                    "-n",
                    str(self.max_tokens),
                    "--temp",
                    str(self.temperature),
                    "--no-display-prompt",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            raw_response = process.stdout.strip()
            payload = _extract_json_object(raw_response)
            return _judgment_from_payload(self.provider, str(self.model_path), payload, raw_response)
        except Exception as exc:
            return TranscriptJudgment(
                provider=self.provider,
                model=str(self.model_path),
                error=f"{type(exc).__name__}: {exc}",
            )
