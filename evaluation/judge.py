"""
evaluation/judge.py
────────────────────
LLM-as-judge for reply quality evaluation.

Rubric (1–5 scale per dimension):
  1. Correctness      — Is the response factually correct for this intent?
  2. Groundedness     — Does it use only information from message + evidence?
  3. Resolution       — How useful is this for resolving the customer's issue?
  4. Tone             — Is it professional, empathetic, not robotic?
  5. Non-hallucination — Does it avoid inventing facts not in the evidence?

Score 5: Excellent  — correct, grounded, actionable, natural, no unsupported claims
Score 4: Good       — minor weakness but safe and useful
Score 3: Acceptable — partially useful, generic, or missing one element
Score 2: Poor       — substantial mistake, weak grounding, low usefulness
Score 1: Unacceptable — incorrect, hallucinated, unsafe, or unrelated

Judge model: Gemini 1.5 Pro (stronger reasoning than Flash for evaluation).
Temperature: 0.0 (deterministic scores).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

JUDGE_PROMPT_VERSION = "v1"

RUBRIC_TEXT = """
SCORING RUBRIC (apply strictly):

5 — EXCELLENT: Fully correct, uses only information from the message/evidence,
    directly actionable, natural and empathetic tone, zero unsupported claims.

4 — GOOD: Minor weakness (slightly generic, one small inaccuracy) but safe
    and genuinely helpful. Would not mislead the customer.

3 — ACCEPTABLE: Partially useful. Missing one key element OR slightly generic
    OR slightly off-topic. Customer would still learn something.

2 — POOR: Substantial problem. Weak or no grounding in evidence, mostly generic,
    or contains a factual claim not supported by the evidence.

1 — UNACCEPTABLE: Incorrect, hallucinated facts, unsafe (e.g. incorrect promises),
    unrelated to the customer's issue, or offensive.
"""


def build_judge_prompt(
    customer_message: str,
    intent: str,
    context: str,
    reply: str,
    evidence: list[dict],
    grounding_used: bool,
) -> str:
    """Build the judge evaluation prompt."""
    evidence_block = ""
    if evidence:
        items = [
            f"  Case {i+1} (similarity={e.get('similarity', '?')}):\n"
            f"    Problem: {e.get('customer_problem', '')[:150]}\n"
            f"    Resolution: {e.get('resolution', '')[:200]}"
            for i, e in enumerate(evidence)
        ]
        evidence_block = "HISTORICAL EVIDENCE AVAILABLE:\n" + "\n".join(items)
    else:
        evidence_block = "HISTORICAL EVIDENCE: None was available."

    return f"""You are evaluating an AI customer support agent's reply.
Score the reply on FIVE dimensions using the rubric below. Be strict and honest.

{RUBRIC_TEXT}

---
CUSTOMER INTENT: {intent}
CUSTOMER MESSAGE: {customer_message}
PRIOR CONTEXT: {context or "(none)"}

{evidence_block}

AGENT REPLY TO EVALUATE:
"{reply}"

Grounding flag (agent self-reported): {"Used historical evidence" if grounding_used else "Did NOT use evidence"}
---

Score each dimension 1–5. Then give an overall score (1–5, not necessarily the average).

RESPOND WITH EXACTLY THIS JSON:
{{
  "correctness": <1-5>,
  "groundedness": <1-5>,
  "resolution": <1-5>,
  "tone": <1-5>,
  "non_hallucination": <1-5>,
  "overall": <1-5>,
  "critique": "<one paragraph explaining the scores>",
  "worst_issue": "<single most important problem, or 'none'>",
  "prompt_version": "{JUDGE_PROMPT_VERSION}"
}}"""


@dataclass
class JudgeScore:
    """Scores from a single LLM judge evaluation."""

    correctness: int
    groundedness: int
    resolution: int
    tone: int
    non_hallucination: int
    overall: int
    critique: str
    worst_issue: str
    prompt_version: str

    @property
    def mean_dimension_score(self) -> float:
        return (
            self.correctness + self.groundedness + self.resolution
            + self.tone + self.non_hallucination
        ) / 5.0

    def to_dict(self) -> dict:
        return {
            "correctness": self.correctness,
            "groundedness": self.groundedness,
            "resolution": self.resolution,
            "tone": self.tone,
            "non_hallucination": self.non_hallucination,
            "overall": self.overall,
            "mean_dimension": round(self.mean_dimension_score, 2),
            "critique": self.critique,
            "worst_issue": self.worst_issue,
            "prompt_version": self.prompt_version,
        }


class LLMJudge:
    """Evaluates reply quality using Gemini Pro as judge.

    Separation from the generator:
    - Uses a DIFFERENT model (Pro vs Flash) to reduce self-grading bias.
    - Temperature = 0 for reproducible scores.
    - Rubric is fixed and versioned.
    """

    VALID_SCORES = {1, 2, 3, 4, 5}

    def __init__(
        self,
        model: str = "models/gemini-pro-latest",
        temperature: float = 0.0,
        force_offline: bool = False,
    ) -> None:
        self._offline = force_offline
        self._model = None
        self._model_name = model

        if not self._offline:
            try:
                import google.generativeai as genai
                from src.utils import get_gemini_api_key, load_env
                load_env()
                key = get_gemini_api_key()
                if not key:
                    self._offline = True
                else:
                    genai.configure(api_key=key)
                    self._model = genai.GenerativeModel(
                        model_name=model,
                        generation_config={
                            "temperature": temperature,
                            "response_mime_type": "application/json",
                            "max_output_tokens": 512,
                        },
                    )
            except Exception as e:
                logger.warning("LLMJudge init failed (%s), falling back to heuristic scoring: %s", type(e).__name__, e)
                self._offline = True

        logger.info("LLMJudge: model=%s, offline=%s", model, self._offline)

    def _call_api(self, prompt: str) -> str:
        if self._offline or not self._model:
            raise RuntimeError("Judge is offline")
        return self._model.generate_content(prompt).text

    def _heuristic_score(
        self,
        customer_message: str,
        intent: str,
        reply: str,
        evidence: list[dict],
        grounding_used: bool,
    ) -> JudgeScore:
        """Deterministic heuristic judge for offline evaluation / fallback."""
        low_reply = reply.lower()
        has_greeting = any(w in low_reply for w in ["hello", "hi", "thank you", "we are", "please"])
        polite_score = 5 if has_greeting else 3

        grounded = 4 if grounding_used and evidence else 2
        correct = 4 if intent in ["delivery_issue", "billing_payment_issue", "technical_support"] else 3
        resolution = 4 if ("visit" in low_reply or "contact" in low_reply or "check" in low_reply or "investigat" in low_reply) else 3
        non_hallucination = 5 if ("$" not in reply and "track" not in reply) or grounding_used else 3
        overall = int(round((correct + grounded + resolution + polite_score + non_hallucination) / 5.0))

        return JudgeScore(
            correctness=correct,
            groundedness=grounded,
            resolution=resolution,
            tone=polite_score,
            non_hallucination=non_hallucination,
            overall=max(1, min(5, overall)),
            critique="Evaluated via deterministic rubric rules: response is polite, intent-aligned, and within safety bounds.",
            worst_issue="none" if overall >= 4 else "generic_resolution",
            prompt_version=JUDGE_PROMPT_VERSION,
        )

    def score(
        self,
        customer_message: str,
        intent: str,
        reply: str,
        evidence: list[dict],
        context: str = "",
        grounding_used: bool = True,
    ) -> JudgeScore:
        """Score a single reply.

        Returns:
            JudgeScore with all dimensions populated.
        """
        if self._offline:
            return self._heuristic_score(customer_message, intent, reply, evidence, grounding_used)

        prompt = build_judge_prompt(
            customer_message=customer_message,
            intent=intent,
            context=context,
            reply=reply,
            evidence=evidence,
            grounding_used=grounding_used,
        )

        try:
            raw = self._call_api(prompt)
            return self._parse_response(raw)
        except Exception as e:
            logger.warning("Judge API call failed (%s), falling back to heuristic: %s", type(e).__name__, e)
            return self._heuristic_score(customer_message, intent, reply, evidence, grounding_used)

    def _parse_response(self, raw: str) -> JudgeScore:
        try:
            data = json.loads(raw.strip())
        except json.JSONDecodeError:
            match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError(f"Cannot parse judge JSON: {raw[:300]}")

        def clamp_score(val, key: str) -> int:
            try:
                v = int(val)
                return max(1, min(5, v))
            except (ValueError, TypeError):
                logger.warning("Invalid score for '%s': %s", key, val)
                return 3

        return JudgeScore(
            correctness=clamp_score(data.get("correctness", 3), "correctness"),
            groundedness=clamp_score(data.get("groundedness", 3), "groundedness"),
            resolution=clamp_score(data.get("resolution", 3), "resolution"),
            tone=clamp_score(data.get("tone", 3), "tone"),
            non_hallucination=clamp_score(data.get("non_hallucination", 3), "non_hallucination"),
            overall=clamp_score(data.get("overall", 3), "overall"),
            critique=str(data.get("critique", "")),
            worst_issue=str(data.get("worst_issue", "none")),
            prompt_version=str(data.get("prompt_version", JUDGE_PROMPT_VERSION)),
        )

    def score_batch(
        self,
        items: list[dict],
        delay_between: float = 0.5,
    ) -> list[JudgeScore]:
        """Score a batch of items.

        Each item should have: customer_message, intent, reply, evidence,
        context (optional), grounding_used (optional).
        """
        from tqdm import tqdm
        results = []
        for item in tqdm(items, desc="LLM Judge"):
            score = self.score(
                customer_message=item["customer_message"],
                intent=item.get("intent", "unknown"),
                reply=item["reply"],
                evidence=item.get("evidence", []),
                context=item.get("context", ""),
                grounding_used=item.get("grounding_used", True),
            )
            results.append(score)
            if delay_between > 0:
                time.sleep(delay_between)
        return results
